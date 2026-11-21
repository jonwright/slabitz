#!/usr/bin/env python3
"""GPU-vs-CPU comparison for the PrefixSum kernel (decoupled-lookback prefix sum).

Builds nothing; expects a prebuilt cubin (tools/build_all.py -> build/ps/ps.cubin).
The decoupled-lookback prefix sum must equal the mathematical oracle: exclusive
prefix sum of `count` (+ count[i] when outputInclusive). Verified over multiple
waves (n a multiple of 32) for both exclusive and inclusive output.

Usage:
    PYTHONPATH=$PWD python3 tests/run_prefixsum_chain.py build/ps/ps.cubin
"""
import ctypes as ct
import random
import struct
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from slabitz import loader

CUBIN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build/ps/ps.cubin")
BLOCK = 32


class Buf(ct.Structure):
    _fields_ = [("data", ct.c_uint64), ("count", ct.c_uint64)]


def dm(drv, src):
    m = loader.DeviceMemory(drv, len(src) * 4)
    m.from_host(struct.pack(f"{len(src)}I", *src))
    return m


def cpu_ref(counts, inclusive):
    """Mathematical oracle: in[i] = exclusive prefix sum (+count[i] if inclusive)."""
    out = []
    acc = 0
    for c in counts:
        out.append(acc + (c if inclusive else 0))
        acc += c
    return out


def run_gpu(drv, fn, counts, inclusive):
    n = len(counts)
    nb = (n + BLOCK - 1) // BLOCK
    idM = dm(drv, counts)          # in-place: counts -> prefix
    lkM = dm(drv, [0] * nb)        # lookback block cells, zero-initialised
    daM = dm(drv, [0] * 64)
    args = [Buf(idM.ptr, n), Buf(lkM.ptr, nb), Buf(daM.ptr, 64),
            ct.c_uint32(0), ct.c_uint32(n), ct.c_uint32(inclusive)]
    fn(grid=(nb, 1, 1), block=(BLOCK, 1, 1), args=args)
    drv.synchronize()
    return list(struct.unpack(f"{n}I", idM.to_host(n * 4)))


def main():
    random.seed(7)
    drv = loader.Driver(0)
    fn = drv.load(open(CUBIN, "rb").read()).function("main_0")

    cases = []
    for n in (32, 64, 96, 128):
        for inclusive in (0, 1):
            counts = [random.randrange(1, 5) for _ in range(n)]
            cases.append((f"n={n} inclusive={inclusive}", counts, inclusive))

    all_ok = True
    for name, counts, inclusive in cases:
        cpu = cpu_ref(counts, inclusive)
        gpu = run_gpu(drv, fn, counts, inclusive)
        bad = [(j, gpu[j], cpu[j]) for j in range(len(cpu)) if gpu[j] != cpu[j]]
        print(f"  {name}: {'PASS' if not bad else 'FAIL'} ({len(bad)} mismatch)")
        if bad:
            print("    first few:", bad[:6])
        all_ok &= (not bad)

    drv.destroy()
    print("OVERALL:", "PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
