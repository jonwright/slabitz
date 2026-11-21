#!/usr/bin/env python3
"""GPU-vs-CPU comparison for the FSE-index PropagateFseIndex kernel.

Builds nothing; expects a prebuilt cubin (tools/build_all.py -> build/pfi/pfi.cubin).
Runs cross-block decoupled-lookback propagation over sequence-stream FseId arrays
and compares the propagated result to kernels/cpu_ref_propagatefse.py.

Usage:
    PYTHONPATH=$PWD python3 tests/run_propagatefse_chain.py build/pfi/pfi.cubin
"""
import ctypes as ct
import struct
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from slabitz import loader
from kernels.cpu_ref_propagatefse import ref_propagate_fse_index, UNUSED, REPEAT

CUBIN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build/pfi/pfi.cubin")
BLOCK = 32


class Buf(ct.Structure):
    _fields_ = [("data", ct.c_uint64), ("count", ct.c_uint64)]


def dm(drv, src):
    m = loader.DeviceMemory(drv, len(src) * 4)
    m.from_host(struct.pack(f"{len(src)}I", *src))
    return m


def run_gpu(drv, fn, fse_ids):
    n = len(fse_ids)
    nb = (n + BLOCK - 1) // BLOCK
    idM = dm(drv, fse_ids)
    lkM = dm(drv, [0] * nb)
    daM = dm(drv, [0] * 64)  # dispatch args (not read with positive workItemCount)
    args = [Buf(idM.ptr, n), Buf(lkM.ptr, nb), Buf(daM.ptr, 64),
            ct.c_uint32(0), ct.c_uint32(n)]
    fn(grid=(nb, 1, 1), block=(BLOCK, 1, 1), args=args)
    drv.synchronize()
    return list(struct.unpack(f"{n}I", idM.to_host(n * 4)))


def check(name, fse_ids):
    cpu = ref_propagate_fse_index(list(fse_ids), len(fse_ids))
    gpu = run_gpu(drv, fn, fse_ids)
    bad = [(j, gpu[j], cpu[j]) for j in range(len(fse_ids)) if gpu[j] != cpu[j]]
    print(f"  {name}: {'PASS' if not bad else 'FAIL'} ({len(bad)} mismatch)")
    if bad:
        print("    first few:", bad[:6])
        print("    first block cpu:", cpu[:BLOCK])
        print("    first block gpu:", gpu[:BLOCK])
    return not bad


drv = loader.Driver(0)
fn = drv.load(open(CUBIN, "rb").read()).function("main_0")

cases = [
    ("valid-only (no repeat)", [5] * 64),
    ("unused-only", [UNUSED] * 64),
    ("repeat chain -> resolve from valid", [5] * 32 + [UNUSED, 7, UNUSED, 7] * 8 + [REPEAT] * 32),
    ("repeat mid-block with trailing repeat", [3] * 32 + [REPEAT] * 10 + [9] * 22 + [REPEAT] * 32),
    ("three-block mixed", [1] * 16 + [2] * 16 + [REPEAT] * 8 + [4] * 24 + [REPEAT] * 32),
]

all_ok = True
for name, ids in cases:
    all_ok &= check(name, ids)
drv.destroy()
print("OVERALL:", "PASS" if all_ok else "FAIL")
sys.exit(0 if all_ok else 1)
