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
import kparams  # tests/, next to this script

CUBIN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build/pfi/pfi.cubin")
BLOCK = 32


class Buf(ct.Structure):
    _fields_ = [("data", ct.c_uint64), ("count", ct.c_uint64)]


def dm(drv, src):
    m = loader.DeviceMemory(drv, len(src) * 4)
    m.from_host(struct.pack(f"{len(src)}I", *src))
    return m


SEQ_STREAMS = 8  # index of Seq_Streams in zstdgpu_Counters (kernels/cpu_ref_initresources.py)


def run_gpu(drv, fn, fse_ids, device_count=False):
    """device_count: workItemCount ~0u, the count read from Counters.Seq_Streams (P4 step 2b,
    as the C library launches it)."""
    n = len(fse_ids)
    nb = (n + BLOCK - 1) // BLOCK
    idM = dm(drv, fse_ids)
    lkM = dm(drv, [0] * nb)
    daM = dm(drv, [0] * 64)  # dispatch args (not read with positive workItemCount)
    ctr = [0] * 22
    ctr[SEQ_STREAMS] = n
    cM = dm(drv, ctr)        # one zstdgpu_Counters (22 x u32 = 88 bytes)
    args = [Buf(idM.ptr, n), Buf(lkM.ptr, nb), Buf(daM.ptr, 64), Buf(cM.ptr, 1),
            ct.c_uint32(0), ct.c_uint32(0xFFFFFFFF if device_count else n)]
    kparams.check_args("PropagateFseIndex", args)
    fn(grid=(nb, 1, 1), block=(BLOCK, 1, 1), args=args)
    drv.synchronize()
    return list(struct.unpack(f"{n}I", idM.to_host(n * 4)))


compared = 0


def check(name, fse_ids, device_count=False):
    global compared
    cpu = ref_propagate_fse_index(list(fse_ids), len(fse_ids))
    gpu = run_gpu(drv, fn, fse_ids, device_count)
    bad = [(j, gpu[j], cpu[j]) for j in range(len(fse_ids)) if gpu[j] != cpu[j]]
    compared += len(fse_ids)
    if device_count:
        name += " (count from Counters)"
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
for dcount in (False, True):
    for name, ids in cases:
        all_ok &= check(name, ids, dcount)
drv.destroy()
print("compared %d FSE ids" % (compared if all_ok else 0))
print("OVERALL:", "PASS" if all_ok else "FAIL")
sys.exit(0 if all_ok and compared > 0 else 1)
