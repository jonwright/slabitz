#!/usr/bin/env python3
"""GPU-vs-CPU comparison for the InitResources kernel.

Builds nothing; expects a prebuilt cubin (tools/build_all.py -> build/ir/ir.cubin).
Runs both stage 0 (counter init) and stage 1 (default FSE tables + RLE entries +
default-prob copies) and compares full buffers to kernels/cpu_ref_initresources.py.

Usage:
    PYTHONPATH=$PWD python3 tests/run_initresources_chain.py build/ir/ir.cubin
"""
import ctypes as ct
import struct
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from slabitz import loader
import kparams  # tests/kparams.py
from kernels.cpu_ref_initresources import ref_init_resources, _FIELD_ORDER

CUBIN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build/ir/ir.cubin")
CMP = 2
FSE_INFOS_SIZE = 258 + 3 * CMP + 8
FSE_ELEMS_SIZE = 256
FSE_PROBS_SIZE = CMP * 192 + 192
DEFAULT_SIZE = 118
MARKER = 0xDEADBEEF


class Buf(ct.Structure):
    _fields_ = [("data", ct.c_uint64), ("count", ct.c_uint64)]


def dm(drv, src, nbytes=4):
    m = loader.DeviceMemory(drv, len(src) * nbytes)
    m.from_host(struct.pack(f'{len(src)}' + ('Q' if nbytes == 8 else 'I'), *src))
    return m


def to_host(m, count):
    return list(struct.unpack(f'{count}I', m.to_host(count * 4)))


def run_gpu(drv, fn, stage, counters, fse_probs_default, fse_infos, fse_elems, fse_probs):
    cM = dm(drv, counters)
    dM = dm(drv, fse_probs_default)
    iM = dm(drv, fse_infos)
    eM = dm(drv, fse_elems)
    pM = dm(drv, fse_probs)
    args = [Buf(cM.ptr, len(counters)), Buf(dM.ptr, len(fse_probs_default)),
            Buf(pM.ptr, len(fse_probs)), Buf(iM.ptr, len(fse_infos)),
            Buf(eM.ptr, len(fse_elems)), ct.c_uint32(stage)]
    ngroups = 1 if stage == 0 else (256 + 31) // 32
    kparams.check_args("InitResources", args)
    fn(grid=(ngroups, 1, 1), block=(32, 1, 1), args=args)
    drv.synchronize()
    return (to_host(cM, len(counters)), to_host(iM, len(fse_infos)),
            to_host(eM, len(fse_elems)), to_host(pM, len(fse_probs)))


compared = 0


def check(name, cpu, gpu):
    global compared
    compared += len(cpu)
    bad = [(j, gpu[j], cpu[j]) for j in range(len(cpu)) if gpu[j] != cpu[j]]
    print(f"  {name}: {'PASS' if not bad else 'FAIL'} ({len(bad)} mismatch)")
    if bad:
        print("    first few:", bad[:5])
    return not bad


def main():
    drv = loader.Driver(0)
    fn = drv.load(open(CUBIN, "rb").read()).function("main_0")

    counters = [MARKER] * 22
    counters[_FIELD_ORDER.index("Blocks_CMP")] = CMP
    # Force a few nonzero-ish markers so stage 0's zeroing is visibly tested,
    # and so stage 1 (which only reads Blocks_CMP) reads a known cmp.
    default = [i * 7 % 100 + 1 for i in range(DEFAULT_SIZE)]
    fse_infos = [MARKER] * FSE_INFOS_SIZE
    fse_elems = [MARKER] * FSE_ELEMS_SIZE
    fse_probs = [MARKER] * FSE_PROBS_SIZE

    all_ok = True
    for stage in (0, 1):
        print(f"stage {stage}:")
        c0, i0, e0, p0 = ref_init_resources(counters, default, fse_infos, fse_elems, fse_probs, stage)
        cg, ig, eg, pg = run_gpu(drv, fn, stage, counters, default, fse_infos, fse_elems, fse_probs)
        all_ok &= check("counters", c0, cg)
        all_ok &= check("fse_infos", i0, ig)
        all_ok &= check("fse_elems", e0, eg)
        all_ok &= check("fse_probs", p0, pg)

    drv.destroy()
    print("compared %d words" % (compared if all_ok else 0))
    print("OVERALL:", "PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok and compared > 0 else 1)


if __name__ == "__main__":
    main()
