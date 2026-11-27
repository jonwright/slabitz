"""RETIRED (T01): checks only that 64 elements were written; compare_initfsetable.py
launches the same kernel and compares every table element with kernels/cpu_ref_ifs.py.

First-launch smoke test for kernels/InitFseTable.slang — pure CUDA (Driver API).

No torch / no cudart: device memory is allocated with ``cuMemAlloc`` and copied with
``cuMemcpy{DtoH,HtoD}``. This validates the Slang->CUDA integration contract that any
CUDA library (including bslz4_to_sparse's ctypes driver) must reproduce:

  * the cubin exports the entry as symbol ``main_0`` (Slang suffixes the name);
  * each `StructuredBuffer<T>`/`RWStructuredBuffer<T>` entry param is passed BY
    VALUE as ``{T* data; size_t count}`` (pointer + element count), NOT a raw
    device pointer (see slang-cuda-prelude.h:2915).  ctypes must pack both;
  * six trailing scalar ``u32`` params (tgOffset, workItemCount, tableType,
    tableStartIndex, tableDataStart, tableDataCount);
  * ``groupshared Lds`` is 803 dwords = 3212 bytes, passed as ``shared``.

Usage::

    python3 tests/smoke_initfsetable.py [path/to/InitFseTable.cubin]

Build the cubin first with tools/build_all.py (-> build/ifs/ifs.cubin, the default).
"""

import ctypes as ct
import struct
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from slabitz import loader
import kparams  # tests/kparams.py

CUBIN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build/ifs/ifs.cubin")

# Slang StructuredBuffer by-value arg layout: {T* data; size_t count}.
class _Buf(ct.Structure):
    _fields_ = [("data", ct.c_uint64), ("count", ct.c_uint64)]


def _i32(src):
    """Raw little-endian int32 bytes -> ctypes c_int32 list."""
    n = len(src) // 4
    return list(struct.unpack(f"{n}i", src))


def main():
    with open(CUBIN, "rb") as f:
        cubin = f.read()

    drv = loader.Driver(0)
    fn = drv.load(cubin).function("main_0")

    # Single-symbol LLen table: accuracyLog 6 (table of 64), one symbol freq 64.
    accuracy, probcount = 6, 1
    fse_info = ((accuracy << 8) | probcount) & 0xFFFFFFFF

    probs = loader.DeviceMemory(drv, 64 * 4)         # int32[64]
    infos = loader.DeviceMemory(drv, 1024 * 4)       # int32[1024], [256 + idx]
    counters = loader.DeviceMemory(drv, 16 * 4)      # int32[16]
    dargs = loader.DeviceMemory(drv, 16 * 4)         # int32[16]
    out = loader.DeviceMemory(drv, 4096 * 4)         # uint32/int32[4096]

    probs.from_host(struct.pack("64i", *([0] * 63), 64))
    fse = [0] * 1024
    fse[256] = fse_info                     # tableIndex = tableStartIndex(0)+groupId(0)
    infos.from_host(struct.pack("1024i", *fse))
    counters.from_host(struct.pack("16i", 1, *([0] * 15)))   # Blocks_CMP
    dargs.from_host(struct.pack("16i", *([0] * 16)))
    out.from_host(struct.pack("4096i", *([-12345678] * 4096)))  # sentinel

    args = [
        _Buf(probs.ptr, 64),                # inFseProbs
        _Buf(out.ptr, 4096),                # inoutFseElems (output DTable)
        _Buf(infos.ptr, 1024),              # inFseInfos
        _Buf(counters.ptr, 16),             # inCounters
        _Buf(dargs.ptr, 16),                # inDispatchArgs
        ct.c_uint32(0),                     # tgOffset
        ct.c_uint32(1),                     # workItemCount
        ct.c_uint32(1),                     # tableType: 1 = LLen
        ct.c_uint32(0),                     # tableStartIndex
        ct.c_uint32(0),                     # tableDataStart
        ct.c_uint32(64),                    # tableDataCount
    ]

    kparams.check_args("InitFseTable", args)
    fn(grid=(1, 1, 1), block=(128, 1, 1), args=args, shared=3212)
    drv.synchronize()

    head = _i32(out.to_host(64 * 4))
    written = sum(1 for v in head if v != -12345678)
    print(f"elements[0:64] written: {written}/64")
    print("first 16:", head[:16])
    if written == 0:
        print("SMOKE FAIL: kernel wrote nothing")
        sys.exit(1)
    print("SMOKE PASS")


if __name__ == "__main__":
    main()
