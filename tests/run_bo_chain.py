#!/usr/bin/env python3
"""GPU-vs-CPU comparison for the ComputeDestBlockOffsets kernel.

Builds nothing; expects a prebuilt cubin (tools/build_all.py -> build/bo/bo.cubin).
Synthetic multi-frame block layout; runs the GPU cubin (pure-CUDA, {data,count}
ABI) and compares the per-block dest offsets to kernels/cpu_ref_bo.py.

Usage:
    PYTHONPATH=$PWD python3 tests/run_bo_chain.py build/bo/bo.cubin
"""
import ctypes as ct
import struct
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from slabitz import loader
import kparams  # tests/kparams.py
from kernels.cpu_ref_bo import ref_frame_bounds_status

CUBIN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build/bo/bo.cubin")


class Buf(ct.Structure):
    _fields_ = [("data", ct.c_uint64), ("count", ct.c_uint64)]


def dm(drv, lst, nbytes=4):
    m = loader.DeviceMemory(drv, len(lst) * nbytes)
    m.from_host(struct.pack(f"{len(lst)}{'Q' if nbytes == 8 else 'i'}", *lst))
    return m


def run_gpu(drv, fn, block_size_prefix, per_frame_block_count_all, frame_refs,
            work_item_count, frame_count):
    nblocks = len(block_size_prefix)
    pM = dm(drv, block_size_prefix)
    fM = dm(drv, per_frame_block_count_all)
    rM = dm(drv, [lo | (hi << 32) for lo, hi in frame_refs], 8)
    oM = loader.DeviceMemory(drv, work_item_count * 4)
    sM = dm(drv, [0] * frame_count)
    args = [Buf(oM.ptr, work_item_count), Buf(pM.ptr, nblocks), Buf(fM.ptr, len(per_frame_block_count_all)),
            Buf(rM.ptr, len(frame_refs)), Buf(sM.ptr, frame_count),
            ct.c_uint32(0), ct.c_uint32(work_item_count), ct.c_uint32(frame_count)]
    ngroups = (work_item_count + 127) // 128
    kparams.check_args("ComputeDestBlockOffsets", args)
    fn(grid=(ngroups, 1, 1), block=(128, 1, 1), args=args)
    drv.synchronize()
    dests = list(struct.unpack(f"{work_item_count}I", oM.to_host(work_item_count * 4)))
    st = struct.unpack(f"{frame_count}I", sM.to_host(frame_count * 4))
    return dests, {i: s for i, s in enumerate(st) if s}


def main():
    with open(CUBIN, "rb") as f:
        cubin = f.read()
    drv = loader.Driver(0)
    fn = drv.load(cubin).function("main_0")

    ok = True
    n = 0
    for name, sizes, frame_refs in CASES:
        good, cnt = check(drv, fn, name, sizes, frame_refs)
        ok = ok and good
        n += cnt
    drv.destroy()
    print("compared %d block offsets" % (n if ok else 0))
    sys.exit(0 if ok and n > 0 else 1)


# 3 frames. Frame 0: 2 blocks (4,6). Frame 1: 3 blocks (5,7,8). Frame 2: 1 block (10).
# frame base dest offsets (into the output): frame0 -> 0, frame1 -> 100, frame2 -> 300
# T09 cases: frame 1 declares 15 bytes (its third block ends past it), then 25
# (its last block ends short of it); frame 2 has one block of 128 KiB + 1.
CASES = [
    ("fit", [4, 6, 5, 7, 8, 10], [(0, 10), (100, 20), (300, 10)]),
    ("past", [4, 6, 5, 7, 8, 10], [(0, 10), (100, 15), (300, 10)]),
    ("short", [4, 6, 5, 7, 8, 10], [(0, 10), (100, 25), (300, 10)]),
    ("too-large", [4, 6, 5, 7, 8, 131073], [(0, 10), (100, 20), (300, 131073)]),
]


def check(drv, fn, name, sizes, frame_refs):
    # DS blockSizePrefix is cumulative-INCLUSIVE: prefix[k] = sum(sizes[0..k]); block k
    # occupies [prefix[k-1], prefix[k]). blockOffset = prefix[blockIdx-1].
    prefix = []
    acc = 0
    for s in sizes:
        acc += s
        prefix.append(acc)
    total_blocks = len(sizes)
    per_frame = [0, 2, 5, total_blocks]  # cumulative block count at each frame start
    frame_count = 3
    work_item_count = total_blocks
    # "fit": expected dest = [0, 4, 100, 105, 112, 300]

    gpu, gst = run_gpu(drv, fn, prefix, per_frame, frame_refs, work_item_count, frame_count)
    cpu, cst = ref_frame_bounds_status(prefix, per_frame, frame_refs, work_item_count, frame_count)
    mism = [(j, gpu[j], cpu[j]) for j in range(work_item_count) if gpu[j] != cpu[j]]
    good = not mism and gst == cst and (name == "fit") == (not cst)
    print(f"{'PASS' if good else 'FAIL'}: ComputeDestBlockOffsets {name} ({work_item_count} blocks, "
          f"{len(mism)} mismatch, status gpu {gst} cpu {cst})")
    if mism:
        print("   mismatches:", mism)
        print("   gpu:", gpu)
        print("   cpu:", cpu)
    return good, work_item_count


if __name__ == "__main__":
    main()
