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
from kernels.cpu_ref_bo import ref_compute_dest_block_offsets

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
    args = [Buf(oM.ptr, work_item_count), Buf(pM.ptr, nblocks), Buf(fM.ptr, len(per_frame_block_count_all)),
            Buf(rM.ptr, len(frame_refs)),
            ct.c_uint32(0), ct.c_uint32(work_item_count), ct.c_uint32(frame_count)]
    ngroups = (work_item_count + 127) // 128
    fn(grid=(ngroups, 1, 1), block=(128, 1, 1), args=args)
    drv.synchronize()
    return [x for x in struct.unpack(f"{work_item_count}i", oM.to_host(work_item_count * 4))]


def main():
    with open(CUBIN, "rb") as f:
        cubin = f.read()
    drv = loader.Driver(0)
    fn = drv.load(cubin).function("main_0")

    # 3 frames. Frame 0: 2 blocks (4,6). Frame 1: 3 blocks (5,7,8). Frame 2: 1 block (10).
    sizes = [4, 6, 5, 7, 8, 10]
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
    # frame base dest offsets (into the output): frame0 -> 0, frame1 -> 100, frame2 -> 300
    frame_refs = [(0, 10), (100, 20), (300, 10)]
    work_item_count = total_blocks
    # expected dest = [0, 4, 100, 105, 112, 300]

    gpu = run_gpu(drv, fn, prefix, per_frame, frame_refs, work_item_count, frame_count)
    cpu = ref_compute_dest_block_offsets(prefix, per_frame, frame_refs, work_item_count, frame_count)
    mism = [(j, gpu[j], cpu[j]) for j in range(work_item_count) if gpu[j] != cpu[j]]
    status = "PASS" if not mism else "FAIL"
    print(f"{status}: ComputeDestBlockOffsets ({work_item_count} blocks, {len(mism)} mismatch)")
    if mism:
        print("   mismatches:", mism)
        print("   gpu:", gpu)
        print("   cpu:", cpu)
    drv.destroy()
    sys.exit(0 if not mism else 1)


if __name__ == "__main__":
    main()
