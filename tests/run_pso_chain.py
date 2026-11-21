#!/usr/bin/env python3
"""GPU-vs-CPU comparison for the PrefixSequenceOffsets kernel.

Builds nothing; expects a prebuilt cubin (tools/build_all.py -> build/pso/pso.cubin).
Two-frames / two-block layout of seq streams; exercises isFirst-in-frame default
repeat decode, intra-block propagation, and cross-block lookback. Full 32-lane
blocks (Seq_Streams a multiple of 32).

Usage:
    PYTHONPATH=$PWD python3 tests/run_pso_chain.py build/pso/pso.cubin
"""
import ctypes as ct
import struct
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from slabitz import loader
from kernels.cpu_ref_prefix_seq_offsets import ref_prefix_sequence_offsets, ENC, REP_BASE
from kernels.cpu_ref_initresources import _FIELD_ORDER

CUBIN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build/pso/pso.cubin")
BLOCK = 32


class Buf(ct.Structure):
    _fields_ = [("data", ct.c_uint64), ("count", ct.c_uint64)]


def dm(drv, src):
    m = loader.DeviceMemory(drv, len(src) * 4)
    m.from_host(struct.pack(f"{len(src)}I", *src))
    return m


def rep_enc(idx, cnt):
    """Encode a repeat-sequence offset (repeat index idx, subtracted-bytes cnt)."""
    return ENC | (idx << REP_BASE) | ((~cnt) & 0x07FFFFFF)


def run_gpu(drv, fn, fin1, fin2, fin3, seq_to_block, per_frame_blocks, per_frame_min, seq_streams, frame_count):
    n = len(fin1)
    nb = (n + BLOCK - 1) // BLOCK
    f1M = dm(drv, fin1); f2M = dm(drv, fin2); f3M = dm(drv, fin3)
    l1M = dm(drv, [0] * nb); l2M = dm(drv, [0] * nb); l3M = dm(drv, [0] * nb)
    minM = dm(drv, per_frame_min); pfbM = dm(drv, per_frame_blocks); sbM = dm(drv, seq_to_block)
    ctr = [0] * 22; ctr[_FIELD_ORDER.index("Seq_Streams")] = seq_streams
    cM = dm(drv, ctr); daM = dm(drv, [0] * 64)
    args = [Buf(f1M.ptr, n), Buf(f2M.ptr, n), Buf(f3M.ptr, n),
            Buf(l1M.ptr, nb), Buf(l2M.ptr, nb), Buf(l3M.ptr, nb),
            Buf(minM.ptr, len(per_frame_min)), Buf(pfbM.ptr, len(per_frame_blocks)),
            Buf(sbM.ptr, n), Buf(cM.ptr, 22), Buf(daM.ptr, 64),
            ct.c_uint32(0), ct.c_uint32(n), ct.c_uint32(frame_count)]
    fn(grid=(nb, 1, 1), block=(BLOCK, 1, 1), args=args)
    drv.synchronize()
    return (list(struct.unpack(f"{n}I", f1M.to_host(n * 4))),
            list(struct.unpack(f"{n}I", f2M.to_host(n * 4))),
            list(struct.unpack(f"{n}I", f3M.to_host(n * 4))))


def check(name, f1, f2, f3, seq_to_block, per_frame_blocks, per_frame_min, frame_count):
    seq_streams = len(f1)
    c1, c2, c3 = list(f1), list(f2), list(f3)
    ref_prefix_sequence_offsets(c1, c2, c3, seq_to_block, per_frame_blocks, per_frame_min, frame_count)
    g1, g2, g3 = run_gpu(drv, fn, f1, f2, f3, seq_to_block, per_frame_blocks, per_frame_min, seq_streams, frame_count)
    bad = [(j, (g1[j], g2[j], g3[j]), (c1[j], c2[j], c3[j])) for j in range(len(f1))
           if (g1[j], g2[j], g3[j]) != (c1[j], c2[j], c3[j])]
    print(f"  {name}: {'PASS' if not bad else 'FAIL'} ({len(bad)} mismatch)")
    if bad:
        print("    first few:", bad[:6])
    return not bad


def case_two_frame():
    n = 64
    seq_to_block = [0] * 32 + [1] * 32
    per_frame_blocks = [0, 1, 2]
    per_frame_min = [0, 32]
    fc = 2
    f1 = [0] * n; f2 = [0] * n; f3 = [0] * n
    for i in range(9):
        f1[i], f2[i], f3[i] = 100, 200, 300
    for i in range(9, 32):
        f1[i] = rep_enc(1, 1); f2[i] = rep_enc(2, 2); f3[i] = rep_enc(3, 3)
    f1[32], f2[32], f3[32] = rep_enc(1, 2), rep_enc(2, 3), rep_enc(3, 4)
    for i in range(33, 64):
        f1[i], f2[i], f3[i] = 1000, 2000, 3000
    return n, seq_to_block, per_frame_blocks, per_frame_min, fc, f1, f2, f3


def case_three_frame():
    # 3 compressed blocks, 96 streams (3 full waves). frame0=block0, frame1=block1, frame2=block2.
    n = 96
    seq_to_block = [0] * 32 + [1] * 32 + [2] * 32
    per_frame_blocks = [0, 1, 2, 3]
    per_frame_min = [0, 32, 64]
    fc = 3
    f1 = [0] * n; f2 = [0] * n; f3 = [0] * n
    for i in range(0, 31):  # block0 lanes 0..30 actual (lane31 encoded => branch B)
        f1[i], f2[i], f3[i] = 10, 20, 30
    for i in range(31, 32):
        f1[i], f2[i], f3[i] = rep_enc(1, 1), rep_enc(2, 1), rep_enc(3, 1)
    for i in range(32, 34):  # block1: first two lanes actual, rest encoded -> lookback to block0
        f1[i], f2[i], f3[i] = 40, 50, 60
    for i in range(34, 64):
        f1[i], f2[i], f3[i] = rep_enc(1, 2), rep_enc(2, 3), rep_enc(3, 4)
    for i in range(64, 96):  # block2: first-in-frame encoded -> defaults; branch A last lane actual
        f1[i], f2[i], f3[i] = 5000, 6000, 7000
    f1[64], f2[64], f3[64] = rep_enc(1, 2), rep_enc(2, 2), rep_enc(3, 2)
    return n, seq_to_block, per_frame_blocks, per_frame_min, fc, f1, f2, f3


drv = loader.Driver(0)
fn = drv.load(open(CUBIN, "rb").read()).function("main_0")
ok = True
for name, maker in (("two-frame first-in-frame", case_two_frame), ("three-frame branchB+lookback", case_three_frame)):
    n, sb, pfb, pmin, fc, f1, f2, f3 = maker()
    ok &= check(name, f1, f2, f3, sb, pfb, pmin, fc)
drv.destroy()
print("OVERALL:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
