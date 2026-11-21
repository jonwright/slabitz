#!/usr/bin/env python3
"""GPU-vs-expected comparison for the MemsetMemcpy kernel (RAW memcpy / RLE memset).

One workgroup per block (grid.x = Blocks_RAW / Blocks_RLE). Builds nothing; expects a
prebuilt cubin (tools/build_all.py -> build/mmcpy/mmcpy.cubin). The kernel is driven
through its pure-CUDA {data,count} ABI with synthetic block refs and scattered
destinations. The output buffer is pre-filled with a sentinel so that bytes written
outside a block (e.g. a word store clobbering a neighbour's edge or a gap) are caught.
Expected output is computed directly per block (no prefix/binary search any more).

Cases: sizes 0..9 (all-head, shorter than 4, tails), unaligned dst and src, adjacent
blocks sharing output words, blocks spanning several 256-word iterations.

Usage:
    PYTHONPATH=$PWD python3 tests/run_memsetmemcpy_chain.py [build/mmcpy/mmcpy.cubin]
"""
import ctypes as ct
import os
import random
import struct
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from slabitz import loader

CUBIN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build/mmcpy/mmcpy.cubin")
SENTINEL = 0xEE


class Buf(ct.Structure):
    _fields_ = [("data", ct.c_uint64), ("count", ct.c_uint64)]


def upload(drv, data, keep):
    n = max(4, (len(data) + 3) & ~3)
    m = loader.DeviceMemory(drv, n)
    m.from_host(bytes(data) + bytes(n - len(data)))
    keep.append(m)
    return m


def run_gpu(drv, fn, comp, dest_offs, refs, gidx, flags, out_bytes, nblocks):
    keep = []
    counters = [0] * 22
    counters[13 if flags else 14] = nblocks
    cntM = upload(drv, struct.pack("22I", *counters), keep)
    cdM = upload(drv, comp + bytes(4), keep)
    dM = upload(drv, struct.pack("%dI" % len(dest_offs), *dest_offs), keep)
    rM = upload(drv, b"".join(struct.pack("II", o, s) for o, s in refs), keep)
    gM = upload(drv, struct.pack("%dI" % len(gidx), *gidx), keep)
    padded = (out_bytes + 3) & ~3
    outM = loader.DeviceMemory(drv, padded)
    outM.from_host(bytes([SENTINEL]) * padded)
    args = [
        Buf(cntM.ptr, 22), Buf(cdM.ptr, (len(comp) + 4 + 3) // 4),
        Buf(dM.ptr, len(dest_offs)), Buf(rM.ptr, len(refs)), Buf(gM.ptr, len(gidx)),
        Buf(outM.ptr, padded), Buf(outM.ptr, padded // 4),
        ct.c_uint32(0), ct.c_uint32(flags),
    ]
    fn(grid=(nblocks, 1, 1), block=(256, 1, 1), args=args)
    drv.synchronize()
    return outM.to_host(padded)


def expected(comp, dest_offs, refs, gidx, flags, out_bytes):
    out = bytearray([SENTINEL]) * ((out_bytes + 3) & ~3)
    n = 0
    for i, (offs, size) in enumerate(refs):
        d = dest_offs[gidx[i]]
        for k in range(size):
            out[d + k] = comp[offs + k] if flags else (offs & 0xFF)
        n += size
    return bytes(out), n


def make_case(rng, flags, sizes, comp_len=70000, gap_max=0, start=0):
    """Blocks packed back to back (gap_max=0 => adjacent, sharing words) in a random
    global order; refs/prefix order is independent of destination order."""
    comp_len = max(comp_len, max(sizes) + 1000)
    comp = bytes(rng.randrange(256) for _ in range(comp_len))
    nb = len(sizes)
    pos = start
    dests = []
    for s in sizes:
        dests.append(pos)
        pos += s + (rng.randrange(gap_max + 1) if gap_max else 0)
    gidx = list(range(nb))
    rng.shuffle(gidx)                       # block i writes to global slot gidx[i]
    dest_offs = [0] * nb
    for i in range(nb):
        dest_offs[gidx[i]] = dests[i]
    refs = []
    for s in sizes:
        if flags:
            refs.append((rng.randrange(0, comp_len - s - 8), s))
        else:
            refs.append((rng.randrange(256), s))
    return comp, dest_offs, refs, gidx, pos + 3


def main():
    with open(CUBIN, "rb") as f:
        cubin = f.read()
    drv = loader.Driver(0)
    fn = drv.load(cubin).function("main_0")
    rng = random.Random(1234)
    cases = []
    for flags, tag in ((1, "RAW"), (0, "RLE")):
        cases.append((tag + " sizes 0..9 adjacent", flags, list(range(10)), 0, 0))
        cases.append((tag + " sizes 0..9 start=1", flags, list(range(9, -1, -1)), 0, 1))
        cases.append((tag + " tiny 1,1,1,1,2,3", flags, [1, 1, 1, 1, 2, 3], 0, 3))
        cases.append((tag + " random small, gaps", flags,
                      [rng.randrange(0, 40) for _ in range(300)], 7, 2))
        cases.append((tag + " random small, adjacent", flags,
                      [rng.randrange(0, 40) for _ in range(300)], 0, 1))
        cases.append((tag + " multi-iteration", flags,
                      [5000, 1023, 1024, 1025, 4099, 17, 131072 + 3], 0, 1))
        cases.append((tag + " one big unaligned", flags, [200001], 0, 3))
    ok = True
    for name, flags, sizes, gap, start in cases:
        comp, dest, refs, gidx, out_bytes = make_case(rng, flags, sizes, gap_max=gap,
                                                      start=start)
        gpu = run_gpu(drv, fn, comp, dest, refs, gidx, flags, out_bytes, len(sizes))
        cpu, compared = expected(comp, dest, refs, gidx, flags, out_bytes)
        mism = [j for j in range(len(cpu)) if gpu[j] != cpu[j]]
        status = "PASS" if not mism and compared > 0 else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"{status}: {name} ({len(sizes)} blocks, {compared} bytes written, "
              f"{len(cpu)} bytes checked, {len(mism)} mismatch)")
        if mism:
            j = mism[0]
            print("   first mismatches at", mism[:8], "gpu %02x cpu %02x" % (gpu[j], cpu[j]))
    drv.destroy()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
