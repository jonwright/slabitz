#!/usr/bin/env python3
"""GPU-vs-CPU comparison for the DecodeHuffmanWeights (raw nibble) kernel.

Builds nothing; expects a prebuilt cubin (tools/build_all.py -> build/dhw2/dhw2.cubin).
Three Huffman-weight tables with odd/even counts; unpacks 4-bit nibble weights
from byte-packed CompressedData segments.

Usage:
    PYTHONPATH=$PWD python3 tests/run_decodehw_chain.py build/dhw2/dhw2.cubin
"""
import ctypes as ct
import struct
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from slabitz import loader
from kernels.cpu_ref_decode_huffman_weights import ref_decode_huffman_weights, MAX_WEIGHTS
from kernels.cpu_ref_initresources import _FIELD_ORDER

CUBIN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build/dhw2/dhw2.cubin")
CMP = 3


class Buf(ct.Structure):
    _fields_ = [("data", ct.c_uint64), ("count", ct.c_uint64)]


def dm(drv, src, nbytes=4):
    m = loader.DeviceMemory(drv, len(src) * nbytes)
    m.from_host(struct.pack(f"{len(src)}{'Q' if nbytes == 8 else 'I'}", *src))
    return m


def pack_words(data):
    words = []
    for k in range(0, (len(data) + 3) // 4 * 4, 4):
        w = 0
        for i in range(4):
            w |= (data[k + i] if k + i < len(data) else 0) << (i * 8)
        words.append(w)
    return words


def run_gpu(drv, fn, counters, data_words, huf_refs, weight_count, out_len):
    cM = dm(drv, counters)
    dM = dm(drv, data_words)
    rM = dm(drv, [lo | (hi << 32) for lo, hi in huf_refs], 8)
    daM = dm(drv, [0] * 64)
    # weightCount is a uint8 buffer: one byte per table index.
    wM = loader.DeviceMemory(drv, CMP)
    wM.from_host(bytes(weight_count))
    oM = loader.DeviceMemory(drv, out_len)
    oM.from_host(bytes(out_len))
    args = [Buf(oM.ptr, out_len), Buf(wM.ptr, CMP), Buf(cM.ptr, len(counters)),
            Buf(dM.ptr, len(data_words)), Buf(rM.ptr, len(huf_refs)),
            Buf(daM.ptr, 64),
            ct.c_uint32(0), ct.c_uint32(len(huf_refs)), ct.c_uint32(len(data_words) * 4)]
    fn(grid=(1, 1, 1), block=(32, 1, 1), args=args)
    drv.synchronize()
    return list(oM.to_host(out_len))


def main():
    drv = loader.Driver(0)
    fn = drv.load(open(CUBIN, "rb").read()).function("main_0")

    counters = [0] * 22
    counters[_FIELD_ORDER.index("HUF_WgtStreams")] = CMP
    counters[_FIELD_ORDER.index("Blocks_CMP")] = CMP

    # byte-packed segments: table2 (idx2, thr0)=[AB CD EF], table1 (idx1)=[12 34],
    # table0 (idx0)=[56 78 9A]
    data = [0xAB, 0xCD, 0xEF, 0x00, 0x12, 0x34, 0x00, 0x00, 0x56, 0x78, 0x9A, 0x00]
    data_words = pack_words(data)
    huf_refs = [(0, 3), (4, 2), (8, 3)]  # index = hufTableIndex
    weight_count = [6, 4, 5]            # index = hufTableIndex

    cpu = [0] * (CMP * MAX_WEIGHTS)
    ref_decode_huffman_weights(counters, data_words, huf_refs, weight_count, cpu)
    gpu_ = run_gpu(drv, fn, counters, data_words, huf_refs, weight_count, CMP * MAX_WEIGHTS)

    # cpu has the same length after init; compare.
    bad = [(j, gpu_[j], cpu[j]) for j in range(len(cpu)) if gpu_[j] != cpu[j]]
    print(f"  DecodeHuffmanWeights: {'PASS' if not bad else 'FAIL'} ({len(bad)} mismatch)")
    if bad:
        print("    first few:", bad[:6])
        # highlight the touched regions
        for h in range(CMP):
            base = h * MAX_WEIGHTS
            print(f"    table{h} cpu:", cpu[base:base + weight_count[h]])
            print(f"    table{h} gpu:", gpu_[base:base + weight_count[h]])
    drv.destroy()
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
