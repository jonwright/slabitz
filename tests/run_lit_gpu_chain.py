#!/usr/bin/env python3
"""GPU-vs-oracle comparison for the DecompressLiterals kernel.

Chain: parse the real block (CPU) -> build the HufW FSE DTable (cpu_ref_ifs) ->
DecompressHuffmanWeights (cpu_ref_dhw) -> run the GPU InitHuffmanTableAndDecompressLiterals
cubin on the same inputs -> read back the decompressed literals -> compare to the
educational decoder's `LITS` hook (and to the CPU literal ref cpu_ref_dl).

Usage:
    run_lit_gpu_chain.py <lit.cubin>
Env: LIT_N and LIT_SEED select the payload (defaults 541/1).
"""
import os
import random
import struct
import sys
import ctypes as ct
import zstandard as zstd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from slabitz import zstd_frame, loader
from kernels import cpu_ref_pcb, cpu_ref_ifs, cpu_ref_dhw, cpu_ref_dl
from tools.edu_oracle import run_oracle

CUBIN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build/lit/lit.cubin")

MAX_LIT = 2048
COUNTERS_ELEMS = 24


class Buf(ct.Structure):
    _fields_ = [("data", ct.c_uint64), ("count", ct.c_uint64)]


def make_payload(n=541, seed=1):
    random.seed(seed)
    return (b"hello world " * 20 + b"\n" + bytes(random.randrange(32, 127) for _ in range(n - 200)))[:n]


def pack_u8_bytes(dst):
    """Pack a byte sequence into a 4-byte-aligned uint32 array (little-endian)."""
    padded = bytes(dst) + b"\x00" * ((-len(dst)) % 4)
    return struct.pack("%dI" % (len(padded) // 4), *[x for x in struct.unpack("%dI" % (len(padded) // 4), padded)])


def main():
    payload = make_payload(int(os.environ.get("LIT_N", 541)), int(os.environ.get("LIT_SEED", 1)))
    comp = zstd.ZstdCompressor(level=5).compress(payload)
    _, blocks = zstd_frame.parse_frame(comp)
    block = [b for b in blocks if b.type_name == "compressed"][0].data
    n = len(block)

    r = cpu_ref_pcb.parse_compressed_block(block, n)
    info = r["fse_info"].get(256)
    acc = info >> 8
    probs = r["fse_probs"][0:64]
    offs, size = r["huf_refs"][0][1]
    lit_streams = [lr[1] for lr in sorted(r["lit_refs"])]
    print(f"block={n} B acc={acc} lit_streams={len(lit_streams)}")

    truth = run_oracle(comp, expected=payload)["LITS"][0]
    print(f"oracle literals len={len(truth)}")

    # CPU weights + Huffman table (validated refs)
    dtable = cpu_ref_ifs.ref_initfse_table(probs, info, 0, 0, 0, 1 << acc)
    fse_elems = [0] * 512
    base = cpu_ref_dhw.compute_fse_data_start_hufw(0, 1)
    for j, v in enumerate(dtable):
        fse_elems[base + j] = v
    cw, ccount = cpu_ref_dhw.decompress_huffman_weights(block, offs, size, fse_elems, acc, 1, 0)

    bits_max, code_table_size, rank_index, rcp, cas = cpu_ref_dl.preinit_huffman_table(cw, ccount)
    huff_table = cpu_ref_dl.expand_huffman_table(bits_max, code_table_size, rank_index, cas)
    cpu_lits = cpu_ref_dl.decompress_huffman_literals(block, lit_streams, huff_table, bits_max)
    print(f"CPU lit ref len={len(cpu_lits)}")

    # ---- GPU setup ----
    data = bytearray(block)
    data += b"\x00" * ((-len(data)) % 4)
    n_u32 = len(data) // 4

    with open(CUBIN, "rb") as f:
        cubin = f.read()
    drv = loader.Driver(0)
    fn = drv.load(cubin).function("main_0")

    comp_m = loader.DeviceMemory(drv, len(data))
    comp_m.from_host(bytes(data))

    lit_group_end = loader.DeviceMemory(drv, 4 * 2)        # StructuredBuffer<uint32>
    lit_group_end.from_host(struct.pack("2I", 4, 4))        # per-bt end; not read for htCount==1

    counters = loader.DeviceMemory(drv, COUNTERS_ELEMS * 4)
    ctr = [0] * COUNTERS_ELEMS
    ctr[9] = len(lit_streams)     # HUF_Streams
    ctr[10] = 1                   # HufLit
    ctr[15] = 1                   # Blocks_CMP
    counters.from_host(struct.pack("%di" % COUNTERS_ELEMS, *ctr))

    lit_refs = loader.DeviceMemory(drv, 16 * 8)
    lr = []
    for (so, ss, do_, ds) in lit_streams:
        lr += [so, ss, do_, ds]
    lit_refs.from_host(struct.pack("%dI" % len(lr), *lr))

    hufwid_to_huflitid = loader.DeviceMemory(drv, 4)       # HufWIdToHufLitId[0] = 0
    hufwid_to_huflitid.from_host(struct.pack("I", 0))
    huflitid_to_litstream = loader.DeviceMemory(drv, 8)    # HufLitIdToLitStreamId[0] = 0, [1] = 4
    huflitid_to_litstream.from_host(struct.pack("2I", 0, len(lit_streams)))

    literals = loader.DeviceMemory(drv, MAX_LIT * 4)       # byte view + dword view share ptr
    literals.from_host(bytes(MAX_LIT * 4))

    huf_weights = loader.DeviceMemory(drv, 256 * 4)
    huf_weights.from_host(pack_u8_bytes(cw))
    huf_wcount = loader.DeviceMemory(drv, 4)
    huf_wcount.from_host(pack_u8_bytes([ccount]))

    dargs = loader.DeviceMemory(drv, 16 * 4)
    dargs.from_host(struct.pack("16i", *([0] * 16)))

    args = [
        Buf(lit_group_end.ptr, 2),     # inLitGroupEndPerHuffmanTable
        Buf(counters.ptr, 1),          # inCounters (zstdgpu_Counters elements)
        Buf(lit_refs.ptr, len(lit_streams)),   # inLitRefs
        Buf(comp_m.ptr, len(data)),    # inCompressedData (ByteAddressBuffer: data, sizeInBytes)
        Buf(hufwid_to_huflitid.ptr, 1),        # inHufWIdToHufLitId
        Buf(huflitid_to_litstream.ptr, 2),     # inHufLitIdToLitStreamId
        Buf(literals.ptr, MAX_LIT),    # inoutDecompressedLiterals (byte view; count in u32 elems)
        Buf(literals.ptr, MAX_LIT),    # inoutDecompressedLiterals_Dwords
        Buf(huf_weights.ptr, 256),     # inDecompressedHuffmanWeights
        Buf(huf_wcount.ptr, 1),        # inDecompressedHuffmanWeightCount
        Buf(dargs.ptr, 16),            # inDispatchArgs
        ct.c_uint32(0),                # tgOffset
        ct.c_uint32(1),                # workItemCount
        ct.c_uint32(32),               # streamsPerGroup
    ]
    print("launching main_0...", flush=True)
    fn(grid=(1, 1, 1), block=(32, 1, 1), args=args, shared=1614 * 4)
    print("launched, synchronizing...", flush=True)
    drv.synchronize()
    print("synced", flush=True)

    raw = literals.to_host(MAX_LIT * 4)
    lit_byte_len = max(do_ + ds for (_, _, do_, ds) in lit_streams)
    g_lits = bytes(raw[:lit_byte_len])
    print(f"GPU literals len={lit_byte_len}")

    mism_g = [i for i in range(min(len(g_lits), len(truth))) if g_lits[i] != truth[i]]
    ok_g = (not mism_g) and len(g_lits) == len(truth)
    print(f"{'PASS' if ok_g else 'FAIL'}: GPU literals vs oracle ({len(g_lits)} vs {len(truth)}, "
          f"{len(mism_g)} mismatch)")
    if mism_g:
        print("  first (idx, gpu, oracle):", [(i, g_lits[i], truth[i]) for i in mism_g[:15]])

    mism_c = [i for i in range(min(len(cpu_lits), len(truth))) if cpu_lits[i] != truth[i]]
    ok_c = (not mism_c) and len(cpu_lits) == len(truth)
    print(f"{'PASS' if ok_c else 'FAIL'}: CPU lit ref vs oracle ({len(mism_c)} mismatch)")

    sys.exit(0 if (ok_g and ok_c) else 1)


if __name__ == "__main__":
    sys.exit(main())
