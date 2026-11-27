#!/usr/bin/env python3
"""GPU end-to-end output comparison for the ExecuteSequences kernel.

Chain: decode literals (cpu_ref_dl) + sequences -> FinaliseSequenceOffsets (resolved byte
offsets) -> run the GPU ExecuteSequences cubin to emit the decompressed output -> compare to the
original payload (== libzstd).

Usage: run_exec_gpu_chain.py <exec.cubin>
Env: DSQ_MULTI=1 for multi-sequence blocks; DSQ_SEED/DSQ_N select the payload.
"""
import os
import random
import sys
import struct
import ctypes as ct
import zstandard as zstd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from slabitz import zstd_frame, loader
from kernels import cpu_ref_pcb, cpu_ref_fso
from tools.edu_oracle import run_oracle
from run_fso_chain import decode_block
from run_e2e_chain import decode_literals, make_payload
import kparams

CUBIN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build/exec/exec.cubin")
MAX_SEQ = 20000


class Buf(ct.Structure):
    _fields_ = [("data", ct.c_uint64), ("count", ct.c_uint64)]


def main():
    payload = make_payload(int(os.environ.get("DSQ_N", 541)), int(os.environ.get("DSQ_SEED", 1)))
    comp = zstd.ZstdCompressor(level=5).compress(payload)
    _, blocks = zstd_frame.parse_frame(comp)
    block = [b for b in blocks if b.type_name == "compressed"][0].data
    n = len(block)

    r = cpu_ref_pcb.parse_compressed_block(block, n)
    # Only CMP literals are Huffman-decoded on the CPU into inDecompressedLiterals; RAW/RLE read
    # literal bytes directly from inCompressedData or a repeated symbol.
    lit_type = r["compressed_blocks"][0][1] & 0xC0000000
    literals = decode_literals(block, r, r["fse_info"][256] >> 8) if lit_type == 0xC0000000 else b""

    # sequences -> resolved byte offsets
    comp2, pl, r2, folded, fo1, fo2, fo3, num_seqs = decode_block(payload)
    flat = run_oracle(comp, expected=payload)["SEQ"][0]
    ll, ml = [], []
    for i in range(num_seqs):
        ll.append(flat[3 * i])
        ml.append(flat[3 * i + 1])
    offs_enc = cpu_ref_fso.finalise_sequence_offsets(
        folded, num_seqs, 1, [0], [0], [1], 1, [0], [fo1], [fo2], [fo3])

    out_size = len(payload)
    print(f"block={n} literals={len(literals)} seqs={num_seqs} out={out_size}")

    # literal section ref: (offs, size-with-littype, litStreamIndex, seqStreamIndex). Detect type.
    lit_val = r["compressed_blocks"][0][1]
    lit_type = lit_val & 0xC0000000
    lit_size_val = lit_val
    lit_offs = r["compressed_blocks"][0][0]
    lit_section = [lit_offs, lit_size_val, 0, 0]   # CompressedBlockData (used for RAW/RLE/CMP)
    # For CMP the literals live in inDecompressedLiterals (offs 0); for RAW/RLE in CompressedData.
    need_decompressed_literals = (lit_type == 0xC0000000)

    # ---- GPU setup ----
    block_pad = block + b"\x00" * ((-len(block)) % 4)
    with open(CUBIN, "rb") as f:
        cubin = f.read()
    drv = loader.Driver(0)
    fn = drv.load(cubin).function("main_0")

    def dev(bytes_or_vals):
        if isinstance(bytes_or_vals, (bytes, bytearray)):
            padded = bytes(bytes_or_vals) + b"\x00" * ((-len(bytes_or_vals)) % 4)
            m = loader.DeviceMemory(drv, len(padded))
            m.from_host(padded)
        else:
            m = loader.DeviceMemory(drv, len(bytes_or_vals) * 4)
            m.from_host(struct.pack("%dI" % len(bytes_or_vals), *bytes_or_vals))
        return m

    # output padded to 16 bytes: the kernel also takes u32 and uint4 views of it
    out_padded = out_size + ((-out_size) % 16)
    out_mem = loader.DeviceMemory(drv, out_padded)
    out_mem.from_host(bytes(out_padded))

    literals_padded = len(literals) + ((-len(literals)) % 4)
    lits_mem = loader.DeviceMemory(drv, max(literals_padded, 4))
    lits_mem.from_host(bytes(literals) + b"\x00" * (lits_mem.size - len(literals)))

    comp_mem = dev(block_pad)

    counters = dev([0] * 24)
    ctr = [0] * 24
    ctr[15] = 1                 # Blocks_CMP
    ctr[8] = 1                  # Seq_Streams
    ctr[6] = num_seqs           # Seq_Streams_DecodedItems
    ctr[18] = 1                 # Frames
    counters.from_host(struct.pack("%dI" % 24, *ctr))

    per_frame_block_count_cmp = dev([0])            # frame 0 begins at cmp block 0
    block_size_prefix = dev([out_size])             # block 0 total decompressed size
    block_dest_offs = dev([0])                      # block 0 dest offset = 0
    seq_llen = dev(ll)
    seq_mlen = dev(ml)
    seq_offs = dev(offs_enc)
    global_block_index = dev([0])
    per_seq_stream_seq_start = dev([0])
    compressed_blocks = dev(lit_section)   # CompressedBlockData (offs,size,litStreamIdx,seqStreamIdx)

    # output buffer count in uint32 elements
    max_bytes = out_padded
    args = [
        Buf(comp_mem.ptr, len(block_pad)),        # inCompressedData (byte buf; count = bytes)
        Buf(lits_mem.ptr, lits_mem.size),         # inDecompressedLiterals (byte buf; count = bytes)
        Buf(out_mem.ptr, max_bytes),              # inoutUnCompressedFramesData (byte buf; count = bytes)
        Buf(out_mem.ptr, max_bytes // 4),         # inoutUnCompressedFramesData32 (u32 view)
        Buf(out_mem.ptr, max_bytes // 16),        # inoutUnCompressedFramesData16 (uint4 view)
        Buf(counters.ptr, 1),                     # inCounters (one 88-byte struct)
        Buf(per_frame_block_count_cmp.ptr, 1),
        Buf(block_size_prefix.ptr, 1),
        Buf(block_dest_offs.ptr, 1),
        Buf(seq_llen.ptr, num_seqs),
        Buf(seq_mlen.ptr, num_seqs),
        Buf(seq_offs.ptr, num_seqs),
        Buf(global_block_index.ptr, 1),
        Buf(per_seq_stream_seq_start.ptr, 1),
        Buf(compressed_blocks.ptr, 1),
        ct.c_uint32(lits_mem.size),               # litCapacity (literal bytes)
        ct.c_uint32(num_seqs),                    # seqCapacity (sequence slots)
    ]
    kparams.check_args("ExecuteSequences", args)

    print("launching exec main_0 (grid=(1,1,1) block=32)...", flush=True)
    fn(grid=(1, 1, 1), block=(32, 1, 1), args=args, shared=0)
    drv.synchronize()
    print("synced", flush=True)

    raw = out_mem.to_host(out_padded)
    g_out = bytes(raw[:out_size])
    ok = g_out == payload
    print(f"{'PASS' if ok else 'FAIL'}: GPU ExecuteSequences output matches libzstd "
          f"({out_size} bytes)")
    if not ok and os.environ.get("DUMP"):
        print("  g_out[0:24]  =", list(g_out[0:24]))
        print("  payload[0:24]=", list(payload[0:24]))
    if not ok:
        for i in range(min(out_size, len(payload))):
            if g_out[i] != payload[i]:
                print(f"  first diff at {i}: got={g_out[i]} want={payload[i]}")
                break
        print(f"  lens out={len(g_out)} payload={len(payload)}")
    print("compared %d bytes" % (out_size if ok else 0))
    return 0 if ok and out_size > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
