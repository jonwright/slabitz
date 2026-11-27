#!/usr/bin/env python3
"""Full-frame GPU decode: `run_frame_chain.py <exec.cubin>`.

A real .zst frame mixes whole-block types:
  raw        -> block.data IS the decompressed output (host memcpy, no shader in DS for it)
  rle        -> block.data[0] repeated (host memset; DS has no shader either)
  compressed -> the full DS pipeline via the ExecuteSequences GPU kernel (as run_exec_multiblock)

The DS emits whole-block RAW/RLE host-side; only CMP blocks go through ExecuteSequences. We mirror
that: emit raw/RLE regions into the output, launch exec (which writes only the CMP regions at their
"global" dest offsets), and compare the assembled output to libzstd (== the oracle's _output).

Env: DSQ_SEED (payload generator seed). The payload alternates large compressible (repeated words)
and large incompressible (random) runs so libzstd emits both raw and compressed blocks.
"""
import os
import random
import struct
import ctypes as ct
import zstandard as zstd

import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from slabitz import zstd_frame, loader
from tools.edu_oracle import run_oracle
from run_exec_multiblock import (decode_block_seqs, global_finalise, Buf, FSE_RLE_TABLE_COUNT,
                                 cpu_ref_ifs)
import kparams

CUBIN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build/exec/exec.cubin")


def make_payload(seed):
    random.seed(seed)
    words = "the quick brown fox jumps over lazy dog run fast stop slow near far".split()
    parts = []
    for _ in range(8):
        parts.append((" ".join(random.choice(words) for _ in range(20000)) + " ").encode())
        parts.append(bytes(random.randrange(0, 256) for _ in range(200000)))
    return b"".join(parts)


def main():
    seed = int(os.environ.get("DSQ_SEED", 7))
    payload = make_payload(seed)
    comp = zstd.ZstdCompressor(level=3).compress(payload)
    fi, blocks = zstd_frame.parse_frame(comp)
    nblk = len(blocks)
    cmp_blocks = [i for i, b in enumerate(blocks) if b.type_name == "compressed"]
    ncmp = len(cmp_blocks)
    print(f"frame: {nblk} blocks ({ncmp} compressed), payload={len(payload)} bytes")

    d = run_oracle(comp, expected=payload)
    seq = d["SEQ"]
    oracle_offs = [x[0] for x in d["OFFS"]]
    assert len(seq) == ncmp, (len(seq), ncmp)

    # ---- decode each CMP block's sequences (threaded FSE alloc + frame-wide fse[]) ----
    cmpbc = ncmp
    fse_alloc_start = {"LLen": 0, "Offs": 0, "MLen": 0}
    frame_fse = [0] * (256 + cmpbc * 2048 * 8)
    for s in range(FSE_RLE_TABLE_COUNT):
        frame_fse[s] = cpu_ref_ifs._pack_fse_elem(s, 0, 0)

    per_cmp = {}          # cmp-ordinal -> info
    infos_order = [None] * ncmp
    starts_order = [0] * ncmp
    blockids_order = [0] * ncmp
    prev = {}
    si = 0                # oracle SEQ index
    acc_s = 0
    ll_all, ml_all = [], []
    lit_all, lit_offs_cmp = [], {}
    cmp_data = b""
    for gi, b in enumerate(blocks):
        if b.type_name != "compressed":
            continue
        ord_ = si
        num_seqs = len(seq[si]) // 3
        info = decode_block_seqs(b.data, num_seqs, cmpbc, fse_alloc_start, prev, frame_fse)
        si += 1
        per_cmp[gi] = info
        infos_order[ord_] = info
        starts_order[ord_] = acc_s
        acc_s += num_seqs
        blockids_order[ord_] = gi
        print(f"  block{gi} (cmp ord {ord_}): seqs={num_seqs} out={info['block_decomp']} "
              f"litType={info['lit_type']:#x} litSize={info['lit_size']}")
        ll_all += info["ll"]
        ml_all += info["ml"]
        if info["lit_type"] == 0xC0000000:
            lit_offs = sum(len(x) for x in lit_all)
            lits = __import__("run_e2e_chain", fromlist=["decode_literals"]).decode_literals(
                b.data, info["r"], info["r"]["fse_info"][256] >> 8)
            lit_all.append(lits)
            lit_offs_cmp[ord_] = lit_offs
        else:
            lit_offs_cmp[ord_] = len(cmp_data) + info["lit_offs_in_block"]
        cmp_data += b.data

    # global (threaded across streams) offset finalise — required for cross-block repeat offsets.
    offs_all = global_finalise(infos_order, starts_order, blockids_order)
    total_seqs = len(offs_all)
    mism = sum(1 for j in range(total_seqs) if offs_all[j] != oracle_offs[j])
    print(f"  global finalise: {mism}/{total_seqs} offset mismatches vs oracle")
    if mism:
        return 1

    # ---- compute sidePrefix/destOffs over ALL blocks (global block index) ----
    size_prefix = []   # cumulative decompressed size through each global block
    dest_offs = []
    acc = 0
    for gi, b in enumerate(blocks):
        if b.type_name == "compressed":
            sz = per_cmp[gi]["block_decomp"]
        else:
            sz = len(b.data)
        acc += sz
        size_prefix.append(acc)
        dest_offs.append(acc - sz)
    out_size = len(payload)
    print(f"assembled: total_seqs={total_seqs} src_over_size={acc} ({'OK' if acc==out_size else 'MISMATCH'})")

    literals = b"".join(lit_all)
    per_seq_stream_seq_start = starts_order
    assert starts_order[len(starts_order) - 1] + len(infos_order[len(infos_order) - 1]["ll"]) == total_seqs

    global_block_index_for_cmp = [cmp_blocks[i] for i in range(ncmp)]
    per_frame_block_count_cmp = [0]
    compressed_blocks = []
    for i, gi in enumerate(cmp_blocks):
        lit_type = per_cmp[gi]["lit_type"]
        lit_size = per_cmp[gi]["lit_size"]
        compressed_blocks += [lit_offs_cmp[i], lit_type | lit_size, i, i]

    # ---- GPU ----
    with open(CUBIN, "rb") as f:
        cubin = f.read()
    drv = loader.Driver(0)
    fn = drv.load(cubin).function("main_0")

    def dev(bytes_or_vals):
        if isinstance(bytes_or_vals, (bytes, bytearray)):
            padded = bytes(bytes_or_vals) + b"\x00" * ((-len(bytes_or_vals)) % 4)
            m = loader.DeviceMemory(drv, len(padded))
            m.from_host(padded)
            return m
        m = loader.DeviceMemory(drv, len(bytes_or_vals) * 4)
        m.from_host(struct.pack("%dI" % len(bytes_or_vals), *bytes_or_vals))
        return m

    comp_mem = dev(cmp_data)
    lit_src = literals if literals else b"\x00" * 4
    lit_bytes = len(lit_src) + ((-len(lit_src)) % 4)
    lit_mem = dev(lit_src + b"\x00" * ((-len(lit_src)) % 4))

    # output buffer: host-emit raw/RLE regions, then exec writes CMP regions.
    host_out = bytearray(bytes(out_size))
    for gi, b in enumerate(blocks):
        if b.type_name != "compressed":
            do = dest_offs[gi]
            if b.type_name == "raw":
                host_out[do:do + len(b.data)] = b.data
            else:  # rle
                host_out[do:do + len(b.data)] = b.data * (len(b.data) and 1) or b""
    out_padded = out_size + ((-out_size) % 16)  # also viewed as u32 and uint4
    out_mem = loader.DeviceMemory(drv, out_padded)
    out_mem.from_host(bytes(host_out) + b"\x00" * (out_padded - len(host_out)))

    ctr = [0] * 24
    ctr[8] = ncmp; ctr[6] = total_seqs; ctr[15] = ncmp; ctr[18] = 1
    counters = dev(ctr)

    pfb_cmp = dev(per_frame_block_count_cmp)
    size_pref = dev(size_prefix)
    dest_offs_d = dev(dest_offs)
    seq_llen = dev(ll_all)
    seq_mlen = dev(ml_all)
    seq_offs = dev(offs_all)
    glob_idx = dev(global_block_index_for_cmp)
    seq_start = dev(per_seq_stream_seq_start)
    comp_blk = dev(compressed_blocks)

    args = [
        Buf(comp_mem.ptr, len(cmp_data)),
        Buf(lit_mem.ptr, len(lit_src)),
        Buf(out_mem.ptr, out_padded),             # inoutUnCompressedFramesData (bytes)
        Buf(out_mem.ptr, out_padded // 4),        # inoutUnCompressedFramesData32 (u32 view)
        Buf(out_mem.ptr, out_padded // 16),       # inoutUnCompressedFramesData16 (uint4 view)
        Buf(counters.ptr, 1),                     # inCounters (one 88-byte struct)
        Buf(pfb_cmp.ptr, len(per_frame_block_count_cmp)),
        Buf(size_pref.ptr, nblk),
        Buf(dest_offs_d.ptr, nblk),
        Buf(seq_llen.ptr, total_seqs),
        Buf(seq_mlen.ptr, total_seqs),
        Buf(seq_offs.ptr, total_seqs),
        Buf(glob_idx.ptr, ncmp),
        Buf(seq_start.ptr, ncmp),
        Buf(comp_blk.ptr, ncmp),                  # inCompressedBlocks (16-byte elements)
        ct.c_uint32(lit_bytes),                   # litCapacity (literal bytes)
        ct.c_uint32(total_seqs),                  # seqCapacity (sequence slots)
    ]
    kparams.check_args("ExecuteSequences", args)

    print(f"launching exec over {ncmp} CMP blocks ({nblk} total blocks)...", flush=True)
    fn(grid=(1, 1, 1), block=(32, 1, 1), args=args, shared=0)
    drv.synchronize()

    g_out = bytes(out_mem.to_host(out_padded)[:out_size])
    ok = g_out == payload
    print(f"{'PASS' if ok else 'FAIL'}: full-frame GPU decode == libzstd "
          f"({out_size} bytes, {nblk} blocks, {ncmp} cmp)")
    if not ok:
        for i in range(min(out_size, len(payload))):
            if g_out[i] != payload[i]:
                print(f"  first diff at {i}: got={g_out[i]} want={payload[i]}")
                break
        print(f"  g_out[{0}:24]  = {list(g_out[0:24])}")
        print(f"  payload[0:24]= {list(payload[0:24])}")
    print("compared %d bytes" % (out_size if ok else 0))
    return 0 if ok and out_size > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
