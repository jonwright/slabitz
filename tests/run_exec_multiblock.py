#!/usr/bin/env python3
"""GPU ExecuteSequences across a MULTI-BLOCK frame via the array-based SRT.

A real .zst frame can contain several compressed blocks. ExecuteSequences is launched once per
frame and loops over the frame's compressed blocks using per-block sequencing and literal refs and
PREFIX arrays:

  per_frame_block_count_cmp[frameIdx]  = first cmp-block index of the frame (prefix over frames)
  inBlockSizePrefix[blockIdx]          = cumulative decompressed size through block blockIdx
  inBlockDestOffs[blockIdx]            = byte dest offset of block blockIdx
  inGlobalBlockIndexPerCmpBlock[i]     = block index for cmp-block i
  inPerSeqStreamSeqStart[stream]       = prefix over seq streams (seqOfs for stream s = pref[s])
  inCompressedBlocks[i]                = (literal.offs, literal.size|type, litStreamIdx, seqStreamIdx)

Because seq tables are SLOTTED per block (mode 0 predefined / mode 1 RLE symbol / mode 2 compressed
/ mode 3 repeat), each block's tables are placed at `ComputeFseDataStartFromFseIndex(kind, fseIndex,
cmpBlockCount)`, with the RLE region (fseElems[s]=PackFseElem(s,0,0), accuracy 0) seeded once and the
shared Fse alloc counters threaded across blocks. Offset history resets per stream (block).

Usage: run_exec_multiblock.py <exec.cubin>
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
from kernels import cpu_ref_pcb, cpu_ref_ifs, cpu_ref_dsq, cpu_ref_fso
from kernels.cpu_ref_pcb import _comp_fse_index, FSE_PROB_TABLE_INDEX_REPEAT
from tools.edu_oracle import run_oracle
from run_fso_chain import DEF_LLEN, DEF_OFFS, DEF_MLEN, FSE_RLE_TABLE_COUNT, FSE_MAX_COUNT_PROBS
from run_e2e_chain import decode_literals
import kparams

CUBIN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build/exec/exec.cubin")
DEF_ACC = {"LLen": 6, "Offs": 5, "MLen": 6}
DEF_PROBS = {"LLen": DEF_LLEN, "Offs": DEF_OFFS, "MLen": DEF_MLEN}


class Buf(ct.Structure):
    _fields_ = [("data", ct.c_uint64), ("count", ct.c_uint64)]


def data_start_from_index(kind, fse_index, cmpbc):
    """zstdgpu_ComputeFseDataStartFromFseIndex<kind>: RLE (<256) uses fse_index directly."""
    if fse_index < FSE_RLE_TABLE_COUNT:
        return fse_index
    base = _comp_fse_index(kind, 0, cmpbc)
    if kind == "LLen":
        return cpu_ref_dsq.compute_fse_data_start_llen(fse_index - base, cmpbc)
    if kind == "Offs":
        return cpu_ref_dsq.compute_fse_data_start_offs(fse_index - base, cmpbc)
    return cpu_ref_dsq.compute_fse_data_start_mlen(fse_index - base, cmpbc)


def build_table(r, kind, mode, index, cmpbc, prev):
    """Returns (elems, accuracy, data_start) for one seq table."""
    if mode == 3:  # repeat: carry previous block's table for this kind
        if prev.get(kind) is None:
            raise RuntimeError(f"repeat table with no previous: {kind}")
        return prev[kind]
    if mode == 2:  # compressed: probs in fse_info/fse_probs
        info = r["fse_info"][index]
        acc = info >> 8
        elems = cpu_ref_ifs.ref_initfse_table(
            r["fse_probs"], info, index - FSE_RLE_TABLE_COUNT, 0, 0, 1 << acc)
        return elems, acc, data_start_from_index(kind, index, cmpbc)
    if mode == 1:  # RLE: single symbol, accuracy 0, data_start = symbol (<256)
        symbol = index
        elems = [cpu_ref_ifs._pack_fse_elem(symbol, 0, 0)]
        return elems, 0, data_start_from_index(kind, symbol, cmpbc)
    # mode 0: predefined defaults
    acc = DEF_ACC[kind]
    dist = DEF_PROBS[kind]
    probs = [0] * FSE_MAX_COUNT_PROBS
    probs[0:len(dist)] = dist
    elems = cpu_ref_ifs.ref_initfse_table(probs, acc << 8 | len(dist), 0, 0, 0, 1 << acc)
    index0 = _comp_fse_index(kind, 0, cmpbc)
    return elems, acc, data_start_from_index(kind, index0, cmpbc)


def decode_block_seqs(block, num_seqs, cmpbc, fse_alloc_start, prev, frame_fse, place=True):
    n = len(block)
    r = cpu_ref_pcb.parse_compressed_block(block, n, cmp_block_count=cmpbc,
                                           global_block_index=0, fse_alloc_start=fse_alloc_start)
    src_offs, src_size = r["seq_stream_to_ref"]
    block_pad = block + b"\x00" * ((-len(block)) % 4)

    params = {}
    for kind in ("LLen", "Offs", "MLen"):
        mode = r["seq_fse_modes"][kind]
        index = r["seq_fse"][kind]
        elems, acc, dstart = build_table(r, kind, mode, index, cmpbc, prev)
        if mode != 3:
            prev[kind] = (elems, acc, dstart)
        if place:
            for j, v in enumerate(elems):
                frame_fse[dstart + j] = v
        params[kind] = (dstart, acc)

    start_llen, acc_llen = params["LLen"]
    start_offs, acc_offs = params["Offs"]
    start_mlen, acc_mlen = params["MLen"]

    ll, folded, ml, fo1, fo2, fo3, total_mlen = cpu_ref_dsq.decompress_sequences_single_stream(
        block_pad, src_offs, src_size, frame_fse, start_llen, start_offs, start_mlen,
        acc_llen, acc_offs, acc_mlen, 0, num_seqs)
    assert len(ll) == num_seqs, (len(ll), num_seqs, params)

    resolved = cpu_ref_fso.finalise_sequence_offsets(
        folded, num_seqs, seq_stream_cnt=1,
        per_seq_stream_seq_start=[0], seq_stream_to_block_id=[0],
        per_frame_block_count_all=[1], frame_cnt=1,
        per_frame_seq_stream_min_idx=[0],
        per_seq_stream_final_offset1=[fo1], per_seq_stream_final_offset2=[fo2],
        per_seq_stream_final_offset3=[fo3])

    lit_val = r["compressed_blocks"][0][1]
    lit_type = lit_val & 0xC0000000
    lit_size = lit_val & ~0xC0000000
    lit_offs_in_block = r["compressed_blocks"][0][0]
    block_decomp = lit_size + total_mlen
    info = {"ll": ll, "resolved": resolved, "ml": ml, "lit_type": lit_type,
            "lit_size": lit_size, "lit_offs_in_block": lit_offs_in_block,
            "block_decomp": block_decomp, "r": r,
            "folded": folded, "fo1": fo1, "fo2": fo2, "fo3": fo3, "total_mlen": total_mlen}
    return info


def global_finalise(infos, seq_stream_starts, block_ids):
    """Finalise offsets across ALL seq streams (blocks) at once, threading per-stream final
    offset history (the DS FinaliseSequenceOffsets does this via per_seq_stream_final_offset*).
    Per-block finalise is wrong when a sequence's repeat offset depends on a prior block."""
    all_folded = []
    fo1 = [info["fo1"] for info in infos]
    fo2 = [info["fo2"] for info in infos]
    fo3 = [info["fo3"] for info in infos]
    for info in infos:
        all_folded += info["folded"]
    return cpu_ref_fso.finalise_sequence_offsets(
        all_folded, len(all_folded), seq_stream_cnt=len(infos),
        per_seq_stream_seq_start=seq_stream_starts, seq_stream_to_block_id=block_ids,
        per_frame_block_count_all=[0], frame_cnt=1, per_frame_seq_stream_min_idx=[0],
        per_seq_stream_final_offset1=fo1, per_seq_stream_final_offset2=fo2,
        per_seq_stream_final_offset3=fo3)


def main():
    seed = int(os.environ.get("DSQ_SEED", 777))
    nw = int(os.environ.get("DSQ_N", 70000))
    random.seed(seed)
    words = "the quick brown fox jumps over lazy dog run fast stop slow near far".split()
    if os.environ.get("DSQ_RAW_MIX"):
        parts = []
        for _ in range(12):
            parts.append(" ".join(random.choice(words) for _ in range(4000)).encode())
            parts.append(bytes(random.randrange(0, 256) for _ in range(2000)))
        payload = b"".join(parts)
    else:
        payload = " ".join(random.choice(words) for _ in range(nw)).encode()
    comp = zstd.ZstdCompressor(level=5, write_checksum=bool(os.environ.get("DSQ_CHECKSUM"))).compress(payload)
    _, blocks = zstd_frame.parse_frame(comp)
    cb = [b for b in blocks if b.type_name == "compressed"]
    nblk = len(cb)
    print(f"frame: {nblk} compressed blocks, payload={len(payload)} bytes")

    d = run_oracle(comp, expected=payload)
    seq = d["SEQ"]
    oracle_offs = [x[0] for x in d["OFFS"]]

    cmpbc = nblk
    fse_alloc_start = {"LLen": 0, "Offs": 0, "MLen": 0}
    frame_fse = [0] * (256 + cmpbc * 2048 * 8)
    for s in range(FSE_RLE_TABLE_COUNT):            # RLE region: FseElems[s] = Pack(s,0,0)
        frame_fse[s] = cpu_ref_ifs._pack_fse_elem(s, 0, 0)

    ll_all, ml_all, block_size_prefix, block_dest_offs = [], [], [], []
    lit_all, lit_offs_by_block = [], {}
    infos = []
    starts = []
    blockids = []
    prev = {}
    size_prefix = 0
    data_prefix = 0
    acc_s = 0
    for i, b in enumerate(cb):
        if i >= len(seq):
            raise RuntimeError(f"oracle SEQ missing for block {i}")
        num_seqs = len(seq[i]) // 3
        info = decode_block_seqs(b.data, num_seqs, cmpbc, fse_alloc_start, prev, frame_fse)
        infos.append(info)
        starts.append(acc_s)
        acc_s += num_seqs
        blockids.append(i)
        ll = info["ll"]
        print(f"  block{i}: ok seqs={num_seqs} llen_sum={sum(ll)} "
              f"mlen_sum={sum(info['ml'])} bsize={info['block_decomp']} "
              f"litType={info['lit_type']:#x} litSize={info['lit_size']}")
        ll_all += ll
        ml_all += info["ml"]
        size_prefix += info["block_decomp"]
        block_size_prefix.append(size_prefix)
        block_dest_offs.append(size_prefix - info["block_decomp"])
        if info["lit_type"] == 0xC0000000:
            lit_offs = sum(len(x) for x in lit_all)
            lits = decode_literals(b.data, info["r"], info["r"]["fse_info"][256] >> 8)
            lit_all.append(lits)
            lit_offs_by_block[i] = lit_offs
        else:
            lit_offs_by_block[i] = data_prefix + info["lit_offs_in_block"]
        data_prefix += len(b.data)

    # global (threaded across streams) offset finalise; also validates against the oracle OFFS.
    offs_all = global_finalise(infos, starts, blockids)
    total_seqs = len(offs_all)
    mism = sum(1 for j in range(total_seqs) if offs_all[j] != oracle_offs[j])
    print(f"  global finalise: {mism}/{total_seqs} offset mismatches vs oracle")
    if mism:
        return 1

    # ---- GPU setup (same arg wiring as run_exec_gpu_chain, multi-block) ----
    literals = b"".join(lit_all)
    per_seq_stream_seq_start = starts
    seq_stream_cnt = nblk

    global_block_index = list(range(nblk))
    per_frame_block_count_cmp = [0]

    compressed_blocks = []
    for i, b in enumerate(cb):
        lit_type = infos[i]["lit_type"]
        lit_size = infos[i]["lit_size"]
        compressed_blocks += [lit_offs_by_block[i], lit_type | lit_size, i, i]

    out_size = len(payload)
    print(f"assembled: total_seqs={total_seqs} streams={seq_stream_cnt} lit_bytes={len(literals)}")

    with open(CUBIN, "rb") as f:
        cubin = f.read()
    drv = loader.Driver(0)
    fn = drv.load(cubin).function("main_0")

    def dev(bytes_or_vals):
        if isinstance(bytes_or_vals, (bytes, bytearray)):
            padded = bytes(bytes_or_vals) + b"\x00" * ((-len(bytes_or_vals)) % 4)
            m = loader.DeviceMemory(drv, len(padded))
            m.from_host(padded)
            return m, len(padded)
        m = loader.DeviceMemory(drv, len(bytes_or_vals) * 4)
        m.from_host(struct.pack("%dI" % len(bytes_or_vals), *bytes_or_vals))
        return m, len(bytes_or_vals)

    comp_data = b"".join(b.data for b in cb)
    m, n = dev(comp_data); comp_mem = m; comp_bytes = n
    lit_src = literals if literals else b"\x00" * 4
    m, n = dev(lit_src + b"\x00" * ((-len(lit_src)) % 4)); lit_mem = m; lit_bytes = n
    out_padded = out_size + ((-out_size) % 16)  # also viewed as u32 and uint4
    m, n = dev(b"\x00" * out_padded); out_mem = m
    m, n = dev([0] * 24); counters = m
    ctr = [0] * 24
    ctr[8] = seq_stream_cnt; ctr[6] = total_seqs; ctr[15] = nblk; ctr[18] = 1
    counters.from_host(struct.pack("24I", *ctr))

    m, n = dev(per_frame_block_count_cmp); pfb_cmp = m
    m, n = dev(block_size_prefix); size_pref = m
    m, n = dev(block_dest_offs); dest_offs = m
    m, n = dev(ll_all); seq_llen = m
    m, n = dev(ml_all); seq_mlen = m
    m, n = dev(offs_all); seq_offs = m
    m, n = dev(global_block_index); glob_idx = m
    m, n = dev(per_seq_stream_seq_start); seq_start = m
    m, n = dev(compressed_blocks); comp_blk = m

    args = [
        Buf(comp_mem.ptr, comp_bytes),
        Buf(lit_mem.ptr, len(literals)),
        Buf(out_mem.ptr, out_padded),             # inoutUnCompressedFramesData (bytes)
        Buf(out_mem.ptr, out_padded // 4),        # inoutUnCompressedFramesData32 (u32 view)
        Buf(out_mem.ptr, out_padded // 16),       # inoutUnCompressedFramesData16 (uint4 view)
        Buf(counters.ptr, 1),                     # inCounters (one 88-byte struct)
        Buf(pfb_cmp.ptr, len(per_frame_block_count_cmp)),
        Buf(size_pref.ptr, nblk),
        Buf(dest_offs.ptr, nblk),
        Buf(seq_llen.ptr, total_seqs),
        Buf(seq_mlen.ptr, total_seqs),
        Buf(seq_offs.ptr, total_seqs),
        Buf(glob_idx.ptr, nblk),
        Buf(seq_start.ptr, nblk),
        Buf(comp_blk.ptr, nblk),                  # inCompressedBlocks (16-byte elements)
        ct.c_uint32(lit_bytes),                   # litCapacity (literal bytes)
        ct.c_uint32(total_seqs),                  # seqCapacity (sequence slots)
    ]
    kparams.check_args("ExecuteSequences", args)

    print(f"launching exec (grid=(1,1,1) block=32) over {nblk} blocks...", flush=True)
    fn(grid=(1, 1, 1), block=(32, 1, 1), args=args, shared=0)
    drv.synchronize()

    raw = out_mem.to_host(out_padded)
    g_out = bytes(raw[:out_size])
    ok = g_out == payload
    print(f"{'PASS' if ok else 'FAIL'}: GPU multi-block ExecuteSequences == libzstd "
          f"({out_size} bytes, {nblk} blocks)")
    if not ok:
        print(f"  g_out[0:24]  = {list(g_out[0:24])}")
        print(f"  payload[0:24]= {list(payload[0:24])}")
        print(f"  literals[0:24]={list(literals[0:24])}")
        for i in range(min(out_size, len(payload))):
            if g_out[i] != payload[i]:
                print(f"  first diff at {i}: got={g_out[i]} want={payload[i]}")
                break
    print("compared %d bytes" % (out_size if ok else 0))
    return 0 if ok and out_size > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
