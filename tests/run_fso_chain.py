#!/usr/bin/env python3
"""CPU-reference validation for FinaliseSequenceOffsets.

Chain: decode sequences (cpu_ref_dsq) to get the DS offset-history-encoded offsets, run the
FinaliseSequenceOffsets scalar logic (cpu_ref_fso), and compare the resolved real byte offsets to
the oracle's `OFFS` hook (zstd compute_offset value per sequence).
"""
import os
import random
import sys
import zstandard as zstd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from slabitz import zstd_frame
from kernels import cpu_ref_pcb, cpu_ref_ifs, cpu_ref_dsq, cpu_ref_fso
from tools.edu_oracle import run_oracle

FSE_RLE_TABLE_COUNT = 256
FSE_MAX_COUNT_PROBS = 64

DEF_LLEN = [4,3,2,2,2,2,2,2,2,2,2,2,2,1,1,1,2,2,2,2,2,2,2,2,2,3,2,1,1,1,1,1,-1,-1,-1,-1]
DEF_OFFS = [1,1,1,1,1,1,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1]
DEF_MLEN = [1,4,3,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1]


def make_payload(n=541, seed=1):
    if os.environ.get("DSQ_MULTI"):
        words = "the quick brown fox jumps over lazy dog run fast stop slow near far".split()
        random.seed(seed)
        return " ".join(random.choice(words) for _ in range(int(n) if n > 10000 else 12000)).encode()
    random.seed(seed)
    return (b"hello world " * 20 + b"\n" + bytes(random.randrange(32, 127) for _ in range(n - 200)))[:n]


def decode_block(payload):
    comp = zstd.ZstdCompressor(level=5).compress(payload)
    _, blocks = zstd_frame.parse_frame(comp)
    block = [b for b in blocks if b.type_name == "compressed"][0].data
    n = len(block)

    r = cpu_ref_pcb.parse_compressed_block(block, n)
    sf = r["seq_fse"]
    info = r["fse_info"]

    def build_dtable(name):
        idx = sf[name]
        local = idx - FSE_RLE_TABLE_COUNT
        if idx in info:
            acc = info[idx] >> 8
            return acc, cpu_ref_ifs.ref_initfse_table(r["fse_probs"], info[idx], local, 0, 0, 1 << acc)
        acc = {"LLen": 6, "Offs": 5, "MLen": 6}[name]
        dist = {"LLen": DEF_LLEN, "Offs": DEF_OFFS, "MLen": DEF_MLEN}[name]
        probs = [0] * (16 * FSE_MAX_COUNT_PROBS)
        probs[local * FSE_MAX_COUNT_PROBS:local * FSE_MAX_COUNT_PROBS + len(dist)] = dist
        return acc, cpu_ref_ifs.ref_initfse_table(probs, acc << 8 | len(dist), local, 0, 0, 1 << acc)

    acc_llen, dl = build_dtable("LLen")
    acc_offs, do = build_dtable("Offs")
    acc_mlen, dm = build_dtable("MLen")

    src_offs, src_size = r["seq_stream_to_ref"]
    block_pad = block + b"\x00" * ((-len(block)) % 4)

    fse = [0] * 2432
    sl = cpu_ref_dsq.compute_fse_data_start_llen(0, 1)
    so = cpu_ref_dsq.compute_fse_data_start_offs(0, 1)
    sm = cpu_ref_dsq.compute_fse_data_start_mlen(0, 1)
    for j, v in enumerate(dl):
        fse[sl + j] = v
    for j, v in enumerate(do):
        fse[so + j] = v
    for j, v in enumerate(dm):
        fse[sm + j] = v

    flat = run_oracle(comp, expected=payload)["SEQ"][0]
    num_seqs = len(flat) // 3

    _, folded, _, fo1, fo2, fo3, _ = cpu_ref_dsq.decompress_sequences_single_stream(
        block_pad, src_offs, src_size, fse, sl, so, sm, acc_llen, acc_offs, acc_mlen, 0, num_seqs)

    per_seq_stream_final1 = [fo1]
    per_seq_stream_final2 = [fo2]
    per_seq_stream_final3 = [fo3]
    return comp, payload, r, folded, fo1, fo2, fo3, num_seqs


def main():
    payload = make_payload(int(os.environ.get("DSQ_N", 541)), int(os.environ.get("DSQ_SEED", 1)))
    comp, pl, r, folded, fo1, fo2, fo3, num_seqs = decode_block(payload)
    d = run_oracle(comp, expected=pl)
    truth = [x[0] for x in d["OFFS"]]

    # single-block / single-seq-stream frame: frameCnt=1, 1 seq stream, frame 0 starts at stream 0.
    resolved = cpu_ref_fso.finalise_sequence_offsets(
        folded, num_seqs, seq_stream_cnt=1,
        per_seq_stream_seq_start=[0], seq_stream_to_block_id=[0],
        per_frame_block_count_all=[1], frame_cnt=1,
        per_frame_seq_stream_min_idx=[0],
        per_seq_stream_final_offset1=[fo1], per_seq_stream_final_offset2=[fo2],
        per_seq_stream_final_offset3=[fo3])

    mism = 0
    for i in range(num_seqs):
        if resolved[i] != truth[i]:
            mism += 1
            if mism <= 5:
                print(f"  seq {i}: res={resolved[i]} oracle={truth[i]} folded={folded[i]}")
    ok = mism == 0 and len(resolved) == len(truth)
    print(f"{'PASS' if ok else 'FAIL'}: FinaliseSequenceOffsets ({num_seqs} seqs, {mism} mismatch)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
