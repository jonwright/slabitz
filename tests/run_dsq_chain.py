#!/usr/bin/env python3
"""CPU-reference validation for the DecompressSequences stage.

Chain: parse the real block -> build the LLen/Offs/MLen FSE DTables (cpu_ref_ifs) ->
run the SingleStream sequence decoder (cpu_ref_dsq) on the seq-stream segment ->
compare to the educational decoder's `SEQ` hook (flat ll ml off triples).
"""
import os
import random
import sys
import zstandard as zstd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from slabitz import zstd_frame
from kernels import cpu_ref_pcb, cpu_ref_ifs, cpu_ref_dsq
from tools.edu_oracle import run_oracle

FSE_RLE_TABLE_COUNT = 256
FSE_MAX_COUNT_PROBS = 64

# DS predefined sequence-table defaults (kzstdgpuFseProbsDefault, zstdgpu.cpp:94)
DEF_LLEN = [4,3,2,2,2,2,2,2,2,2,2,2,2,1,1,1,2,2,2,2,2,2,2,2,2,3,2,1,1,1,1,1,-1,-1,-1,-1]
DEF_OFFS = [1,1,1,1,1,1,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1]
DEF_MLEN = [1,4,3,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1]
assert len(DEF_LLEN) == 36 and len(DEF_OFFS) == 29 and len(DEF_MLEN) == 53


def make_payload(n=541, seed=1):
    # Default: the classic 'hello world' shape (cross-checks the literals/DHW stages).
    # When DSQ_MULTI is set, build word-soup text so the block carries many short matches
    # and literals (hundreds/thousands of sequences) to exercise the offset-history machinery.
    if os.environ.get("DSQ_MULTI"):
        words = "the quick brown fox jumps over lazy dog run fast stop slow near far".split()
        random.seed(seed)
        w = " ".join(random.choice(words) for _ in range(int(n) if n > 10000 else 12000))
        return w.encode()
    random.seed(seed)
    return (b"hello world " * 20 + b"\n" + bytes(random.randrange(32, 127) for _ in range(n - 200)))[:n]


def main():
    payload = make_payload(int(os.environ.get("DSQ_N", 541)), int(os.environ.get("DSQ_SEED", 1)))
    comp = zstd.ZstdCompressor(level=5).compress(payload)
    _, blocks = zstd_frame.parse_frame(comp)
    block = [b for b in blocks if b.type_name == "compressed"][0].data
    n = len(block)

    r = cpu_ref_pcb.parse_compressed_block(block, n)
    sf = r["seq_fse"]
    info = r["fse_info"]

    if r["seq_stream_to_ref"] is None or sf["LLen"] == 0x3FFFFFFF:
        print("FAIL: no sequence stream in block; nothing compared")
        return 1

    # Sequence FSE tables: mode may be compressed (probs parsed into r["fse_probs"], accuracy
    # in r["fse_info"]) or predefined (mode-0, DS default distributions at accuracy 6/5/6).
    def build_dtable(name):
        ds_idx = sf[name]
        local_idx = ds_idx - FSE_RLE_TABLE_COUNT
        if ds_idx in info:
            acc = info[ds_idx] >> 8
            probs = r["fse_probs"]
            return acc, cpu_ref_ifs.ref_initfse_table(
                probs, info[ds_idx], local_idx, 0, 0, 1 << acc)
        acc = {"LLen": 6, "Offs": 5, "MLen": 6}[name]
        probs = [0] * (16 * FSE_MAX_COUNT_PROBS)
        dist = {"LLen": DEF_LLEN, "Offs": DEF_OFFS, "MLen": DEF_MLEN}[name]
        probs[local_idx * FSE_MAX_COUNT_PROBS:local_idx * FSE_MAX_COUNT_PROBS + len(dist)] = dist
        return acc, cpu_ref_ifs.ref_initfse_table(
            probs, acc << 8 | len(dist), local_idx, 0, 0, 1 << acc)

    acc_llen, dtable_llen = build_dtable("LLen")
    acc_offs, dtable_offs = build_dtable("Offs")
    acc_mlen, dtable_mlen = build_dtable("MLen")

    fse_elems = [0] * 2432
    start_llen = cpu_ref_dsq.compute_fse_data_start_llen(0, 1)
    start_offs = cpu_ref_dsq.compute_fse_data_start_offs(0, 1)
    start_mlen = cpu_ref_dsq.compute_fse_data_start_mlen(0, 1)
    for j, v in enumerate(dtable_llen):
        fse_elems[start_llen + j] = v
    for j, v in enumerate(dtable_offs):
        fse_elems[start_offs + j] = v
    for j, v in enumerate(dtable_mlen):
        fse_elems[start_mlen + j] = v

    src_offs, src_size = r["seq_stream_to_ref"]
    # Pad the block to a dword boundary so the backward bit-buffer can read the final dword
    # (its last read lands at (datasz-1)&~3, which may extend past the raw block by up to 3 bytes).
    block_pad = block + b"\x00" * ((-len(block)) % 4)
    truth_flat = run_oracle(comp, expected=payload)["SEQ"][0]
    truth = [(truth_flat[3 * i], truth_flat[3 * i + 1], truth_flat[3 * i + 2])
             for i in range(len(truth_flat) // 3)]
    num_seqs = len(truth)

    out_llen, out_offs, out_mlen, fo1, fo2, fo3, total_mlen = (
        cpu_ref_dsq.decompress_sequences_single_stream(
            block_pad, src_offs, src_size, fse_elems, start_llen, start_offs, start_mlen,
            acc_llen, acc_offs, acc_mlen, out_start=0, out_count=num_seqs))

    mism = 0
    for i in range(num_seqs):
        ll, ml, off = out_llen[i], out_mlen[i], out_offs[i]
        tl, tm, to = truth[i]
        if ll != tl or ml != tm:
            mism += 1
            print(f"  seq {i}: cfg=({ll},{ml},{off}) oracle=({tl},{tm},{to}) LLEN_MLEN_MISMATCH")
        # Offs: DS emits the offset-history-encoded value; for a non-repeat coded offset (>3)
        # it equals the oracle's coded offset. Rep offsets (to<=3) need matching the folded value.
        if to > 3 and off != to:
            mism += 1
            print(f"  seq {i}: off cfg={off} oracle(coded)={to} OFFS_MISMATCH")

    ok = mism == 0 and len(out_llen) == num_seqs
    print(f"{'PASS' if ok else 'FAIL'}: cfg seqs={len(out_llen)} oracle={num_seqs} mismatches={mism}")
    print(f"  LLen={out_llen} MLen={out_mlen} Offs={out_offs} (final offs {fo1:#x},{fo2:#x},{fo3:#x} totalMLen={total_mlen})")
    print("compared %d sequences" % (num_seqs if ok else 0))
    return 0 if ok and num_seqs > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
