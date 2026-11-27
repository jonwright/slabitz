#!/usr/bin/env python3
"""Validate the CPU literal-decompression reference against the oracle.

Chain: parse a real block (CPU) -> build the HufW FSE DTable (cpu_ref_ifs) ->
DecompressHuffmanWeights (cpu_ref_dhw) -> build the Huffman code table
(cpu_ref_dl.preinit+expand) -> decompress the literal streams (cpu_ref_dl) ->
compare the literal bytes to the educational decoder's `LITS` hook (byte-exact).

Usage:
    run_lit_chain.py   (no GPU needed; the oracle is the only reference)
Env: LIT_N and LIT_SEED select the payload (defaults 541/1).
"""
import os
import random
import sys
import zstandard as zstd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from slabitz import zstd_frame
from kernels import cpu_ref_pcb, cpu_ref_ifs, cpu_ref_dhw, cpu_ref_dl
from tools.edu_oracle import run_oracle


def make_payload(n=541, seed=1):
    random.seed(seed)
    return (b"hello world " * 20 + b"\n" + bytes(random.randrange(32, 127) for _ in range(n - 200)))[:n]


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
    print(f"block={n} B acc={acc} weights[{offs}:{offs+size}] lit_streams={len(lit_streams)} "
          f"sizes={[s[1] for s in lit_streams]} dst={[s[3] for s in lit_streams]}")

    # ground truth literals from the DS-shipped educational decoder (checked byte-exact)
    truth = run_oracle(comp, expected=payload)["LITS"][0]
    print(f"oracle literals len={len(truth)}")

    # --- build HufW FSE DTable + decode weights (validated cpu_ref_ifs / cpu_ref_dhw) ---
    dtable = cpu_ref_ifs.ref_initfse_table(probs, info, 0, 0, 0, 1 << acc)
    fse_elems = [0] * 512
    base = cpu_ref_dhw.compute_fse_data_start_hufw(0, 1)
    for j, v in enumerate(dtable):
        fse_elems[base + j] = v
    cw, ccount = cpu_ref_dhw.decompress_huffman_weights(block, offs, size, fse_elems, acc, 1, 0)
    print(f"decoded {ccount} weights")

    # --- build Huffman table ---
    bits_max, code_table_size, rank_index, rcp, cas = cpu_ref_dl.preinit_huffman_table(cw, ccount)
    huff_table = cpu_ref_dl.expand_huffman_table(bits_max, code_table_size, rank_index, cas)
    print(f"Huffman bits_max={bits_max} code_table_size={code_table_size} "
          f"state_pair_cnt={len(huff_table)}")

    # --- decode literals ---
    lits = cpu_ref_dl.decompress_huffman_literals(block, lit_streams, huff_table, bits_max)
    print(f"CPU literals len={len(lits)}")

    mism = [i for i in range(min(len(lits), len(truth))) if lits[i] != truth[i]]
    ok = (not mism) and len(lits) == len(truth)
    print(f"{'PASS' if ok else 'FAIL'}: CPU literal ref vs oracle ({len(lits)} vs {len(truth)}, "
          f"{len(mism)} mismatch)")
    if mism:
        print("  first mismatches (idx, cpu, oracle):", [(i, lits[i], truth[i]) for i in mism[:15]])
    print("compared %d bytes" % (len(lits) if ok else 0))
    sys.exit(0 if ok and len(lits) > 0 else 1)


if __name__ == "__main__":
    sys.exit(main())
