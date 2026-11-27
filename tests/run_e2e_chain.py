#!/usr/bin/env python3
"""End-to-end decode proof: combine MY CPU-decoded literals + sequences -> decompressed output.

Runs the full CPU chain (parse -> HufW DTable + DHW -> Huffman table -> literals; then the
LLen/Offs/MLen DTables + DecompressSequences + FinaliseSequenceOffsets) and reconstructs the
output with a scalar ExecuteSequences (literal copy + overlapping match copy using the resolved
byte offsets). Compares the result to the original payload (== libzstd).

Usage: run_e2e_chain.py
Env: E2E_N / E2E_SEED / DSQ_MULTI select the payload.
"""
import os
import random
import sys
import zstandard as zstd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from slabitz import zstd_frame
from kernels import cpu_ref_pcb, cpu_ref_ifs, cpu_ref_dhw, cpu_ref_dl, cpu_ref_dsq, cpu_ref_fso
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
    if os.environ.get("DSQ_RAW"):
        # incompressible random chunks interleaved with a repeated word -> a compressed block whose
        # literal section is RAW (0x40000000) while retaining sequences (matches).
        random.seed(seed)
        word = b"THE_QUICK_BROWN_FOX_"
        parts = []
        for i in range(random.randint(3, 8)):
            parts.append(bytes(random.randrange(0, 256) for _ in range(random.randint(200, 900))))
            parts.append(word + bytes([i]))
        return b"".join(parts)
    random.seed(seed)
    return (b"hello world " * 20 + b"\n" + bytes(random.randrange(32, 127) for _ in range(n - 200)))[:n]


def decode_literals(block, r, acc):
    offs, size = r["huf_refs"][0][1]
    lit_streams = [lr[1] for lr in sorted(r["lit_refs"])]
    dtable = cpu_ref_ifs.ref_initfse_table(r["fse_probs"][0:64], r["fse_info"][256], 0, 0, 0, 1 << acc)
    fse_elems = [0] * 512
    base = cpu_ref_dhw.compute_fse_data_start_hufw(0, 1)
    for j, v in enumerate(dtable):
        fse_elems[base + j] = v
    cw, ccount = cpu_ref_dhw.decompress_huffman_weights(block, offs, size, fse_elems, acc, 1, 0)
    bits_max, cts, rank_index, rcp, cas = cpu_ref_dl.preinit_huffman_table(cw, ccount)
    huff_table = cpu_ref_dl.expand_huffman_table(bits_max, cts, rank_index, cas)
    return cpu_ref_dl.decompress_huffman_literals(block, lit_streams, huff_table, bits_max)


def decode_sequences(block, r):
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
    return block_pad, src_offs, src_size, fse, sl, so, sm, acc_llen, acc_offs, acc_mlen


def execute(literals, llen, mlen, offs, out_len):
    out = bytearray(out_len)
    lp = op = 0
    for l, m, o in zip(llen, mlen, offs):
        out[op:op + l] = literals[lp:lp + l]
        lp += l
        op += l
        src = op - o
        for k in range(m):
            out[op + k] = out[src + k]
        op += m
    out[op:op + len(literals) - lp] = literals[lp:len(literals)]
    return bytes(out)


def main():
    payload = make_payload(int(os.environ.get("E2E_N", 541)), int(os.environ.get("E2E_SEED", 1)))
    comp = zstd.ZstdCompressor(level=5).compress(payload)
    _, blocks = zstd_frame.parse_frame(comp)
    block = [b for b in blocks if b.type_name == "compressed"][0].data
    n = len(block)

    r = cpu_ref_pcb.parse_compressed_block(block, n)
    acc = r["fse_info"][256] >> 8
    literals = decode_literals(block, r, acc)
    print(f"block={n} B literals={len(literals)}")

    block_pad, src_offs, src_size, fse, sl, so, sm, al, ao, am = decode_sequences(block, r)
    flat = run_oracle(comp, expected=payload)["SEQ"][0]
    num_seqs = len(flat) // 3
    ll, folded, ml, fo1, fo2, fo3, _ = cpu_ref_dsq.decompress_sequences_single_stream(
        block_pad, src_offs, src_size, fse, sl, so, sm, al, ao, am, 0, num_seqs)
    offs = cpu_ref_fso.finalise_sequence_offsets(
        folded, num_seqs, 1, [0], [0], [1], 1, [0], [fo1], [fo2], [fo3])
    print(f"seqs={num_seqs}")

    output = execute(literals, ll, ml, offs, len(payload))
    ok = output == payload
    print(f"{'PASS' if ok else 'FAIL'}: end-to-end decode ({len(output)} bytes, "
          f"{'match' if ok else 'mismatch'} libzstd)")
    if not ok:
        for i in range(min(len(output), len(payload))):
            if output[i] != payload[i]:
                print(f"  first diff at {i}: got={output[i]} want={payload[i]}")
                break
        print(f"  lens out={len(output)} payload={len(payload)}")
    print("compared %d bytes" % (len(output) if ok else 0))
    return 0 if ok and len(output) > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
