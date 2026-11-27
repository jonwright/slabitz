#!/usr/bin/env python3
"""Ground-truth oracle for intermediate decoder stages, built from the zstd educational decoder.

DirectStorage ships the educational decoder (ThirdParty/zstd_educational_decoder) and uses it as
its own reference. We compile that copy (vendored in third_party/zstd_educational_decoder) with a few dump hooks injected (text patches below) and
run it on a .zst frame; each hook prints one line `TAG n: v0 v1 ...` to stderr. The final output
is also checked against the expected payload, so a hook can only be trusted if the decode is
byte-exact.

Hooks so far:
  HUFW      -- FSE-decoded Huffman weights, one line per FSE-compressed weight sub-block, in
               decode order (== DecompressHuffmanWeights output; count includes the final symbol).
  LITS      -- decoded literal bytes per compressed-literals block.
  SEQ       -- decoded sequences (literal_length, match_length, coded offset) per compressed block.
  OFFS      -- resolved real byte offset per sequence (== FinaliseSequenceOffsets output).
  RAW_HUFW  -- raw direct-nibble Huffman weights (header >= 128); rarely fired by libzstd.
  LITBLK    -- per compressed block: literals-section type (0 raw, 1 RLE, 2 compressed, 3
               treeless) and its 2-bit size format. Fires for every literals section.
  LITSTREAM -- per compressed/treeless literals section: block_type, size_format, stream count.
               Fires only for compressed/treeless literals (never raw/RLE), so it never emits a
               payload; distinct from the LITS payload tag.
  SEQMODES  -- per compressed block with sequences: LL, OF and ML table modes plus seq count.
  SEQCOUNT  -- per compressed block: the sequence count, including zero (fires for 0-sequence
               blocks, which skip decompress_sequences and so produce no SEQ/SEQMODES).

Add more hooks (literals, sequences, FSE tables) here as later kernels need them.

Usage (library):  from tools.edu_oracle import run_oracle; d = run_oracle(zst_bytes)
                  d["HUFW"] -> list of weight lists
CLI:              edu_oracle.py file.zst
"""
import os, pathlib, subprocess, sys, tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
DS_EDU = pathlib.Path(os.environ.get(
    "DS_EDU_DIR", ROOT / "third_party" / "zstd_educational_decoder"))
BUILD = ROOT / "build" / "edu_oracle"

HOOKS = [
    # FSE-compressed Huffman weights (header < 128), in fse_decode_hufweights. This line is
    # UNGUARDED (not under `#ifdef ZSTDGPU_REFERENCE_STORE_H`), unlike the Report_* calls that
    # are compiled out under `-DDISABLE_ZSTDGPU_REFERENCE_STORE`. So anchors below must be
    # unguarded lines, else the hook is dead code.
    ("    *num_symbs = FSE_decompress_interleaved2(&dtable, weights, in);\n",
     '    { const uint8_t *p_ = weights->ptr - *num_symbs; fprintf(stderr, "HUFW %d:", *num_symbs);\n'
     '      for (int i_ = 0; i_ < *num_symbs; i_++) fprintf(stderr, " %d", p_[i_]);\n'
     '      fprintf(stderr, "\\n"); }\n'),
    # decompressed literal bytes per compressed-literals block, right after the HUF decode.
    # `hook` is appended after the anchor (build() does anchor + hook), so repeat NOTHING here.
    ("    if (num_streams == 1) {\n"
     "        symbols_decoded = HUF_decompress_1stream(&ctx->literals_dtable, &lit_stream, &huf_stream);\n"
     "    } else {\n"
     "        symbols_decoded = HUF_decompress_4stream(&ctx->literals_dtable, &lit_stream, &huf_stream);\n"
     "    }\n",
     '    { fprintf(stderr, "LITS %d:", regenerated_size);\n'
      '      for (int i_ = 0; i_ < regenerated_size; i_++) fprintf(stderr, " %d", (*literals)[i_]);\n'
      '      fprintf(stderr, "\\n"); }\n'),
    # decoded sequences (literal_length, match_length, coded offset) per compressed block, in the
    # oracle's "TAG count: v0 v1 ..." format. Anchor is the whole decode loop; the hook is appended
    # after its closing brace (build() does anchor + hook), so it prints once per block.
    ("    for (size_t i = 0; i < num_sequences; i++) {\n"
     "        // Decode sequences one by one\n"
     "        sequences[i] = decode_sequence(&states, src, &bit_offset, i==num_sequences-1);\n"
     "    }\n",
     '    { fprintf(stderr, "SEQ %d:", (int)(3 * num_sequences));\n'
     '      for (size_t j_ = 0; j_ < num_sequences; j_++) fprintf(stderr, " %u %u %u", sequences[j_].literal_length, sequences[j_].match_length, sequences[j_].offset);\n'
     '      fprintf(stderr, "\\n"); }\n'),
    # resolved real byte offset per sequence (== FinaliseSequenceOffsets output, the zstd
    # compute_offset/value: seq.offset>=3 -> seq.offset-3; repeat offsets resolved via offset_hist).
    # One OFFS line per sequence (sequence index, resolved byte offset), in block order. Anchor is
    # the unguarded `offset = compute_offset(...)` inside execute_sequences; `offset` is in scope.
    ("        size_t const offset = compute_offset(seq, offset_hist);\n",
     '        { fprintf(stderr, "OFFS 1: %zu", offset); fprintf(stderr, "\\n"); }\n'),
    # raw direct-nibble weights (header >= 128): num_symbs = header - 127, 2 weights/byte.
    # Anchor is the odd branch + loop close (contiguous, unguarded). (RAW_HUFW does not fire
    # for libzstd frames -- libzstd always FSE-compresses the weights -- but fires for others.)
    ("                weights[i] = weight_src[i / 2] & 0xf;\n            }\n        }\n",
     '        { fprintf(stderr, "RAW_HUFW %d:", num_symbs);\n'
     '          for (int i_ = 0; i_ < num_symbs; i_++) fprintf(stderr, " %d", weights[i_]);\n'
     '          fprintf(stderr, "\\n"); }\n'),
    # literals-section type and its 2-bit size format, per compressed block. Fires for every
    # literals section (raw, RLE, compressed, treeless) because decode_literals() is the single
    # entry point. block_type: 0=raw, 1=RLE, 2=compressed(HUF table), 3=treeless(reuse table).
    ("    int block_type = (int)IO_read_bits(in, 2);\n"
     "    int size_format = (int)IO_read_bits(in, 2);\n",
     '    { fprintf(stderr, "LITBLK 2: %d %d", block_type, size_format);\n'
     '      fprintf(stderr, "\\n"); }\n'),
    # stream count for Huffman-compressed / treeless literals, plus block_type+size_format so a
    # caller can pair it with the matching LITBLK. Fires only in decode_literals_compressed (so
    # only for compressed/treeless literals, never raw/RLE). num_streams: 1 when size_format==0,
    # else 4 (the spec). block_type and size_format are decode_literals_compressed's parameters.
    ("    if (regenerated_size > MAX_LITERALS_SIZE) {\n"
     "        CORRUPTION();\n"
     "    }\n",
     '    { fprintf(stderr, "LITSTREAM 3: %d %d %d", block_type, size_format, num_streams);\n'
     '      fprintf(stderr, "\\n"); }\n'),
    # sequence-section compression modes (LL, OF, ML table modes) plus the sequence count. Fires
    # once per compressed block that has sequences (decompress_sequences is not reached when there
    # are none). compression_modes bits: 7-6 LL, 5-4 OF, 3-2 ML.
    ("    u8 compression_modes = IO_read_bits(in, 8);\n",
     '    { fprintf(stderr, "SEQMODES 4: %d %d %d %d", (compression_modes >> 6) & 3,\n'
     '              (compression_modes >> 4) & 3, (compression_modes >> 2) & 3, (int)num_sequences);\n'
     '      fprintf(stderr, "\\n"); }\n'),
    # sequence count per compressed block, including zero. decode_sequences returns early on a
    # zero-sequence block (before decompress_sequences), so SEQMODES/SEQ do not fire for those;
    # this tag does. num_sequences is the one robust measure of "0-sequence blocks".
    ("        num_sequences = IO_read_bits(in, 16) + 0x7F00;\n"
     "    }\n",
     '    { fprintf(stderr, "SEQCOUNT 1: %u", (unsigned)num_sequences);\n'
     '      fprintf(stderr, "\\n"); }\n'),
]

MAIN_C = r"""
#include <stdio.h>
#include <stdlib.h>
#include "zstd_decompress.h"
int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: oracle in.zst out.bin\n"); return 2; }
    FILE *f = fopen(argv[1], "rb"); if (!f) return 3;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *src = malloc(n); if (fread(src, 1, n, f) != (size_t)n) return 4; fclose(f);
    size_t cap = ZSTD_get_decompressed_size(src, n);
    if (cap == (size_t)-1) cap = (size_t)n * 64 + (1 << 20);
    unsigned char *dst = malloc(cap ? cap : 1);
    size_t m = ZSTD_decompress(dst, cap, src, n);
    f = fopen(argv[2], "wb"); fwrite(dst, 1, m, f); fclose(f);
    return 0;
}
"""


def build(cc="gcc"):
    exe = BUILD / "oracle"
    src = DS_EDU / "zstd_decompress.c"
    if exe.exists() and exe.stat().st_mtime > max(src.stat().st_mtime, pathlib.Path(__file__).stat().st_mtime):
        return exe
    BUILD.mkdir(parents=True, exist_ok=True)
    text = src.read_text()
    for anchor, hook in HOOKS:
        if text.count(anchor) != 1:
            raise RuntimeError("edu_oracle: hook anchor not found exactly once: %r" % anchor)
        text = text.replace(anchor, anchor + hook)
    (BUILD / "zstd_decompress.c").write_text(text)
    (BUILD / "zstd_decompress.h").write_text((DS_EDU / "zstd_decompress.h").read_text())
    (BUILD / "main.c").write_text(MAIN_C)
    subprocess.run([cc, "-O1", "-DDISABLE_ZSTDGPU_REFERENCE_STORE", "-o", str(exe),
                    str(BUILD / "main.c"), str(BUILD / "zstd_decompress.c")], check=True)
    return exe


def run_oracle(zst, expected=None):
    """Decode `zst` (bytes) with the instrumented educational decoder.

    Returns {tag: [list_of_ints, ...]} plus "_output" (decoded bytes). If `expected` is given,
    raises unless the decoded output equals it.
    """
    exe = build()
    with tempfile.TemporaryDirectory(dir=BUILD) as td:
        i, o = pathlib.Path(td) / "in.zst", pathlib.Path(td) / "out.bin"
        i.write_bytes(zst)
        r = subprocess.run([str(exe), str(i), str(o)], capture_output=True, text=True, check=True)
        out = o.read_bytes()
    if expected is not None and out != expected:
        raise RuntimeError("edu_oracle: decoded output differs from expected payload")
    d = {"_output": out}
    for line in r.stderr.splitlines():
        tag, _, rest = line.partition(" ")
        if ":" not in rest:
            continue
        cnt, _, vals = rest.partition(":")
        v = [int(x) for x in vals.split()]
        assert len(v) == int(cnt), line
        d.setdefault(tag, []).append(v)
    return d


if __name__ == "__main__":
    d = run_oracle(pathlib.Path(sys.argv[1]).read_bytes())
    for k, vs in d.items():
        if k != "_output":
            for v in vs:
                print(k, len(v), v)
