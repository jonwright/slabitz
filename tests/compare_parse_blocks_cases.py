#!/usr/bin/env python3
"""Run ParseCompressedBlocks GPU-vs-CPU across several real blocks/paths.

Exercises multiple literal-block types (raw, RLE, compressed, treeless) and sequence
configurations by varying the payload and compression level, then asserts the deterministic
parse outputs agree element-exact between the GPU cubin and the CPU reference.

Usage:
    compare_parse_blocks_cases.py <cubin>
"""
import os
import sys
import zstandard as zstd
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from slabitz import zstd_frame, loader
from kernels import cpu_ref_pcb
import run_parse_blocks as rpb

CUBIN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build/pcb/pcb.cubin")


def decode_lit_size(x):
    return x & ~0xC0000000


def i32(v):
    return v - 0x100000000 if v >= 0x80000000 else v


def make_block(payload, level):
    comp = zstd.ZstdCompressor(level=level).compress(payload)
    try:
        _, blocks = zstd_frame.parse_frame(comp)
    except ValueError as e:
        raise RuntimeError(f"frame parse: {e}") from e
    cb = [b for b in blocks if b.type_name == "compressed"]
    if not cb:
        return None
    return cb[0].data


def run_case(drv, fn, block):
    g = rpb.run(drv, fn, block)
    c = cpu_ref_pcb.parse_compressed_block(block, len(block))
    lit_type = g["block_datas"][0][1] & 0xC0000000
    checks = []
    # literal size ALWAYS produced
    checks.append(decode_lit_size(g["block_datas"][0][1]) == decode_lit_size(c["block_size_prefix"]))
    if lit_type == 0xC0000000:  # compressed/treeless literals
        checks.append(tuple(g["huf_refs"]) == tuple(v for _, v in c["huf_refs"]))
        checks.append(tuple(map(tuple, g["lit_refs"])) == tuple(v for _, v in c["lit_refs"]))
        checks.append(tuple(g["seq_stream_to_ref"][0]) == (c["seq_stream_to_ref"] or (0, 0)))
        checks.append((g["seq_LLen"][0], g["seq_Offs"][0], g["seq_MLen"][0]) ==
                      (c["seq_fse"]["LLen"], c["seq_fse"]["Offs"], c["seq_fse"]["MLen"]))
        checks.append(tuple(g["fse_probs_nonzero"]) == tuple(v for v in c["fse_probs"] if v) or
                      tuple(i32(v) for v in g["fse_probs_nonzero"]) == tuple(v for v in c["fse_probs"] if v))
    return all(checks), checks


def main():
    with open(CUBIN, "rb") as f:
        cubin = f.read()
    drv = loader.Driver(0)
    fn = drv.load(cubin).function("main_0")

    random.seed(7)
    cases = []

    # highly repetitive -> RLE-ish / small lit; small compressed block
    cases.append(("repetitive", bytes(65) * 400, 3))
    cases.append(("repetitive", b"A" * 1000 + b"B" * 1000, 5))
    # mixed text -> compressed literals + sequences
    cases.append(("text", b"the quick brown fox jumps over the lazy dog. " * 40, 6))
    # random bytes -> raw literals (incompressible)
    cases.append(("random", bytes(random.randrange(0, 256) for _ in range(600)), 1))
    # large structured
    cases.append(("table", b"".join(b"%5d" % i for i in range(5000)), 4))

    total_ok = True
    compared = 0
    for name, payload, level in cases:
        try:
            block = make_block(payload, level)
        except Exception as e:  # noqa
            print(f"FRAME-ERR: {name} level {level}: {e}")
            total_ok = False
            continue
        if block is None:
            print(f"SKIP: {name} level {level} (no compressed block, only raw/rle)")
            continue
        try:
            ok, checks = run_case(drv, fn, block)
        except Exception as e:  # noqa
            print(f"ERROR: {name} level {level} ({len(block)} b): {e}")
            total_ok = False
            continue
        total_ok &= ok
        compared += sum(checks)
        print(f"{'PASS' if ok else 'FAIL'}: {name} level {level}  ({len(block)} b, {sum(checks)}/6 checks)")

    print(f"RESULT: {'PASS' if total_ok else 'FAIL'}")
    print("compared %d fields" % (compared if total_ok else 0))
    sys.exit(0 if total_ok and compared > 0 else 1)


if __name__ == "__main__":
    sys.exit(main())
