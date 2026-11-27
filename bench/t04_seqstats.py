#!/usr/bin/env python3
"""Sequence statistics of pac-zstd chunks, from the educational decoder (T04 question 1).

  PAC_LIB=.../libpac.so python3 bench/t04_seqstats.py [--case silicon_u32_maskmax-1] [--frames 4]

Encodes the gate's input frames for one real case with the pac-zstd in PAC_LIB, splits each
bitshuffle chunk into its zstd block frames, decodes each with tools/edu_oracle.py (byte-exact
against the frame) and prints, per chunk version: sequences, literal bytes, and match bytes and
counts by offset class (1, P = one bit plane, other), the matches longer than their offset
(merged plane repeats, copied in offset-sized steps by zstdgpu_MatchCopy), and the number of
those steps. No GPU needed.
"""
import argparse
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import gate  # noqa: E402
from tools.edu_oracle import run_oracle  # noqa: E402


def block_frames(chunk):
    """[(zst bytes, uncompressed block bytes)] of one bitshuffle-zstd chunk."""
    total, blk = struct.unpack(">QI", chunk[:12])
    pos, left, out = 12, total, []
    while left > 0:
        (n,) = struct.unpack(">I", chunk[pos:pos + 4])
        out.append((chunk[pos + 4:pos + 4 + n], min(blk, left)))
        pos += 4 + n
        left -= min(blk, left)
    return out


def branch(off, ml):
    """The zstdgpu_MatchCopy branch (current kernel) a match takes, ignoring the block-end clamp
    and the FusedCopy path (llen + mlen within one wave) that comes before it."""
    if off == 1:
        return "off1 fill >=32" if ml >= 32 else "off1 fill <32"
    if off >= min(32, ml):
        if ml >= 64 and off >= 64 and off % 16 == 0:
            return "aligned uint4"
        if ml >= 32 and off >= 32 and off % 4 == 0:
            return "aligned uint32"
        return "byte loop"
    return "swizzle (1<off<32, off<ml)"


def stats(chunks, esize):
    s = dict(blocks=0, seqs=0, lit=0, m1=0, m1n=0, mP=0, mPn=0, mo=0, mon=0, over=0,
             over_bytes=0, steps=0, P_steps_hist={}, br={}, brb={})
    for ch in chunks:
        for zst, ub in block_frames(ch):
            d = run_oracle(zst)
            s["blocks"] += 1
            # bitshuffle: full blocks of ub bytes = ub/esize elements; a plane is elems/8 bytes
            P = ub // esize // 8
            offs = [o[0] for o in d.get("OFFS", [])]
            k = 0
            for seq in d.get("SEQ", []):
                for j in range(0, len(seq), 3):
                    ll, ml = seq[j], seq[j + 1]
                    off = offs[k]
                    k += 1
                    s["seqs"] += 1
                    s["lit"] += ll
                    cls = "m1" if off == 1 else ("mP" if off == P else "mo")
                    br = branch(off, ml)
                    s["br"][br] = s["br"].get(br, 0) + 1
                    s["brb"][br] = s["brb"].get(br, 0) + ml
                    s[cls] += ml
                    s[cls + "n"] += 1
                    if ml > off:
                        s["over"] += 1
                        s["over_bytes"] += ml
                        st = -(-ml // off)
                        if off > 1:
                            s["steps"] += st
                            s["P_steps_hist"][st] = s["P_steps_hist"].get(st, 0) + 1
            assert k == len(offs)
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--case", default="silicon_u32_maskmax-1")
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--codec", default="pac-zstd", help="a gate codec: pac-zstd, zstd3, zstd19")
    a = ap.parse_args()
    data, dt, mask = a.case.split("_", 2)
    fr = gate.cpu_frames_real(data, dt, mask, a.frames)
    chunks = gate.encode_real(a.codec, fr, [None])
    s = stats(chunks, fr.dtype.itemsize)
    print("PAC_LIB=%s (%s)" % (os.environ.get("PAC_LIB"), gate.pac_version()))
    print("codec %s, case %s, %d frames, %d chunk bytes, %d blocks" % (
        a.codec, a.case, a.frames, sum(len(c) for c in chunks), s["blocks"]))
    print("sequences %d, literal bytes %d" % (s["seqs"], s["lit"]))
    for c, name in (("m1", "off 1"), ("mP", "off P"), ("mo", "other")):
        print("  %-6s matches %7d  bytes %10d  mean %8.1f" % (
            name, s[c + "n"], s[c], s[c] / max(1, s[c + "n"])))
    print("  matches longer than their offset: %d (%d bytes); off>1 steps %d" % (
        s["over"], s["over_bytes"], s["steps"]))
    for b in sorted(s["br"]):
        print("  branch %-28s %7d matches %10d bytes" % (b, s["br"][b], s["brb"][b]))
    print("  steps per merged off>1 match: %s" % dict(sorted(s["P_steps_hist"].items())))


if __name__ == "__main__":
    main()
