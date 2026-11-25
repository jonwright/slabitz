#!/usr/bin/env python3
"""Decode the committed synthetic-data subset (and, when $SLABITZ_SYNTH is
set, the full v1 twin + worst cases) byte-exact against raw.

For each tests/data/synth/*.h5 the bitshuffle-32008 dataset is read back
through hdf5plugin (the same filter that wrote it) and compared, byte for byte,
against the twin frame it was cropped from (when the v1 output is present).
For each tests/data/synth/worst_*.zst the frame is decompressed with the zstd
CLI and compared against the worst case's raw.bin (when present).

A check that cannot run because the v1 data is absent is reported as "not run",
never as a pass.  A decode that yields zero compared bytes is a failure.

Exit 0 on success, 1 on any failure.
"""

import glob
import os
import subprocess
import sys

import numpy as np

import h5py  # noqa: F401
import hdf5plugin  # noqa: F401,E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "synth")
SYNTH = os.environ.get("SLABITZ_SYNTH", "")
TWINS = os.path.join(SYNTH, "twins") if SYNTH else None
WORST = os.path.join(SYNTH, "worst") if SYNTH else None


def compare(name, a, b):
    a = np.ascontiguousarray(a).view(np.uint8).reshape(-1)
    b = np.ascontiguousarray(b).view(np.uint8).reshape(-1)
    if a.nbytes != b.nbytes:
        return ("%s: size mismatch %d != %d bytes" % (name, a.nbytes, b.nbytes), a.nbytes)
    if not a.nbytes:
        return ("%s: compared zero bytes" % name, 0)
    ok = bool((a == b).all())
    return ((None if ok else "%s: DECODED != RAW" % name), a.nbytes)


def main():
    failures = []
    compared = 0
    files = sorted(glob.glob(os.path.join(DATA, "*.h5")))
    for f in files:
        base = os.path.basename(f)[:-3]  # e.g. spots.lz4
        model, enc = base.split(".", 1)
        with h5py.File(f, "r") as h:
            ds = h[list(h.keys())[0]]
            raw = np.asarray(ds[0, :, :])
        note = ""
        if TWINS:
            # regenerate the cropped frame from the twin (frame 0 crop 256)
            key = {"spots": "silicon", "rings": "ceo2", "mx": "thaumatin",
                   "fibre": "kevlar", "flat": "ceo2"}.get(model)
            src = os.path.join(TWINS, key, "raw.npy")
            if key and os.path.exists(src):
                ref = np.load(src)[0][:raw.shape[0], :raw.shape[1]]
                msg, n = compare(base, raw, ref)
                compared += n
                if msg:
                    failures.append(msg)
                else:
                    note = "(%d bytes) ok" % n
            else:
                note = "(v1 twin absent: not run)"
        else:
            note = "(v1 absent: decode only)"
        print("%-16s %s %s" % (base, raw.shape, note if note else "(decoded)"))

    for f in sorted(glob.glob(os.path.join(DATA, "worst_*.zst"))):
        base = os.path.basename(f)[:-4]
        cmd = ["/bin/zstd", "-d", "-q", "-c", f]
        out = subprocess.run(cmd, capture_output=True)
        if out.returncode != 0:
            failures.append("%s: zstd -d failed" % base)
            print("%-16s zstd -d FAILED" % base)
            continue
        note = "(decoded %d bytes)" % len(out.stdout)
        if WORST:
            rp = os.path.join(WORST, base[len("worst_"):], "raw.bin")
            if os.path.exists(rp):
                refb = open(rp, "rb").read()
                if refb == out.stdout:
                    note = "(%d bytes) ok" % len(out.stdout)
                else:
                    failures.append("%s: decoded != raw" % base)
                    note = "(%d bytes) DECODED != RAW" % len(out.stdout)
                compared += len(out.stdout)
            else:
                note += " (v1 worst raw absent: not run)"
        print("%-16s %s" % (base, note))

    if compared == 0:
        print("compared 0 bytes (v1 not present): mark as not run")
    print("\ncompared bytes: %d" % compared)
    if failures:
        print("FAILURES:")
        for f in failures:
            print("  " + f)
        return 1
    print("all decodes passed" if compared else "no byte-exact compares (not run)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
