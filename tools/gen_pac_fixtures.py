#!/usr/bin/env python3
"""Generate the committed pac fixture set (T02).

Each case is one or more detector frames, cropped to a contiguous range of whole
rows (full row width, so the bit-plane structure is real) and encoded with
libpac.so (the pac-zstd encoder, $PAC_LIB) as filter-32008 chunks, one chunk per
frame.  Output, per case, in data/pac/:

  <case>.chunk.bin   the frames' chunks concatenated (one chunk per frame)
  <case>.index       JSON: [{"offset": N, "length": M}, ...] per frame
  <case>.json        elem_size, block_bytes, per-frame source/crop/mask and the
                     SHA-256 of each decoded frame, plus the pac version/commit

Only these three files are written (the expected output is stored as a hash in
<case>.json, not as a file, to keep the repository small).  The fixture set is
kept under 3 MB.  The all-zero case is a synthetic all-zero u16 frame (no
source file needed).

Usage:  python3 tools/gen_pac_fixtures.py      (needs $PAC_LIB and $SLABITZ_EXAMPLES)

The cases below were chosen so that, per zstd case, the encoder's three fast
match kinds all appear (checked by tests/run_pac_fixtures.py --stats): offsets
1 (runs), off==P with len<=off (a plane copy) and off==P with len>off (a merged
match).  The detector frames are mostly zero, so off==1 and the two off==P
kinds come from the mask fill: mask=0 yields the zero runs, mask=max-1 yields
the constant-fill planes that pac merges.
"""
import ctypes as ct
import hashlib
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PAC_LIB = os.environ.get("PAC_LIB", "")
EX = os.environ.get("SLABITZ_EXAMPLES", "")
OUT = os.path.join(ROOT, "data", "pac")

# pac-zstd source checkout (read only) that built $PAC_LIB
PAC_SRC = os.path.dirname(os.path.dirname(PAC_LIB)) if PAC_LIB else ""

# one case: source file, dataset, frames (indices), rows [r0,r1), dtype, mask,
# codec, block_bytes (0 -> pac's recommended default for the codec)
CASES = [
    ("silicon_u16_zstd_mask0",     "silicon_u16_zstd2_256k.h5", "entry_0000/ESRF-ID11/eiger/data", [25, 26, 27], 512, "u16", 0, "zstd", 0),
    ("silicon_u16_zstd_maskmax",   "silicon_u16_zstd2_256k.h5", "entry_0000/ESRF-ID11/eiger/data", [0, 1], 256, "u16", "max-1", "zstd", 0),
    ("silicon_u32_zstd_mask0",     "silicon_u16_zstd2_256k.h5", "entry_0000/ESRF-ID11/eiger/data", [25, 26, 27], 512, "u32", 0, "zstd", 0),
    ("silicon_u32_zstd_maskmax",   "silicon_u16_zstd2_256k.h5", "entry_0000/ESRF-ID11/eiger/data", [0, 1], 256, "u32", "max-1", "zstd", 0),
    ("ceo2_u32_zstd_maskmax",      "ceo2_u32_zstd2_256k.h5",    "entry_0000/ESRF-ID11/eiger/data", [0, 1], 48, "u32", "max-1", "zstd", 0),
    ("silicon_u16_lz4",            "silicon_u16_zstd2_256k.h5", "entry_0000/ESRF-ID11/eiger/data", [0, 1], 256, "u16", None, "lz4", 0),
    ("all_zero_u16",               None, None, [0], 512, "u16", None, "zstd", 0),
]

PAC_LZ4, PAC_ZSTD = 1, 2
CODEC = {"lz4": PAC_LZ4, "zstd": PAC_ZSTD}
BCODE = {"zstd": 253952, "lz4": 57344}  # pac's recommended block sizes


def load_pac(pac_path):
    pac = ct.CDLL(pac_path)
    pac.pac_create.restype = ct.c_void_p
    pac.pac_create.argtypes = [ct.c_int]
    pac.pac_free.argtypes = [ct.c_void_p]
    pac.pac_bound.restype = ct.c_size_t
    pac.pac_bound.argtypes = [ct.c_size_t, ct.c_size_t, ct.c_size_t]
    pac.pac_compress_chunk.restype = ct.c_size_t
    pac.pac_compress_chunk.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t,
                                       ct.c_void_p, ct.c_size_t, ct.c_size_t, ct.c_size_t]
    return pac


def pac_ident():
    """(version, commit) of the pac-zstd source that built $PAC_LIB."""
    version = "?"
    if PAC_SRC:
        try:
            import re
            txt = open(os.path.join(PAC_SRC, "src", "pac.h")).read()
            m = re.search(r'#define PAC_VERSION "([^"]+)"', txt)
            if m:
                version = m.group(1)
        except Exception:
            pass
    commit = "?"
    if PAC_SRC:
        try:
            commit = subprocess.run(["git", "-C", PAC_SRC, "rev-parse", "HEAD"],
                                    capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            commit = subprocess.run(["git", "-C", PAC_SRC, "describe", "--always", "--dirty"],
                                    capture_output=True, text=True).stdout.strip() or "?"
    return (version, commit)


def frame_from(case, fi, mask4m, datasets):
    """The cropped detector frame for one case frame index."""
    _, src, dsname, frames, rows, dtname, mval, codec, _ = case
    dt = np.uint16 if dtname == "u16" else np.uint32
    dmax = int(np.iinfo(dt).max)
    if src is None:  # synthetic all-zero frame
        W = datasets["width"]
        return np.zeros((rows, W), dtype=dt), {"source": "NONE", "frame": fi, "rows": [0, rows],
                                               "width": W, "mask": None}
    ds = datasets[src]
    base = ds[frames[fi]]
    fr = np.clip(base[:rows].astype(np.uint64), 0, dmax).astype(dt)
    if mval is not None:
        m = int(dmax - 1) if mval == "max-1" else int(mval)
        fr[mask4m[:rows]] = m
    fr = np.ascontiguousarray(fr)
    meta = {"source": src, "dataset": dsname, "frame": int(frames[fi]), "rows": [0, rows],
            "width": int(base.shape[1]), "mask": None if mval is None else (
                "max-1" if mval == "max-1" else int(mval))}
    return fr, meta


def main():
    if not PAC_LIB or not os.path.isfile(PAC_LIB):
        raise SystemExit("gen_pac_fixtures: $PAC_LIB (%s) not set or not a file" % PAC_LIB)
    if not EX or not os.path.isdir(EX):
        raise SystemExit("gen_pac_fixtures: $SLABITZ_EXAMPLES (%s) not set or not a dir" % EX)
    os.makedirs(OUT, exist_ok=True)

    import h5py  # noqa
    import hdf5plugin  # noqa  (reads the bitshuffle datasets)
    pac = load_pac(PAC_LIB)
    encs = {}
    for codec in ("lz4", "zstd"):
        encs[codec] = pac.pac_create(CODEC[codec])

    with h5py.File(os.path.join(EX, "mask.h5"), "r") as f:
        mask4m = f["mask"][:].astype(bool)

    datasets = {"width": None, "_files": {}}
    for case in CASES:
        src = case[1]
        if src and src not in datasets:
            h = h5py.File(os.path.join(EX, src), "r")
            datasets["_files"][src] = h
            datasets[src] = h[case[2]]
            if datasets["width"] is None:
                datasets["width"] = int(datasets[src].shape[2])

    ver, commit = pac_ident()
    total_size = 0

    for case in CASES:
        name, src, dsname, frames, rows, dtname, mval, codec, block = case
        e = 2 if dtname == "u16" else 4
        block_bytes = block or BCODE[codec]
        enc = encs[codec]
        chunks = []
        metas = []
        for fi in range(len(frames)):
            fr, meta = frame_from(case, fi, mask4m, datasets)
            fb = fr.nbytes
            cap = pac.pac_bound(fb, e, block_bytes)
            buf = np.empty(cap, dtype=np.uint8)
            m = pac.pac_compress_chunk(enc, buf.ctypes.data, cap, fr.ctypes.data, fb, e,
                                       block_bytes)
            if m <= 0:
                raise SystemExit("gen_pac_fixtures: %s frame %d: pac_compress_chunk rc=%d"
                                 % (name, fi, m))
            data = bytes(buf[:m].tobytes())
            chunks.append(data)
            meta.update({"elem_size": e, "codec": codec,
                         "decoded_bytes": int(fb),
                         "sha256": hashlib.sha256(fr.tobytes()).hexdigest()})
            metas.append(meta)

        index = []
        off = 0
        for ch in chunks:
            index.append({"offset": off, "length": len(ch)})
            off += len(ch)
        chunk_bin = b"".join(chunks)
        total_size += len(chunk_bin)

        j = {"case": name, "codec": codec, "elem_size": e, "block_bytes": int(block_bytes),
             "frames": metas, "pac_version": ver, "pac_commit": commit}
        base = os.path.join(OUT, name)
        with open(base + ".chunk.bin", "wb") as fh:
            fh.write(chunk_bin)
        with open(base + ".index", "w") as fh:
            json.dump(index, fh)
        with open(base + ".json", "w") as fh:
            json.dump(j, fh, indent=1)
        print("%-26s %8d B, %d frame(s), codec=%s, block=%d"
              % (name, len(chunk_bin), len(chunks), codec, block_bytes))

    for enc in encs.values():
        pac.pac_free(enc)
    print("total data/pac: %d B" % total_size)


if __name__ == "__main__":
    main()
