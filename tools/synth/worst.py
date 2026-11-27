"""Named worst cases for the decoders (and later the encoders).

Most are built in "stream space": the bytes the entropy coder sees after the
bitshuffle transpose.  ``unshuffle`` turns a stream into the element data that
produces it, so a case can be written both as a plain zstd frame (the stream
itself) and as filter-32008 HDF5 (the data; the filter transposes it back).
A few are hand-built frames libzstd will not emit (Huffman literals with a
chosen code, zero-sequence blocks, a block of exactly 131072 bytes).

``build`` writes each case into a directory; ``check`` measures the property
the case claims and returns a table.
"""

import json
import os
import struct
import subprocess

import numpy as np

from . import metrics, zstdlib as Z

BLOCK = 131072                      # zstd's largest block


# --------------------------------------------------------------- transpose
def shuffle(data, itemsize):
    """Bitshuffle of one block (data: 1-D array of unsigned ints, n % 8 == 0):
    the stream the codec sees, as a uint8 array."""
    return metrics.bitshuffle_planes(np.ascontiguousarray(data, dtype="<u%d" % itemsize)).reshape(-1)


def unshuffle(stream, itemsize):
    """The element array whose bitshuffle is `stream`."""
    st = np.asarray(stream, dtype=np.uint8)
    planes = st.reshape(8 * itemsize, -1)
    bits = np.unpackbits(planes, axis=1, bitorder="little")        # (8*is, n)
    by = np.packbits(bits.T, axis=1, bitorder="little")            # (n, is)
    return np.ascontiguousarray(by).view("<u%d" % itemsize).reshape(-1)


def lz4_offsets(block):
    """All match offsets of one LZ4 block (the payload after the length word)."""
    i, offs, n = 0, [], len(block)
    while i < n:
        tok = block[i]
        i += 1
        ll = tok >> 4
        if ll == 15:
            while True:
                b = block[i]; i += 1; ll += b
                if b != 255:
                    break
        i += ll
        if i >= n:
            break
        offs.append(block[i] | (block[i + 1] << 8))
        i += 2
        ml = tok & 15
        if ml == 15:
            while True:
                b = block[i]; i += 1; ml += b
                if b != 255:
                    break
    return offs


# ------------------------------------------------------------ plain zstd
def zstd(data, level=3, extra=(), fcs=True):
    """A zstd frame of data from the zstd CLI.  Reading stdin the CLI cannot know
    the size, so without --stream-size the frame has no Frame_Content_Size (which
    the decoder rejects: AGENTS.md, known limits)."""
    cmd = ["/bin/zstd", "-%d" % level, "-q", "-c", "--no-check", "--single-thread"]
    if fcs:
        cmd.append("--stream-size=%d" % len(data))
    r = subprocess.run(cmd + list(extra), input=bytes(data), capture_output=True, check=True)
    return r.stdout


def zstd_decode(z):
    return subprocess.run(["/bin/zstd", "-d", "-q", "-c"], input=z, capture_output=True, check=True).stdout


# ------------------------------------------------- hand-built Huffman literals
def _huf_codes(lens):
    """Canonical codes the way zstd assigns them: by weight (longest first),
    by symbol within a weight.  lens: {symbol: length}, a complete code."""
    table_log = max(lens.values())
    weights = {s: table_log + 1 - L for s, L in lens.items()}
    codes, nxt = {}, 0
    for w in range(1, table_log + 1):
        for s in sorted(s for s, x in weights.items() if x == w):
            codes[s] = (nxt >> (w - 1), table_log + 1 - w)
            nxt += 1 << (w - 1)
    assert nxt == 1 << table_log, "code is not complete"
    return codes, weights, table_log


def _huf_tree(weights):
    """Direct weights (4 bits each); the last symbol's weight is implied."""
    maxs = max(weights)
    ws = [weights.get(s, 0) for s in range(maxs)]
    assert len(ws) <= 128
    if len(ws) % 2:
        ws.append(0)
    body = bytes((ws[i] << 4) | ws[i + 1] for i in range(0, len(ws), 2))
    return bytes([127 + (maxs)]) + body


def _huf_stream(data, codes):
    d = np.frombuffer(bytes(data), dtype=np.uint8).astype(np.int64)[::-1]
    code = np.array([codes[s][0] if s in codes else 0 for s in range(256)], dtype=np.int64)[d]
    ln = np.array([codes[s][1] if s in codes else 0 for s in range(256)], dtype=np.int64)[d]
    pos = np.concatenate([[0], np.cumsum(ln)])
    total = int(pos[-1])
    bits = np.zeros(total + 1, dtype=np.uint8)
    for j in range(int(ln.max())):
        m = ln > j
        bits[pos[:-1][m] + j] = (code[m] >> j) & 1
    bits[total] = 1
    return np.packbits(bits, bitorder="little").tobytes()


def huffman_literals(data, lens, tree=True, last=False, streams=4):
    """A compressed block: Huffman literals with the given code lengths and no
    sequences.  tree False makes it a treeless block (reuses the last table)."""
    codes, weights, tl = _huf_codes(lens)
    data = bytes(data)
    n = len(data)
    if streams == 4:
        seg = (n + 3) // 4
        parts = [data[i * seg:(i + 1) * seg] for i in range(4)]
        enc = [_huf_stream(p, codes) for p in parts]
        body = struct.pack("<HHH", len(enc[0]), len(enc[1]), len(enc[2])) + b"".join(enc)
    else:
        body = _huf_stream(data, codes)
    if tree:
        body = _huf_tree(weights) + body
    csize = len(body)
    kind = 2 if tree else 3
    if streams == 1 and n < 1024 and csize < 1024:
        hdr = struct.pack("<I", kind | (0 << 2) | (n << 4) | (csize << 14))[:3]
    elif streams == 4 and n < 1024 and csize < 1024:
        hdr = struct.pack("<I", kind | (1 << 2) | (n << 4) | (csize << 14))[:3]
    elif n < 16384 and csize < 16384:
        hdr = struct.pack("<I", kind | (2 << 2) | (n << 4) | (csize << 18))
    else:
        hdr = struct.pack("<Q", kind | (3 << 2) | (n << 4) | (csize << 22))[:5]
    blk = hdr + body + b"\x00"
    return Z.block_header(last, 2, len(blk)) + blk


def geometric_lens(depth):
    """Code lengths 1, 2, ..., depth, depth: a complete code of maximum depth."""
    lens = {s: s + 1 for s in range(depth)}
    lens[depth] = depth
    return lens


def sample(lens, n, rng):
    syms = np.array(sorted(lens))
    p = np.array([2.0 ** -lens[s] for s in syms])
    return rng.choice(syms, size=n, p=p / p.sum()).astype(np.uint8)


# ---------------------------------------------------------------- the cases
class Case:
    def __init__(self, name, claim, raw=None, zst=None, h5=None, meta=None):
        self.name, self.claim, self.raw, self.zst = name, claim, raw, zst
        self.h5 = h5 or []          # [(file name, array/dict, enc spec)]
        self.meta = meta or {}


def _words_stream(n, rng, nwords=64):
    words = rng.integers(0, 256, size=(nwords, 4), dtype=np.uint8)
    return words[rng.integers(0, nwords, size=n // 4)].reshape(-1)


def _short_offsets(rng, n=BLOCK):
    out = bytearray()
    p = 1
    while len(out) < n:
        unit = rng.integers(0, 256, p, dtype=np.uint8).tobytes()
        out += rng.integers(0, 256, 3, dtype=np.uint8).tobytes()
        out += (unit * (300 // p + 2))[:300]
        p = p % 31 + 1
    return np.frombuffer(bytes(out[:n]), dtype=np.uint8).copy()


def _far_offsets(rng):
    r = rng.integers(0, 256, BLOCK, dtype=np.uint8)
    b1 = np.concatenate([r[1000:5000], rng.integers(0, 256, BLOCK - 4000, dtype=np.uint8)])
    b2 = np.concatenate([rng.integers(0, 256, 2000, dtype=np.uint8), r[:3000],
                         rng.integers(0, 256, BLOCK - 5000, dtype=np.uint8)])
    return np.concatenate([r, b1, b2])


def _fse_max(rng, n=BLOCK):
    out = bytearray(rng.integers(0, 256, 4096, dtype=np.uint8).tobytes())
    while len(out) < n:
        ll = int(rng.integers(0, 21))
        ml = int(rng.integers(8, 41))
        out += rng.integers(0, 256, ll, dtype=np.uint8).tobytes()
        maxoff = len(out) - ml
        off = int(np.exp(rng.uniform(0, np.log(maxoff))))
        off = max(1, min(off, maxoff))
        start = len(out) - off
        out += out[start:start + ml]
    return np.frombuffer(bytes(out[:n]), dtype=np.uint8).copy()


def _alternating(itemsize, nelem, rng):
    nplanes = 8 * itemsize
    st = np.zeros((nplanes, nelem // 8), dtype=np.uint8)
    for k in range(1, nplanes, 2):
        st[k] = rng.integers(0, 256, nelem // 8, dtype=np.uint8)
    return st.reshape(-1)


def build_cases(rng_seed=7):
    """Return the list of Case objects (data in memory; ``write`` stores them)."""
    R = lambda i: np.random.default_rng([rng_seed, i])
    cases = []

    # incompressible
    for es, dt in ((1, "u8"), (2, "u16"), (4, "u32"), (8, "u64")):
        a = R(es).integers(0, 1 << (8 * es), 65536 // es, dtype="u%d" % es) if es < 8 else \
            R(es).integers(0, 1 << 63, 8192, dtype="u8")
        raw = a.tobytes()
        cases.append(Case("random_" + dt, "incompressible: every zstd block raw", raw=raw, zst=zstd(raw),
                          h5=[("lz4_8k", a), ("zstd2_256k", a)], meta={"prop": "all_raw"}))

    # trivial
    z = np.zeros(32768, dtype="<u2")
    cases.append(Case("zeros", "zero frame: RLE or tiny blocks", raw=z.tobytes(), zst=zstd(z.tobytes()),
                      h5=[("lz4_8k", z), ("zstd2_256k", z)], meta={"prop": "tiny"}))
    for es in (2, 4):
        m = np.full(32768, (1 << (8 * es)) - 1, dtype="<u%d" % es)
        cases.append(Case("all_max_u%d" % (8 * es), "every pixel 2^n-1 (masked-detector extreme)",
                          raw=m.tobytes(), zst=zstd(m.tobytes()),
                          h5=[("lz4_8k", m), ("zstd2_256k", m)], meta={"prop": "tiny"}))
    h = np.zeros(65536, dtype="<u2")
    h[12345] = 1
    cases.append(Case("single_hot_pixel", "one non-zero element in 65536", raw=h.tobytes(), zst=zstd(h.tobytes()),
                      h5=[("lz4_8k", h), ("zstd2_256k", h)], meta={"prop": "tiny"}))

    # plane structure: planes alternately zero and dense
    es, ne = 2, 65536
    st = _alternating(es, ne, R(20))
    a = unshuffle(st, es)
    cases.append(Case("alternating_planes", "bit planes alternately zero and dense (off=P / off=1 switches)",
                      raw=st.tobytes(), zst=zstd(st.tobytes()), h5=[("zstd2_256k", a.astype("<u2"))],
                      meta={"prop": "alt_planes"}))

    # most sequences
    st = _words_stream(BLOCK, R(21))
    cases.append(Case("min_matches", "4-byte matches back to back: the most sequences per block",
                      raw=st.tobytes(), zst=zstd(st.tobytes(), 1, ["--zstd=mml=4"]),
                      meta={"prop": "min_matches"}))

    st = _short_offsets(R(22))
    cases.append(Case("short_offsets_1_31", "overlapping matches at offsets 1-31 (the DS swizzle loop)",
                      raw=st.tobytes(), zst=zstd(st.tobytes()), meta={"prop": "short_offsets"}))

    st = _far_offsets(R(23))
    cases.append(Case("far_offsets", "zstd matches reaching back more than one block",
                      raw=st.tobytes(), zst=zstd(st.tobytes()), meta={"prop": "far_offsets"}))
    # LZ4: 8-bit elements, one 128 kB block, a match at offset 65535
    r = R(24)
    s = r.integers(0, 256, BLOCK, dtype=np.uint8)
    s[70000:] = s[70000 - 65535:BLOCK - 65535]
    cases.append(Case("far_offsets_lz4", "LZ4 match at offset 65535 (the maximum)",
                      raw=s.tobytes(), h5=[("lz4_128k", unshuffle(s, 1).astype("u1"))],
                      meta={"prop": "lz4_far"}))

    # Huffman literals
    r = R(30)
    lens = geometric_lens(11)
    lit = sample(lens, 100000, r)
    blk = huffman_literals(lit, lens, last=True)
    zf = Z.frame_header(len(lit), window_log=17) + blk
    cases.append(Case("huffman_max_depth", "Huffman table of the maximum 11 bits", raw=lit.tobytes(), zst=zf,
                      meta={"prop": "huf_depth", "depth": 11}))
    lens2 = {s: 7 for s in range(128)}                                 # 128 symbols of 7 bits
    lit = r.integers(0, 128, BLOCK, dtype=np.uint8)
    blk = huffman_literals(lit, lens2, last=True)
    cases.append(Case("literals_128k", "one block that is all literals: 131072 Huffman-coded bytes",
                      raw=lit.tobytes(), zst=Z.frame_header(len(lit), window_log=17) + blk,
                      meta={"prop": "lit128k"}))

    # FSE tables at maximum accuracy
    st = _fse_max(R(31))
    cases.append(Case("fse_max_accuracy", "LL 9, OF 8, ML 9 accuracy logs",
                      raw=st.tobytes(), zst=zstd(st.tobytes(), 19, ["--zstd=mml=4"]),
                      meta={"prop": "fse_max"}))

    # an expected reject: a frame without Frame_Content_Size
    spots = (np.random.default_rng([rng_seed, 50]).random(65536) < 0.02).astype(np.uint8)
    cases.append(Case("no_fcs_stream", "streaming frame without Frame_Content_Size (decoder must reject until T09/T10)",
                      raw=spots.tobytes(), zst=zstd(spots.tobytes(), fcs=False),
                      meta={"prop": "no_fcs", "expected_reject": True}))

    # blocks with zero sequences (raw, RLE and Huffman literals, treeless)
    r = R(32)
    l1 = geometric_lens(6)
    d1, d2, d3 = sample(l1, 5000, r), sample(l1, 7000, r), sample(l1, 300, r)
    d0 = bytes(r.integers(0, 256, 3000, dtype=np.uint8))
    blocks = (Z.literals_block(d0) + Z.rle_literals_block(7, 4000) + huffman_literals(d1, l1) +
              huffman_literals(d2, l1, tree=False) + huffman_literals(d3, l1, tree=False, streams=1) +
              Z.raw_block(b"tail") + Z.rle_block(9, 100, last=True))
    raw = d0 + bytes([7]) * 4000 + d1.tobytes() + d2.tobytes() + d3.tobytes() + b"tail" + bytes([9]) * 100
    cases.append(Case("zero_seq_blocks", "compressed blocks with no sequences: raw, RLE, Huffman, treeless literals",
                      raw=raw, zst=Z.frame_header(len(raw), window_log=17) + blocks, meta={"prop": "zero_seq"}))

    # tiny frames, many frames, block tails (HDF5 only: the shape is the point)
    r = R(40)
    dts = {1: "u1", 2: "<u2", 4: "<u4", 8: "<u8"}
    tiny = {"u%d_k%d" % (8 * es, k): r.poisson(3, k).astype(dts[es]) for es in dts for k in range(1, 8)}
    cases.append(Case("tiny_frames", "chunks of 1-7 elements (tail only), every element size",
                      raw=b"".join(a.tobytes() for a in tiny.values()), zst=None,
                      h5=[("lz4_8k", tiny), ("zstd2_256k", tiny)], meta={"prop": "tiny_frames"}))
    many = {"u%d" % (8 * es): r.poisson(2, (1000, 8)).astype(dts[es]) for es in (2, 4)}
    cases.append(Case("many_frames", "1000 chunks of 8 elements each",
                      raw=b"".join(a.tobytes() for a in many.values()), zst=None,
                      h5=[("lz4_8k", many), ("zstd2_256k", many)], meta={"prop": "many_frames"}))
    for enc, bb in (("lz4_8k", 8192), ("zstd2_256k", 262144)):
        tails = {}
        for es in dts:
            be = bb // es
            for n in [5, 8, 9, 1000, be + 5] + [2 * be + t for t in range(8)]:
                tails["u%d_n%d" % (8 * es, n)] = r.poisson(1.5, n).astype(dts[es])
        cases.append(Case("block_tail_mix_" + enc.split("_")[0], "chunk lengths with every tail 0-7 and a partial last block, every element size",
                          raw=b"".join(a.tobytes() for a in tails.values()), zst=None,
                          h5=[(enc, tails)], meta={"prop": "block_tails", "block_bytes": bb}))

    # a block of exactly 131072 bytes
    lits = bytes(R(33).integers(0, 256, 131068, dtype=np.uint8))
    blk = Z.literals_block(lits, last=True)
    assert len(blk) == 3 + 131072
    cases.append(Case("exact_128k_block", "compressed block of exactly 131072 bytes (decoder must accept; encoder must avoid)",
                      raw=lits, zst=Z.frame_header(len(lits), window_log=17) + blk, meta={"prop": "exact128k"}))
    return cases


# ------------------------------------------------------------------- output
def _sha(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def write(cases, outdir, with_large=True):
    """Write every case to outdir/<name>/{raw.bin, data.zst, <enc>.h5, manifest.json}."""
    from . import encode
    names = []
    for c in cases:
        d = os.path.join(outdir, c.name)
        os.makedirs(d, exist_ok=True)
        files = {}
        if c.raw is not None:
            p = os.path.join(d, "raw.bin")
            open(p, "wb").write(c.raw)
            files["raw.bin"] = _sha(p)
        if c.zst is not None:
            p = os.path.join(d, "data.zst")
            open(p, "wb").write(c.zst)
            files["data.zst"] = _sha(p)
        for enc, arr in c.h5:
            p = os.path.join(d, enc + ".h5")
            encode.write_h5(arr if isinstance(arr, dict) else np.asarray(arr), p, enc)
            files[enc + ".h5"] = _sha(p)
        man = {"case": c.name, "claim": c.claim, "prop": c.meta.get("prop"), "files": files,
               "expected_reject": bool(c.meta.get("expected_reject"))}
        with open(os.path.join(d, "manifest.json"), "w") as fh:
            json.dump(man, fh, indent=1, sort_keys=True)
            fh.write("\n")
        names.append(c.name)
    return names


def _oracle(z, raw):
    from tools.edu_oracle import run_oracle
    return run_oracle(z, expected=raw)


def _seq_counts(d):
    return sum(len(v) // 3 for v in d.get("SEQ", []))


def check(cases):
    """Measure each case's claimed property.  Returns [(case, claim, measured, ok)]."""
    rows = []
    for c in cases:
        prop = c.meta.get("prop")
        meas, ok = "", True
        if c.zst is not None and c.raw is not None:
            dec_ok = zstd_decode(c.zst) == bytes(c.raw)
            if not dec_ok:
                rows.append((c.name, c.claim, "zstd -d output differs from raw", False))
                continue
        try:
            if prop == "all_raw":
                m = Z_mix(c.zst)
                meas, ok = "raw %d compressed %d rle %d" % (m["raw"], m["compressed"], m["rle"]), \
                    m["raw"] >= 1 and m["compressed"] == 0 and m["rle"] == 0
            elif prop == "tiny":
                r = len(c.zst) / len(c.raw)
                meas, ok = "%d bytes (ratio %.2g)" % (len(c.zst), r), r < 2e-3
            elif prop == "alt_planes":
                pl = metrics.bitshuffle_planes(np.asarray(c.h5[0][1], dtype="<u2"))
                zf = float((~pl.any(axis=1)).mean())
                meas, ok = "zero planes %.3f" % zf, abs(zf - 0.5) < 1e-9
            elif prop == "min_matches":
                d = _oracle(c.zst, bytes(c.raw))
                n = _seq_counts(d)
                bound = len(c.raw) // 4
                meas, ok = "%d sequences, %.3f of the 4-byte bound %d" % (n, n / bound, bound), n >= 0.9 * bound
            elif prop == "short_offsets":
                d = _oracle(c.zst, bytes(c.raw))
                offs = {v[0] for v in d["OFFS"]}
                got = sorted(o for o in offs if 1 <= o <= 31)
                meas, ok = "offsets 1-31 seen: %d of 31" % len(got), len(got) >= 28
            elif prop == "far_offsets":
                d = _oracle(c.zst, bytes(c.raw))
                mx = max(v[0] for v in d["OFFS"])
                meas, ok = "max offset %d (block 131072)" % mx, mx >= 131072
            elif prop == "lz4_far":
                ch = metrics.h5_chunks(np.asarray(c.h5[0][1], dtype=np.uint8).reshape(1, -1), "lz4_128k")[0]
                _, _, blocks = metrics.chunk_blocks(ch, 1)
                mx = max(max(lz4_offsets(b) or [0]) for b in blocks)
                meas, ok = "max LZ4 offset %d" % mx, mx == 65535
            elif prop == "huf_depth":
                f = Z.analyse(c.zst)[0]["blocks"][0]
                meas, ok = "table log %s (%s weights)" % (f.get("huf_table_log"), f.get("huf_weights")), \
                    f.get("huf_table_log") == 11
                _oracle(c.zst, bytes(c.raw))
            elif prop == "lit128k":
                f = Z.analyse(c.zst)[0]["blocks"][0]
                meas = "%s literals, regen %d, %d streams, nseq %d" % (f["lits"], f["lit_regen"], f["lit_streams"], f["nseq"])
                ok = f["lits"] == "compressed" and f["lit_regen"] == 131072 and f["nseq"] == 0
                _oracle(c.zst, bytes(c.raw))
            elif prop == "fse_max":
                blks = [b for f in Z.analyse(c.zst) for b in f["blocks"] if b["type"] == "compressed"]
                best = max(blks, key=lambda b: sum(x or 0 for x in b["accuracy"].values()) if b.get("nseq") else 0)
                acc = best["accuracy"]
                meas = "LL %s OF %s ML %s, %d sequences, modes %s" % (acc["ll"], acc["of"], acc["ml"], best["nseq"],
                                                                   "/".join(best["modes"].values()))
                ok = (acc["ll"], acc["of"], acc["ml"]) == (9, 8, 9)
                _oracle(c.zst, bytes(c.raw))
            elif prop == "zero_seq":
                blks = [b for f in Z.analyse(c.zst) for b in f["blocks"] if b["type"] == "compressed"]
                kinds = sorted({b["lits"] for b in blks})
                meas = "%d compressed blocks, all nseq 0: %s; literals %s" % (
                    len(blks), all(b["nseq"] == 0 for b in blks), ",".join(kinds))
                ok = all(b["nseq"] == 0 for b in blks) and kinds == ["compressed", "raw", "rle", "treeless"]
                _oracle(c.zst, bytes(c.raw))
            elif prop in ("tiny_frames", "many_frames", "block_tails", "large"):
                meas, ok = _check_h5(c, prop)
            elif prop == "no_fcs":
                f = Z.analyse(c.zst)[0]
                meas, ok = "FCS %s (frame content size absent)" % f["fcs"], f["fcs"] is None
            elif prop == "exact128k":
                blks = [b for f in Z.analyse(c.zst) for b in f["blocks"]]
                meas = "block types/sizes: %s" % ", ".join("%s %d" % (b["type"], b["size"]) for b in blks)
                ok = any(b["type"] == "compressed" and b["size"] == 131072 for b in blks)
                _oracle(c.zst, bytes(c.raw))
            else:
                meas = "no check defined"
        except Exception as e:                                   # noqa: BLE001
            meas, ok = "ERROR %s: %s" % (type(e).__name__, e), False
        rows.append((c.name, c.claim, meas, ok))
    return rows


def _check_h5(c, prop):
    import h5py
    import hdf5plugin  # noqa: F401
    from . import encode
    import tempfile
    enc, arrays = c.h5[0]
    arrays = arrays if isinstance(arrays, dict) else {"data": arrays}
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "c.h5")
        encode.write_h5(arrays if len(arrays) > 1 or prop != "large" else arrays["data"], p, enc)
        n = 0
        with h5py.File(p, "r") as f:
            for name, a in arrays.items():
                got = f[name][...]
                if not np.array_equal(got, a):
                    return "dataset %s decoded != written" % name, False
                n += a.nbytes
    if prop == "tiny_frames":
        lens = sorted({a.size for a in arrays.values()})
        return "%d datasets, lengths %s, %d bytes round-tripped" % (len(arrays), lens, n), lens == list(range(1, 8)) and n > 0
    if prop == "many_frames":
        sh = {a.shape for a in arrays.values()}
        return "shapes %s, %d bytes round-tripped" % (sorted(sh), n), sh == {(1000, 8)} and n > 0
    if prop == "block_tails":
        bb = c.meta["block_bytes"]
        tails = set()
        partial = 0
        for a in arrays.values():
            be = bb // a.dtype.itemsize
            tails.add(a.size % 8)
            if a.size % be:
                partial += 1
        return "tails %s, %d of %d datasets end in a partial block, %d bytes round-tripped" % (
            sorted(tails), partial, len(arrays), n), tails == set(range(8)) and partial > 0 and n > 0
    a = arrays["data"]
    bits = int(a.max()).bit_length()
    return "shape %s max %d (%d bit planes used), %d bytes round-tripped" % (a.shape, a.max(), bits, n), bits >= 20


def large_case(seed=11):
    """One high-dynamic-range u32 Eiger2 16M frame, from the ceo2 pattern scaled
    up 3000 times (values to about 3e7: every low and middle bit plane used)."""
    from . import keys as K, twin
    params, arrays = twin.load_params("ceo2", K.PARAMS_DIR)
    fr = twin.frames(params, arrays, 1, seed, dtype="u4", shape=(4362, 4148), lam_scale=3000.0)
    return Case("large_frame_16m_u32", "one 16M u32 frame of high dynamic range (large: not in the default set)",
                raw=fr.tobytes(), h5=[("lz4_8k", {"data": fr[0]}), ("zstd2_256k", {"data": fr[0]})],
                meta={"prop": "large"})


def Z_mix(z):
    return Z.block_mix(z)
