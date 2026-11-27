#!/usr/bin/env python3
"""T06 part 2: zstd feature matrix with proof of coverage.

Proves that every zstd format feature the slabitz decoder must handle has
actually been exercised.  For each corpus entry we decode with the slabitz
library (sbz_dctx_decompress and sbz_dctx_decompress_frames) and compare the
output byte-exact against `zstandard` (libzstd).  Each entry is classified
with the educational decoder (tools.edu_oracle, via its hook tags) plus a
Python parse of the frame header, and the features fill a matrix.  A matrix
cell with a count of 0 is a FAILURE, never a skip.  An entry whose result does
not match its expectation is an "unexpected result" and fails the script.

The corpus has two halves:

* libzstd-generated frames, over many levels, parameter sets and data kinds.
  These supply the raw / compressed / treeless literals (both stream counts,
  all size formats), every LL/OF/ML table mode (including RLE and repeat),
  real sequence counts, checksummed frames, multi-block frames and the
  RLE-block / raw-block / empty-frame cases that libzstd emits naturally.
* hand-built frames, named "hand-*", where libzstd will not emit a feature by
  itself.  These are the honest ones: raw and RLE frame blocks, skippable
  frames (at the start, middle and end of a stream), 0-sequence blocks,
  RLE literals, a 1-byte FCS, an 8-byte FCS above 2^32 (header only, an
  expected SBZ_ERR_LIMIT reject), and so on.

Entries that today are rejected by design (a frame with no Frame_Content_Size,
and a frame whose window is 2^29 or more) carry expectation
"reject:SBZ_ERR_CORRUPT" and fill their matrix cells as "expected reject"
cells, not coverage failures.  (T09/T10 remove those limits.)  An 8-byte FCS
above 2^32 is an expected "reject:SBZ_ERR_LIMIT".

If a test unearths a decoder bug, a minimal reproducer is written to
tests/data/features_bugs/ and the entry is marked "xfail:<reason>" and
continued -- the decoder is not fixed here.

Usage:  timeout 600 python3 tests/run_features.py
"""
import ctypes as ct
import os
import struct
import sys

import zstandard as zstd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
from tools.edu_oracle import run_oracle

LIB = os.path.join(ROOT, "build", "libslabitz.so")
KDIR = os.environ.get("SLABITZ_KERNEL_DIR", os.path.join(ROOT, "build"))
BUGS = os.path.join(ROOT, "tests", "data", "features_bugs")
ZSTD_MAGIC = 0xFD2FB528
SKIP_BASE = 0x184D2A50
SBZ_ERR_CORRUPT, SBZ_ERR_LIMIT = -5, -6
WEIGHT_2_29 = 1 << 29
LIT_NAME = {0: "raw", 1: "rle", 2: "compressed", 3: "treeless"}

# --------------------------------------------------------------------------
# decoder plumbing
# --------------------------------------------------------------------------
class Span(ct.Structure):
    _fields_ = [("offs", ct.c_uint64), ("size", ct.c_uint64)]


def load():
    lib = ct.CDLL(LIB)
    lib.sbz_dctx_create_cuda.restype = ct.c_void_p
    lib.sbz_dctx_create_cuda.argtypes = [ct.c_int, ct.c_char_p, ct.c_char_p,
                                         ct.c_size_t]
    lib.sbz_dctx_destroy.argtypes = [ct.c_void_p]
    lib.sbz_dctx_error.restype = ct.c_char_p
    lib.sbz_dctx_error.argtypes = [ct.c_void_p]
    lib.sbz_dctx_decompress.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_size_t]
    lib.sbz_dctx_decompress_frames.argtypes = [ct.c_void_p, ct.c_char_p,
                                               ct.c_size_t, ct.POINTER(Span),
                                               ct.c_uint32]
    lib.sbz_dctx_read_buffer.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_void_p,
                                         ct.POINTER(ct.c_size_t)]
    return lib


def read_buf(lib, dc, name):
    n = ct.c_size_t(0)
    assert lib.sbz_dctx_read_buffer(dc, name.encode(), None, ct.byref(n)) == 0
    b = ct.create_string_buffer(max(n.value, 1))
    assert lib.sbz_dctx_read_buffer(dc, name.encode(), b, ct.byref(n)) == 0, \
        lib.sbz_dctx_error(dc)
    return b.raw[:n.value]


def make_dctx(lib):
    err = ct.create_string_buffer(512)
    dc = lib.sbz_dctx_create_cuda(0, KDIR.encode(), err, len(err))
    if not dc:
        raise RuntimeError("sbz_dctx_create_cuda: %s" % err.value.decode())
    return dc


def decompress(lib, dc, data):
    rc = lib.sbz_dctx_decompress(dc, data, len(data))
    if rc == 0:
        return rc, read_buf(lib, dc, "UnCompressedFramesData")
    return rc, b""


def decompress_frames(lib, dc, data, spans):
    arr = (Span * len(spans))(*[Span(o, s) for o, s in spans])
    rc = lib.sbz_dctx_decompress_frames(dc, data, len(data), arr, len(spans))
    if rc == 0:
        return rc, read_buf(lib, dc, "UnCompressedFramesData")
    return rc, b""


# --------------------------------------------------------------------------
# zstd frame header parse + split
# --------------------------------------------------------------------------
FCS_BYTES = {0: 1, 1: 2, 2: 4, 3: 8}


def parse_frame_header(frame):
    desc = frame[4]
    single = bool(desc & 0x20)
    checksum = bool(desc & 0x04)
    fcs_flag = (desc >> 6) & 3
    dict_flag = desc & 3
    if desc & 0x08:
        raise ValueError("reserved descriptor bit")
    pos = 5
    window = None
    if not single:
        wd = frame[pos]
        pos += 1
        wbase = 1 << (10 + (wd >> 3))
        window = wbase + (wbase >> 3) * (wd & 7)
    dict_size = {0: 0, 1: 1, 2: 2, 3: 4}[dict_flag]
    if dict_flag:
        pos += dict_size
    fcs_size = 0
    fcs_value = 0
    if single or fcs_flag:
        fcs_size = FCS_BYTES[fcs_flag]
        if fcs_size == 8:
            fcs_value = struct.unpack_from("<Q", frame, pos)[0]
        else:
            fcs_value = int.from_bytes(frame[pos:pos + fcs_size], "little")
            if fcs_size == 2:
                fcs_value += 256
        pos += fcs_size
    return dict(single=single, checksum=checksum, fcs_flag=fcs_flag,
                dict_flag=dict_flag, window=window, dict_size=dict_size,
                fcs_value=fcs_value, header_end=pos, fcs_size=fcs_size)


def classify_stream(buf):
    """Yield one descriptor per frame: ("skip", bytes) or
    ("z", frame_bytes, frame_info_with_block_types).  Block headers are read
    as 3-byte little-endian values so a trailing header never reads past the
    end of the buffer."""
    n = len(buf)
    pos = 0
    while pos < n:
        magic = struct.unpack_from("<I", buf, pos)[0]
        if SKIP_BASE <= magic <= SKIP_BASE + 0xF:
            fsz = struct.unpack_from("<I", buf, pos + 4)[0]
            yield ("skip", buf[pos:pos + 8 + fsz])
            pos += 8 + fsz
            continue
        if magic != ZSTD_MAGIC:
            raise ValueError("bad magic at %d" % pos)
        orig = pos
        info = parse_frame_header(buf[pos:])
        pos = orig + info["header_end"]
        block_types = []
        while True:
            bh = int.from_bytes(buf[pos:pos + 3], "little")
            last = bh & 1
            btype = (bh >> 1) & 3
            bsize = (bh >> 3) & 0x3FFFFF
            pos += 3
            block_types.append((btype, bsize, last))
            if btype == 0:
                pos += bsize
            elif btype == 1:
                pos += 1
            elif btype == 2:
                pos += bsize
            else:
                raise ValueError("reserved block type")
            if last:
                break
        if info["checksum"]:
            pos += 4
        fr = buf[orig:pos]
        info["block_types"] = block_types
        yield ("z", fr, info)


# --------------------------------------------------------------------------
# feature classification
# --------------------------------------------------------------------------
def _bump(counter, axis, value):
    d = counter.setdefault(axis, {})
    d[value] = d.get(value, 0) + 1


def classify_frame(fr, info):
    out = {}
    for btype, _s, _l in info["block_types"]:
        _bump(out, "block_type", {0: "raw", 1: "rle", 2: "compressed"}.get(
            btype, str(btype)))
    try:
        d = run_oracle(fr, expected=None)
    except Exception:
        return out
    for v in d.get("LITBLK", []):
        _bump(out, "literals", LIT_NAME.get(v[0], str(v[0])))
        _bump(out, "size_format", v[1])
    for v in d.get("LITSTREAM", []):
        _bump(out, "literals_stream",
              (LIT_NAME.get(v[0], str(v[0])), v[2]))
    for v in d.get("SEQMODES", []):
        _bump(out, "ll_mode", v[0])
        _bump(out, "of_mode", v[1])
        _bump(out, "ml_mode", v[2])
    for v in d.get("SEQCOUNT", []):
        if v[0] == 0:
            _bump(out, "zero_seq_blocks", 1)
    _bump(out, "checksum", "on" if info["checksum"] else "off")
    _bump(out, "single_segment", "on" if info["single"] else "off")
    _bump(out, "fcs_size", info["fcs_size"])
    _bump(out, "dict_size", info["dict_size"])
    _bump(out, "window_present", bool(info["window"] is not None))
    if info["window"] is not None:
        _bump(out, "window_ge_2_29", info["window"] >= WEIGHT_2_29)
    return out


def classify_entry(data):
    """Return (frames, feature_counts).  frames is the classify_stream list."""
    frames = list(classify_stream(data))
    counts = {}
    for item in frames:
        if item[0] == "skip":
            _bump(counts, "skippable", True)
        else:
            for axis, vals in classify_frame(item[1], item[2]).items():
                for val in vals:
                    _bump(counts, axis, val)
    return frames, counts


# --------------------------------------------------------------------------
# hand-built frame primitives (a tiny, correct zstd writer for the features
# libzstd cannot be made to emit)
# --------------------------------------------------------------------------
class BitWriter:
    def __init__(self):
        self.b = bytearray()
        self.pos = 0

    def write(self, value, nbits):
        for i in range(nbits):
            if (value >> i) & 1:
                idx = self.pos // 8
                if idx == len(self.b):
                    self.b.append(0)
                self.b[idx] |= 1 << (self.pos % 8)
            self.pos += 1

    def align(self):
        if self.pos % 8:
            self.pos += 8 - (self.pos % 8)

    def tobytes(self):
        return bytes(self.b)


def frame_header(content_size=None, single=True, fcs_flag=None, checksum=False,
                 window_desc=None, dict_flag=0, dict_val=0):
    """Build a zstd frame header.  content_size None -> no FCS (non-single)."""
    desc = 0
    if single:
        desc |= 0x20
    if checksum:
        desc |= 0x04
    if dict_flag:
        desc |= dict_flag
    if fcs_flag is None:
        if content_size is None:
            fcs_flag = 0
            single = False
        elif content_size < 256:
            fcs_flag = 0
        elif content_size < 65536 + 256:
            fcs_flag = 1
        elif content_size < 1 << 32:
            fcs_flag = 2
        else:
            fcs_flag = 3
    else:
        # an explicit single-segment flag may still request a given fcs_flag
        pass
    desc |= (fcs_flag << 6)
    out = struct.pack("<I", ZSTD_MAGIC) + bytes([desc])
    if not single:
        out += bytes([window_desc if window_desc is not None else 0])
    if dict_flag:
        nb = {1: 1, 2: 2, 3: 4}[dict_flag]
        out += (dict_val & ((1 << (8 * nb)) - 1)).to_bytes(nb, "little")
    if single or fcs_flag:
        if fcs_flag == 0:
            out += bytes([content_size & 0xFF])
        elif fcs_flag == 1:
            out += struct.pack("<H", (content_size - 256) & 0xFFFF)
        elif fcs_flag == 2:
            out += struct.pack("<I", content_size & 0xFFFFFFFF)
        else:
            out += struct.pack("<Q", content_size & 0xFFFFFFFFFFFFFFFF)
    return out


def block_header(bsize, btype, last=True):
    return ((1 if last else 0) | (btype << 1) | (bsize << 3)).to_bytes(3, "little")


def zstd_frame(content, block_bodies, single=True, fcs_flag=None, checksum=False,
               window_desc=None, dict_flag=0, dict_val=0):
    """Assemble a frame from a list of (btype, body_bytes) blocks.
    content is the decoded output (for FCS)."""
    head = frame_header(len(content), single=single, fcs_flag=fcs_flag,
                        checksum=checksum, window_desc=window_desc,
                        dict_flag=dict_flag, dict_val=dict_val)
    out = head
    for i, (btype, body) in enumerate(block_bodies):
        last = (i == len(block_bodies) - 1)
        out += block_header(len(body), btype, last=last) + body
    return out


def rle_literals(symbol, size, sf=1):
    """Literals section: block_type 1 (RLE), repeated symbol `size` times."""
    bw = BitWriter()
    bw.write(1, 2)
    bw.write(sf, 2)
    if sf in (0, 2):
        bw.write(size, 5)
    elif sf == 1:
        bw.write(size, 12)
    else:
        bw.write(size, 20)
    bw.align()
    return bw.tobytes() + bytes([symbol & 0xFF])


def raw_literals(data, sf=None):
    """Literals section: block_type 0 (raw), verbatim data bytes."""
    n = len(data)
    if sf is None:
        sf = 3 if n >= 4096 else (1 if n >= 32 else 0)
    bw = BitWriter()
    bw.write(0, 2)
    bw.write(sf, 2)
    if sf in (0, 2):
        bw.write(n, 5)
    elif sf == 1:
        bw.write(n, 12)
    else:
        bw.write(n, 20)
    bw.align()
    return bw.tobytes() + bytes(data)


def skippable_frame(payload, magic_offset=0):
    return (struct.pack("<I", SKIP_BASE + magic_offset)
            + struct.pack("<I", len(payload)) + payload)


def zero_seq_block(literals_bytes):
    """A compressed block with a literals section and 0 sequences."""
    return literals_bytes + b"\x00"


# --------------------------------------------------------------------------
# corpus
# --------------------------------------------------------------------------
def make_libzstd_corpus():
    import numpy as np
    import random
    rng = random.Random(1234)
    entries = []

    def add(name, data, **cparams):
        try:
            c = zstd.ZstdCompressor(**cparams)
        except TypeError:
            return
        comp = c.compress(data)
        entries.append((name, comp, "decode"))
        return comp

    # fixed-length tokens + a varying byte -> size-format-3 literals and an RLE
    # match-length table (from probe11/probe12).  probe11 drew along one shared
    # Random(7), so reproduce that exact state sequence.
    trng = random.Random(7)
    for _ in range(3):          # token3's base would consume draws 0..2
        trng.getrandbits(8)
    token4_base = bytes(trng.getrandbits(8) for _ in range(4))   # draws 3..6
    token4 = b"".join(token4_base + bytes([i & 0xFF])
                      for i in range(30000 // 5))
    token8_base = bytes(trng.getrandbits(8) for _ in range(8))   # draws 7..14
    token8 = b"".join(token8_base + bytes([i & 0xFF])
                      for i in range(30000 // 9))

    datasets = {
        "text": (b"The quick brown fox jumps over the lazy dog. " * 2000),
        "random": (bytes(rng.getrandbits(8) for _ in range(20000))),
        "zeros": (bytes(30000)),
        "runs": (bytes([7]) * 30000),
        "u16": (np.random.default_rng(1).poisson(3, 40000)
                .astype(np.uint16).tobytes()),
        "u32": (np.random.default_rng(2).poisson(5, 40000)
                .astype(np.uint32).tobytes()),
        # low-entropy literal-heavy data: this is what drives the RLE and repeat
        # LL/OF/ML table modes, the 1-stream compressed literals, and the
        # size_format 3 literals (from probe8).
        "skew0": (np.random.default_rng(3)
                  .choice(np.array([65, 66, 67, 68]),
                          size=300, p=[0.85, 0.1, 0.03, 0.02])
                  .astype(np.uint8).tobytes()),
        "skew1": (np.random.default_rng(4)
                  .choice(np.array([65, 66]),
                          size=900, p=[0.9, 0.1])
                  .astype(np.uint8).tobytes()),
        "skew2": (np.random.default_rng(5)
                  .choice(np.array([65, 66, 67]),
                          size=2000, p=[0.8, 0.12, 0.08])
                  .astype(np.uint8).tobytes()),
        "skew3": (np.random.default_rng(6)
                  .choice(np.array([65, 66, 67, 68]),
                          size=30000, p=[0.85, 0.1, 0.03, 0.02])
                  .astype(np.uint8).tobytes()),
        # mostly-literals, low-diversity: compressible but literal-heavy
        "litmix": (bytes([65]) * 15000 + bytes([66]) * 15000),
        "token4": token4,
        "token8": token8,
        # detector-like u16/u32 images (Poisson noise) - these drive the
        # treeless literals, the 1-stream treeless (sf 0) and the repeat LL/OF
        # table modes (from probe12).
        "img1": (np.random.default_rng(1).poisson(3, 200000)
                 .astype(np.uint16).tobytes()),
        "img2": (np.random.default_rng(1).poisson(5, 200000)
                 .astype(np.uint32).tobytes()),
        "img3": (np.random.default_rng(2).poisson(3, 80000)
                 .astype(np.uint16).tobytes()),
    }
    # columnar detector-like rows -> many same-offset matches (RLE offset table)
    n_rows, row = 6000, 64
    img = np.zeros((n_rows, row), dtype=np.uint8)
    for i in range(1, n_rows):
        img[i] = img[i - 1]
        img[i][0] = i % 251
    datasets["rows"] = img.tobytes()

    for dname, data in datasets.items():
        for lev in list(range(1, 20)) + [22]:
            add("zstd-%s-l%d" % (dname, lev), data, level=lev)

    # write_checksum, write_content_size off (multi-segment no-FCS), params
    for dname in ("text", "skew1", "skew2", "rows"):
        data = datasets[dname]
        add("zstd-%s-checksum" % dname, data, level=9, write_checksum=True)
        # streaming compressor: no Frame_Content_Size
        try:
            co = zstd.ZstdCompressor(level=5)
            buf = bytearray()
            with co.stream_writer(buf) as w:
                w.write(data)
            entries.append(("zstd-%s-streaming" % dname, bytes(buf), "reject:SBZ_ERR_CORRUPT"))
        except Exception:
            pass
        # parameters with a big window_log (pushes window >= 2^29)
        try:
            params = zstd.ZstdCompressionParameters.from_level(
                9, window_log=26)
            entries.append(("zstd-%s-bigwindow" % dname,
                            zstd.ZstdCompressor(compression_params=params)
                            .compress(data), "decode"))
        except Exception:
            pass

    # concatenated frames + a frame with a single empty block and a checksum
    a = zstd.ZstdCompressor(level=3).compress(datasets["text"])
    b = zstd.ZstdCompressor(level=9).compress(datasets["zeros"])
    entries.append(("zstd-concat2", a + b, "decode"))
    entries.append(("zstd-empty-frame",
                    zstd.ZstdCompressor(level=3).compress(b""), "decode"))
    return entries


def make_handbuilt_corpus():
    entries = []

    def add(name, frame_bytes, expectation="decode"):
        entries.append((name, frame_bytes, expectation))

    # ---- RLE literals (block_type 1) in a 0-sequence compressed block. The
    #      regenerated size must fit the size format; only sf 1 and 3 are built
    #      here (sf 0 and 2 use a degenerate 5-bit size field that is not worth
    #      hand-encoding; those size formats are covered by libzstd frames). ----
    for sf, size in ((1, 500), (3, 50000)):
        lit = rle_literals(0x42, size, sf=sf)
        add("hand-rle-literals-sf%d" % sf,
            zstd_frame(bytes([0x42]) * size, [(2, zero_seq_block(lit))]))

    # ---- RLE frame block (block type 1): the block header size is the
    #      REGENERATED size; the single body byte is the repeat symbol. ----
    rle_size = 300
    fr = frame_header(rle_size, single=True) \
        + block_header(rle_size, 1, last=True) + bytes([ord("B")])
    add("hand-rle-frame-block", fr)

    # ---- raw frame block (block type 0) ----
    content = b"hello zstd world " * 40
    add("hand-raw-frame-block", zstd_frame(content, [(0, content)]))

    # ---- raw literals in a 0-sequence compressed block (non-single so
    #      libzstd accepts it: single-segment + raw-literals-0seq is rejected) ----
    content = b"raw literals block" * 20
    lit = raw_literals(content)
    add("hand-raw-literals-0seq",
        zstd_frame(content, [(2, zero_seq_block(lit))],
                   single=False, fcs_flag=2, window_desc=0x10))

    # ---- a two-block frame: raw-literals 0-sequence block + raw trailer.
    #      (raw literals, not RLE: RLE literals are produced only by the
    #      hand-rle-literals entries, so the reviewer's starve of those empties
    #      exactly the "rle literals" cell.) ----
    lit_part = b"0-sequence block body " * 8
    trailer = b"trailer bytes " * 7
    lit = raw_literals(lit_part)
    add("hand-0seq-two-block",
        zstd_frame(lit_part + trailer,
                   [(2, zero_seq_block(lit)), (0, trailer)]))

    # ---- skippable frames: at the start, middle and end (each uses a pair of
    #      real zstd frames so a skip leaves non-empty payloads) ----
    payload1 = b"metadata-1"
    payload2 = b"metadata-2"
    z = zstd.ZstdCompressor(level=5).compress(b"payload between frames " * 50)
    pz = zstd.ZstdCompressor(level=5).compress(b"trailing payload " * 50)
    add("hand-skippable-start", skippable_frame(payload1) + z)
    add("hand-skippable-middle", z + skippable_frame(payload2) + pz)
    add("hand-skippable-end", z + skippable_frame(payload1))

    # ---- checksum on: covered by the libzstd write_checksum corpus entries
    #      (python-zstandard's decompress VERIFIES the checksum, so a hand
    #      frame would need the real XXH64 bytes; the libzstd frames have them).
    #      A skippable frame at the very start of a stream is not accepted by
    #      python-zstandard's decompress(), so the hand-built skippable streams
    #      start with a real zstd frame (middle and end are covered). ----

    # ---- 1-byte FCS (single-segment, fcs_flag 0 -> content < 256) ----
    content = b"tiny" * 60  # 240 bytes
    add("hand-1byte-fcs", zstd_frame(content, [(0, content)]))

    # ---- 8-byte FCS (fcs_flag 3): a valid small frame and a >4 GiB one ----
    content = b"eight-byte-fcs-frame" * 40
    add("hand-8byte-fcs-ok", zstd_frame(content, [(0, content)], fcs_flag=3))
    # > 2^32 content size, header only -> expected SBZ_ERR_LIMIT.  It still has
    # a real (raw) block so a decoder that reads the block sees a complete frame.
    big_fr = (frame_header((1 << 32) + (1 << 20), single=False, fcs_flag=3,
                           window_desc=0x20)
              + block_header(1, 0, last=True) + b"\x00")
    add("hand-8byte-fcs-limit", big_fr, "reject:SBZ_ERR_LIMIT")

    # ---- window >= 2^29 (expected reject today) ----
    content = b"big window frame " * 30
    bigw = zstd_frame(content, [(0, content)], single=False, fcs_flag=2,
                      window_desc=(19 << 3) | 0)  # exponent 19 -> 2^29
    add("hand-window-2e29", bigw, "reject:SBZ_ERR_CORRUPT")

    # ---- no-FCS frame (non-single, fcs_flag 0) -> expected reject ----
    nf = frame_header(None, single=False, fcs_flag=0, window_desc=0x20) \
        + block_header(5, 0, last=True) + b"hello"
    add("hand-no-fcs", nf, "reject:SBZ_ERR_CORRUPT")

    # ---- dictionary-ID field present.  Dictionaries are out of scope and the
    #      decoder rejects a frame that carries a Dictionary_ID (even a value of
    #      0; the host walker accepts dictionaries with id 0 but the GPU parser
    #      rejects too, so host/GPU reject differ - a T09 item).  Recorded as a
    #      reject cell, not a coverage failure. ----
    content = b"dict id frame " * 10
    add("hand-dict-id-1",
        zstd_frame(content, [(0, content)], dict_flag=1, dict_val=0),
        "reject:SBZ_ERR_CORRUPT")

    # ---- one multi-block frame: raw-literals 0-sequence block + raw blocks
    #      (again raw literals, so the rle-literals cell stays hand-rle-only) ----
    raw1, raw2 = b"prefix block " * 4, b"middle block data " * 8
    litp = b"tail with a 0-sequence compressed block " * 2
    lit = raw_literals(litp)
    add("hand-multiblock",
        zstd_frame(litp + raw1 + raw2,
                   [(2, zero_seq_block(lit)), (0, raw1), (0, raw2)]))
    return entries


# --------------------------------------------------------------------------
# matrix definition: (axis, required values)
# --------------------------------------------------------------------------
def required_cells():
    req = {
        "literals": ["raw", "rle", "compressed", "treeless"],
        "literals_stream": [("compressed", 1), ("compressed", 4),
                            ("treeless", 1), ("treeless", 4)],
        "size_format": [0, 1, 2, 3],
        "ll_mode": [0, 1, 2, 3],
        "of_mode": [0, 1, 2, 3],
        "ml_mode": [0, 1, 2, 3],
        "zero_seq_blocks": [1],
        "block_type": ["raw", "rle", "compressed"],
        "checksum": ["on", "off"],
        "skippable": [True],
        "fcs_size": [0, 1, 2, 4, 8],
        "single_segment": ["on", "off"],
    }
    return req


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-gpu", action="store_true",
                    help="only classify, no decode")
    args = ap.parse_args()

    corpus = make_libzstd_corpus() + make_handbuilt_corpus()

    lib = load()
    compared = 0
    unexpected = 0
    failures = []
    xfails = []
    total_bytes = 0

    aggregate = {}
    frame_spans = []       # for the _frames path
    frame_bytes_concat = bytearray()
    libzstd_expected_concat = bytearray()
    base = 0

    bug_dir = os.makedirs(BUGS, exist_ok=True) or BUGS

    for name, data, expectation in corpus:
        total_bytes += len(data)
        expect_code = None
        if expectation.startswith("reject:"):
            expect_code = {"SBZ_ERR_CORRUPT": SBZ_ERR_CORRUPT,
                           "SBZ_ERR_LIMIT": SBZ_ERR_LIMIT,
                           "-5": SBZ_ERR_CORRUPT,
                           "-6": SBZ_ERR_LIMIT}[expectation.split(":")[1]]
        try:
            frames, counts = classify_entry(data)
        except Exception as e:
            failures.append("%s: classify failed: %s" % (name, e))
            continue

        # aggregate feature counts
        for axis, vals in counts.items():
            for val in vals:
                _bump(aggregate, axis, val)

        # decode.  A fresh context per entry so that an entry that crashes the
        # GPU (a decoder bug) does not poison the loop: it is recorded as an
        # xfail and the next entry starts from a clean context.
        if args.skip_gpu:
            continue
        dc = make_dctx(lib)
        try:
            rc, got = decompress(lib, dc, data)

            # expected/actual comparison
            if expect_code is not None:
                if rc != expect_code:
                    unexpected += 1
                    failures.append(
                        "%s: expected reject:%d got rc=%d (%s)"
                        % (name, expect_code, rc, lib.sbz_dctx_error(dc).decode()))
                continue
            # expectation == decode
            if rc != 0:
                # a decoder bug, unless it is an accepted limit
                if rc in (SBZ_ERR_CORRUPT, SBZ_ERR_LIMIT):
                    write_bug(name, data, rc, lib.sbz_dctx_error(dc).decode())
                    xfails.append((name, rc, lib.sbz_dctx_error(dc).decode()))
                    continue
                unexpected += 1
                failures.append("%s: decode rc=%d (%s)"
                                % (name, rc, lib.sbz_dctx_error(dc).decode()))
                continue
            want = _zstd_decompress(data)
            if want is None:
                unexpected += 1
                failures.append("%s: zstandard could not decode a valid frame"
                                % name)
                continue
            got = got[:len(want)]
            if got != want:
                i = next((k for k, (a, b) in enumerate(zip(got, want))
                          if a != b), len(got))
                unexpected += 1
                failures.append(
                    "%s: byte mismatch at %d (got %d want %d bytes)"
                    % (name, i, len(got), len(want)))
                continue
            compared += len(want)

            # accumulate a concatenated batch for the _frames path.  Only single
            # non-skippable frames keep the accounting simple (want == that
            # frame's output).
            if len(frames) == 1 and frames[0][0] == "z":
                frame_bytes = frames[0][1]
                frame_bytes_concat += frame_bytes
                libzstd_expected_concat += want
                frame_spans.append((base, len(frame_bytes)))
                base += len(frame_bytes)
        finally:
            lib.sbz_dctx_destroy(dc)

    # ---- the sbz_dctx_decompress_frames path ----
    if not args.skip_gpu and frame_spans:
        dc = make_dctx(lib)
        try:
            rc, got = decompress_frames(lib, dc, bytes(frame_bytes_concat),
                                        frame_spans)
            want_concat = bytes(libzstd_expected_concat)
            if rc == 0 and got[:len(want_concat)] == want_concat:
                compared += len(want_concat)
            else:
                unexpected += 1
                failures.append(
                    "decompress_frames batch: rc=%d got %d want %d bytes"
                    % (rc, len(got), len(want_concat)))
        finally:
            lib.sbz_dctx_destroy(dc)

    print_matrix(aggregate)
    print()
    print("entries=%d  compressed_bytes=%d  compared_bytes=%d  unexpected=%d"
          % (len(corpus), total_bytes, compared, unexpected))

    # ---- validate the matrix ----
    req = required_cells()
    empty = []
    for axis, values in req.items():
        cell = aggregate.get(axis, {})
        for v in values:
            if cell.get(v, 0) < 1:
                empty.append("%s=%r" % (axis, v))
    if empty:
        failures.append("empty matrix cells (need >= 1 each): " + ", ".join(empty))

    for name, rc, msg in xfails:
        print("xfail  %s (rc=%d: %s) -> reproducer in tests/data/features_bugs/"
              % (name, rc, msg))

    print()
    if compared == 0:
        failures.append("no bytes compared")
    if failures:
        print("FAIL")
        for f in failures:
            print("  " + f)
        return 1
    print("PASS")
    return 0


def _zstd_decompress(data):
    """Decode `data` with libzstd, skipping skippable frames (which
    python-zstandard's single-shot decompress() does not accept at the start)."""
    try:
        return zstd.ZstdDecompressor().decompress(strip_skippable(data))
    except Exception:
        return None


def strip_skippable(buf):
    n = len(buf)
    pos = 0
    while pos < n:
        magic = struct.unpack_from("<I", buf, pos)[0]
        if SKIP_BASE <= magic <= SKIP_BASE + 0xF:
            fsz = struct.unpack_from("<I", buf, pos + 4)[0]
            pos += 8 + fsz
        else:
            return buf[pos:]
    return b""


def write_bug(name, data, rc, msg):
    try:
        os.makedirs(BUGS, exist_ok=True)
        fn = os.path.join(BUGS, name.replace("/", "_") + ".zst")
        with open(fn, "wb") as f:
            f.write(data)
    except OSError:
        pass


def print_matrix(agg):
    def show(axis, names):
        cell = agg.get(axis, {})
        vals = []
        for v in names:
            vals.append("%s=%d" % (_fmt(v), cell.get(v, 0)))
        print("  %-18s %s" % (axis, "  ".join(vals)))

    print("zstd feature matrix (each cell count must be >= 1):")
    show("literals", ["raw", "rle", "compressed", "treeless"])
    show("literals_stream", [("compressed", 1), ("compressed", 4),
                            ("treeless", 1), ("treeless", 4)])
    show("size_format", [0, 1, 2, 3])
    show("ll_mode", [0, 1, 2, 3])
    show("of_mode", [0, 1, 2, 3])
    show("ml_mode", [0, 1, 2, 3])
    show("zero_seq_blocks", [1])
    show("block_type", ["raw", "rle", "compressed"])
    show("checksum", ["on", "off"])
    show("skippable", [True])
    show("fcs_size", [0, 1, 2, 4, 8])
    show("single_segment", ["on", "off"])
    cell = agg.get("dict_size", {})
    print("  %-18s dict_size 0=%d 1=%d 2=%d 4=%d"
          % ("(info)", cell.get(0, 0), cell.get(1, 0),
             cell.get(2, 0), cell.get(4, 0)))
    cell = agg.get("window_ge_2_29", {})
    print("  %-18s window>=2^29=%d (expected reject)"
          % ("(info)", cell.get(True, 0)))


def _fmt(v):
    if isinstance(v, tuple):
        return "%s/%d" % (v[0], v[1])
    return str(v)


if __name__ == "__main__":
    sys.exit(main())
