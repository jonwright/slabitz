#!/usr/bin/env python3
"""T11: corrupt compressed blocks on the GPU, and compute-sanitizer.

Mutates bytes INSIDE the compressed blocks of valid zstd frames and
bitshuffle chunks and decodes each mutant on the GPU in its own subprocess
under a timeout, separately counting every outcome:

  ok-identical   rc == 0 and the middle frame/chunk decoded to the original
                 bytes (the mutation did not matter) and both neighbours are
                 intact
  error-code     rc < 0 with a normal sbz error (recorded)
  wrong-output   rc == 0, neighbours intact, but the middle differs from its
                 original decode (zstd checksums are not verified on the GPU)
  canary-hit     rc == 0 but a neighbour frame/chunk is damaged -- an
                 out-of-bounds write, the thing this job hunts (REQUIRED ZERO)
  err-neighbour  not used any more (kept in the table as 0). An error leaves
                 the whole call's output undefined: an early error writes
                 nothing (bitshuffle output is not written at all), and a
                 corrupt frame can spoil other frames of the call through
                 shared decode state (literal and sequence prefix sums) without
                 writing outside its range. Writes outside a frame's range show
                 as canary-hit (rc 0); every error counts as error-code.
  gpu-fault      the call returned an error whose message contains a CUresult
                 (e.g. "CUresult 700"); the process can no longer use the GPU
                 (REQUIRED ZERO)
  crash          subprocess killed by a signal or nonzero exit without a result
                 line (REQUIRED ZERO)
  hang           the subprocess timed out (REQUIRED ZERO)

Because the decoder owns its output buffer (a caller cannot pad it), the OOB
canary is a TRIPLE: each mutant is decoded in ONE call between two unmodified
good frames/chunks (sbz_dctx_decompress_frames, sbz_dctx_bshuf_decompress or
sbz_dctx_bslz4_decompress), and the two neighbours are compared byte-exact
against the good input's own decode.  The decoder lays the three frames out by
their content sizes (UnCompressedFramesRefs[f].offs = sum of the preceding
frames' content-size fields), so the middle frame's region is [csA, csA+csM)
and the second neighbour starts at csA+csM.  A mutation that overruns its own
region damages the second neighbour; both neighbours matching the reference
means the canary is intact.  Writes outside the whole allocation are
compute-sanitizer's job (tests/run_sanitizer.sh).

Mutation kinds (all four): single bit flip, single byte set to 0x00, single
byte set to 0xFF, an 8-byte run of random bytes, and truncating a compressed
block's last 4 bytes (its header size is lowered by 4; a Frame_Content_Size is
left as it is, so the frame then lies about its size).  Only bytes inside
compressed blocks (block type 2) are touched, from a Python walk of the frame
header and block headers; for bitshuffle-LZ4, only the LZ4 block payload bytes
are touched.

Named seeds (decoded first): the two T06 seeds (a malformed RLE-literals
section whose regenerated size disagrees with the content size, and a
sequence-section compression-modes byte set to repeat(3) for LL, OF and ML on
the first and a later block), then the T07 patch-from frame (decoded alone).

Usage (from the worktree):
  timeout 5400 gpu-run --time 01:30:00 python3 tests/run_gpu_corrupt.py
  python3 tests/run_gpu_corrupt.py --replay build/corrupt_cases/gpu-fault-0.bin
"""
import argparse
import ctypes as ct
import json
import os
import struct
import subprocess
import sys
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = os.path.join(ROOT, "build", "libslabitz.so")
KDIR = os.environ.get("SLABITZ_KERNEL_DIR", os.path.join(ROOT, "build"))
OUT = os.path.join(ROOT, "build", "corrupt_cases")
INDEX = os.path.join(OUT, "INDEX.txt")
WORK = os.path.join(os.environ.get("TMPDIR", os.path.join(ROOT, "scratch_work")),
                    "corrupt_work")
JOBS = os.path.join(WORK, "jobs")

ZSTD_MAGIC = 0xFD2FB528
SKIP_BASE = 0x184D2A50
SBZ_ERR_BACKEND, SBZ_ERR_ARGS, SBZ_ERR_CORRUPT, SBZ_ERR_LIMIT = -1, -3, -5, -6

# 40% stock zstd, 30% pac-zstd chunks, 30% bitshuffle-LZ4 (the brief's mix).
STOCK = [
    ("v1.5.7-l3-det", "tests/data/compat/v1.5.7--level3_det_u16.zst", "zstd"),
    ("v1.5.7-l19-det", "tests/data/compat/v1.5.7--level19_det_u16.zst", "zstd"),
    ("v1.5.7-l3-text", "tests/data/compat/v1.5.7--level3_text.zst", "zstd"),
    ("v1.5.7-l19-text", "tests/data/compat/v1.5.7--level19_text.zst", "zstd"),
]
PAC_ZSTD = [
    ("pac-ceo2-u32-zstd", "data/pac/ceo2_u32_zstd_maskmax.chunk.bin", "bshuf"),
    ("pac-silicon-u16-zstd", "data/pac/silicon_u16_zstd_mask0.chunk.bin", "bshuf"),
]
PAC_LZ4 = [
    ("pac-silicon-u16-lz4", "data/pac/silicon_u16_lz4.chunk.bin", "bslz4"),
]
BASE_TYPES = [("stock", STOCK), ("pac-zstd", PAC_ZSTD), ("pac-lz4", PAC_LZ4)]
MUT_KINDS = ["bit", "zero", "ff", "run8", "trunc"]

DEFAULT_SEED = 20261127
TIME_BIN = 20
PARALLEL = 4

RESULT_ORDER = ["ok-identical", "error-code", "wrong-output", "canary-hit",
                "err-neighbour", "gpu-fault", "crash", "hang"]


class GpuFault(RuntimeError):
    pass


# --------------------------------------------------------------------------
# decoder plumbing
# --------------------------------------------------------------------------
class Span(ct.Structure):
    _fields_ = [("offs", ct.c_uint64), ("size", ct.c_uint64)]


def load():
    lib = ct.CDLL(LIB)
    lib.sbz_dctx_create_cuda.restype = ct.c_void_p
    lib.sbz_dctx_create_cuda.argtypes = [ct.c_int, ct.c_char_p, ct.c_char_p, ct.c_size_t]
    lib.sbz_dctx_destroy.argtypes = [ct.c_void_p]
    lib.sbz_dctx_error.restype = ct.c_char_p
    lib.sbz_dctx_error.argtypes = [ct.c_void_p]
    lib.sbz_dctx_decompress.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_size_t]
    lib.sbz_dctx_decompress_frames.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_size_t,
                                               ct.POINTER(Span), ct.c_uint32]
    lib.sbz_dctx_bshuf_decompress.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_size_t,
                                              ct.POINTER(Span), ct.c_uint32, ct.c_uint32]
    lib.sbz_dctx_bslz4_decompress.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_size_t,
                                              ct.POINTER(Span), ct.c_uint32, ct.c_uint32]
    lib.sbz_dctx_read_buffer.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_void_p,
                                         ct.POINTER(ct.c_size_t)]
    lib.sbz_dctx_parse.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_size_t]
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
        raise GpuFault("sbz_dctx_create_cuda: %s" % err.value.decode())
    return dc


# --------------------------------------------------------------------------
# zstd structure walk (Python port, enough to find the compressed-block byte
# ranges and the Frame_Content_Size)
# --------------------------------------------------------------------------
def frame_spans(buf):
    """Yield (offset, size) of each zstd frame in `buf` (skipping skippable)."""
    n = len(buf)
    pos = 0
    spans = []
    while pos < n:
        magic = struct.unpack_from("<I", buf, pos)[0]
        if SKIP_BASE <= magic <= SKIP_BASE + 0xF:
            fsz = struct.unpack_from("<I", buf, pos + 4)[0]
            pos += 8 + fsz
            continue
        if magic != ZSTD_MAGIC:
            raise ValueError("bad magic at %d" % pos)
        orig = pos
        desc = buf[pos + 4]
        single = bool(desc & 0x20)
        cab = 5
        if not single:
            cab += 1
        dict_flag = desc & 3
        cab += {0: 0, 1: 1, 2: 2, 3: 4}[dict_flag]
        fcs_flag = (desc >> 6) & 3
        if single or fcs_flag:
            cab += {0: 1, 1: 2, 2: 4, 3: 8}[fcs_flag]
        pos = orig + cab
        while True:
            bh = int.from_bytes(buf[pos:pos + 3], "little")
            last = bh & 1
            btype = (bh >> 1) & 3
            bsize = (bh >> 3) & 0x3FFFFF
            pos += 3
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
        if desc & 0x04:
            pos += 4
        spans.append((orig, pos - orig))
    return spans


def frame_blocks(frame):
    """Yield (btype, body_start, body_len) per block of one frame."""
    desc = frame[4]
    single = bool(desc & 0x20)
    pos = 5
    if not single:
        pos += 1
    dict_flag = desc & 3
    pos += {0: 0, 1: 1, 2: 2, 3: 4}[dict_flag]
    fcs_flag = (desc >> 6) & 3
    if single or fcs_flag:
        pos += {0: 1, 1: 2, 2: 4, 3: 8}[fcs_flag]
    out = []
    while True:
        bh = int.from_bytes(frame[pos:pos + 3], "little")
        last = bh & 1
        btype = (bh >> 1) & 3
        bsize = (bh >> 3) & 0x3FFFFF
        body_start = pos + 3
        out.append((btype, body_start, bsize))
        pos = body_start
        if btype == 0:
            pos += bsize
        elif btype == 1:
            pos += 1
        elif btype == 2:
            pos += bsize
        if last:
            break
    return out


def frame_content_size(frame):
    desc = frame[4]
    single = bool(desc & 0x20)
    pos = 5
    if not single:
        pos += 1
    dict_flag = desc & 3
    pos += {0: 0, 1: 1, 2: 2, 3: 4}[dict_flag]
    fcs_flag = (desc >> 6) & 3
    if not (single or fcs_flag):
        return None
    nb = {0: 1, 1: 2, 2: 4, 3: 8}[fcs_flag]
    if nb == 8:
        return struct.unpack_from("<Q", frame, pos)[0]
    v = int.from_bytes(frame[pos:pos + nb], "little")
    if nb == 2:
        v += 256
    return v


# --------------------------------------------------------------------------
# mutation (frame level): bit / zero / ff / run8 / trunc inside a compressed
# block body.  Returns the mutated frame (may be shorter for trunc).
# --------------------------------------------------------------------------
def _set_block_size(fr, header_pos, new_size):
    h = fr[header_pos] | (fr[header_pos + 1] << 8) | (fr[header_pos + 2] << 16)
    last = h & 1
    btype = (h >> 1) & 3
    nbh = last | (btype << 1) | (new_size << 3)
    fr[header_pos] = nbh & 0xFF
    fr[header_pos + 1] = (nbh >> 8) & 0xFF
    fr[header_pos + 2] = (nbh >> 16) & 0xFF


def mutate_frame(frame, body_start, body_len, kind, byte_off, rng):
    fr = bytearray(frame)
    if kind == "trunc":
        new_len = body_len - 4
        _set_block_size(fr, body_start - 3, new_len)
        del fr[body_start + body_len - 4:body_start + body_len]
        return bytes(fr)
    abs_off = body_start + byte_off
    if kind == "bit":
        fr[abs_off] ^= (1 << rng.randrange(8))
    elif kind == "zero":
        fr[abs_off] = 0x00
    elif kind == "ff":
        fr[abs_off] = 0xFF
    elif kind == "run8":
        run = min(8, body_len - byte_off)
        for i in range(run):
            fr[body_start + byte_off + i] = rng.randrange(256)
    else:
        raise ValueError("unknown kind %r" % kind)
    return bytes(fr)


# --------------------------------------------------------------------------
# bitshuffle chunk walk: [BE u64 total][BE u32 block bytes] then per block
# [BE u32 c][c bytes].  Returns (total, block_bytes, records).
# --------------------------------------------------------------------------
def chunk_records(chunk, elem):
    total = int.from_bytes(chunk[0:8], "big")
    block_bytes = int.from_bytes(chunk[8:12], "big")
    S = total // elem
    B = block_bytes // elem
    nfull = S // B
    last = S - nfull * B
    last2 = last - last % 8
    nf = nfull + (1 if last2 else 0)
    pos = 12
    recs = []
    for _ in range(nf):
        cs = int.from_bytes(chunk[pos:pos + 4], "big")
        payload_start = pos + 4
        recs.append(dict(start=payload_start, size=cs,
                         payload=chunk[payload_start:payload_start + cs]))
        pos = payload_start + cs
    return total, block_bytes, recs


def build_chunk(total, block_bytes, payloads):
    out = bytearray()
    out += struct.pack(">Q", total)
    out += struct.pack(">I", block_bytes)
    for p in payloads:
        out += struct.pack(">I", len(p))
        out += p
    return bytes(out)


# --------------------------------------------------------------------------
# locate the sequence-section compression_modes byte (for the repeat-mode seed)
# --------------------------------------------------------------------------
class _BR:
    def __init__(self, data):
        self.d = data
        self.bit = 0

    def read(self, n):
        v = 0
        for i in range(n):
            b = (self.d[self.bit // 8] >> (self.bit % 8)) & 1
            v |= b << i
            self.bit += 1
        return v

    def rewind(self, n):
        self.bit -= n

    def align(self):
        if self.bit % 8:
            self.bit = (self.bit // 8 + 1) * 8


def seq_modes_offset(body):
    """Offset within `body` of the sequence-section compression_modes byte, or
    None if the block has no sequences.  Bits: 7-6 LL, 5-4 OF, 3-2 ML."""
    br = _BR(body)
    block_type = br.read(2)
    size_format = br.read(2)
    if block_type <= 1:
        if size_format in (0, 2):
            br.rewind(1)
            size = br.read(5)
        elif size_format == 1:
            size = br.read(12)
        else:
            size = br.read(20)
        br.align()
        payload = size if block_type == 0 else 1
    else:
        if size_format in (0, 1):
            br.read(10)
            comp = br.read(10)
        elif size_format == 2:
            br.read(14)
            comp = br.read(14)
        else:
            br.read(18)
            comp = br.read(18)
        br.align()
        payload = comp
    seq_start = (br.bit // 8) + payload
    if seq_start >= len(body):
        return None
    header = body[seq_start]
    if header == 0:
        return None
    if header < 128:
        mf = 1
    elif header < 255:
        mf = 2
    else:
        mf = 3
    off = seq_start + mf
    return off if off < len(body) else None


# --------------------------------------------------------------------------
# bases (pure Python; the GPU reference is decoded in each worker subprocess)
# --------------------------------------------------------------------------
def load_bases():
    out = []
    for ctype, group in BASE_TYPES:
        for name, path, kind in group:
            src = open(path, "rb").read()
            elem = 1
            if kind != "zstd":
                jp = os.path.join(ROOT, path.replace(".chunk.bin", ".json"))
                if os.path.exists(jp):
                    elem = json.load(open(jp))["elem_size"]
            if kind == "zstd":
                cs = frame_content_size(src)
                if cs is None:
                    raise ValueError("base %s has no FCS" % name)
            else:
                cs = int.from_bytes(src[0:8], "big")
            out.append(dict(name=name, kind=kind, path=path, elem=elem,
                            src=src, cs=cs))
    return out


def decode_ref(lib, base):
    """Decode the base once; return its genuine output, truncated to cs."""
    dc = make_dctx(lib)
    try:
        ref = decode_one(lib, dc, base["kind"], base["src"], base["elem"])
    finally:
        lib.sbz_dctx_destroy(dc)
    return ref[:base["cs"]]


def decode_one(lib, dc, kind, src, elem):
    if kind == "zstd":
        rc = lib.sbz_dctx_decompress(dc, src, len(src))
        if rc != 0:
            raise GpuFault("base decode rc=%d: %s"
                           % (rc, lib.sbz_dctx_error(dc).decode()))
        return read_buf(lib, dc, "UnCompressedFramesData")
    fn = "sbz_dctx_bshuf_decompress" if kind == "bshuf" else "sbz_dctx_bslz4_decompress"
    arr = (Span * 1)(Span(0, len(src)))
    rc = getattr(lib, fn)(dc, src, len(src), arr, 1, elem)
    if rc != 0:
        raise GpuFault("base decode rc=%d: %s" % (rc, lib.sbz_dctx_error(dc).decode()))
    return read_buf(lib, dc, "BshufOutput")


def parse_ok(lib, base):
    if base["kind"] == "zstd":
        src = base["src"]
    else:
        # a chunk's first record is a whole zstd frame (bshuf) or an LZ4 block;
        # parse against sbz_dctx_parse only makes sense for zstd, so just decode
        return True
    dc = make_dctx(lib)
    try:
        rc = lib.sbz_dctx_parse(dc, src, len(src))
        return rc == 0
    finally:
        lib.sbz_dctx_destroy(dc)


# --------------------------------------------------------------------------
# build one mutant job (pure Python)
# --------------------------------------------------------------------------
def build_mutant(base, rng):
    """Return (label_suffix, triple, spans, layout) for a random mutant, or
    raise ValueError if a mutant cannot be built for this base."""
    if base["kind"] == "zstd":
        frame = base["src"]
        blocks = [(b, s, l) for b, s, l in frame_blocks(frame) if b == 2 and l >= 8]
        if not blocks:
            raise ValueError("no compressed blocks with len>=8")
        btype, body_start, body_len = blocks[rng.randrange(len(blocks))]
        mkind = rng.choice(MUT_KINDS)
        if mkind == "trunc":
            byte_off = 0
        elif mkind == "run8":
            byte_off = rng.randrange(max(1, body_len - 7))
        else:
            byte_off = rng.randrange(body_len)
        mut = mutate_frame(frame, body_start, body_len, mkind, byte_off, rng)
        csA = base["cs"]
        csM = frame_content_size(mut) or csA
        triple = base["src"] + mut + base["src"]
        spans = frame_spans(triple)
        layout = dict(csA=csA, csM=csM)
        return mkind, triple, spans, layout
    else:
        chunk = base["src"]
        total, block_bytes, recs = chunk_records(chunk, base["elem"])
        cands = [r for r in recs if r["size"] >= 8]
        if not cands:
            raise ValueError("no records with size>=8")
        rec = cands[rng.randrange(len(cands))]
        mkind = rng.choice(MUT_KINDS)
        if base["kind"] == "bshuf":
            payload = bytes(rec["payload"])
            pframes = [(b, s, l) for b, s, l in frame_blocks(payload) if b == 2 and l >= 8]
            if not pframes:
                raise ValueError("no compressed blocks inside chunk frame")
            bt, bs_, bl = pframes[rng.randrange(len(pframes))]
            if mkind == "trunc":
                bo = 0
            elif mkind == "run8":
                bo = rng.randrange(max(1, bl - 7))
            else:
                bo = rng.randrange(bl)
            mut_payload = mutate_frame(payload, bs_, bl, mkind, bo, rng)
        else:
            p = bytearray(rec["payload"])
            if mkind == "trunc":
                mut_payload = bytes(p[:len(p) - 4])
            elif mkind == "zero":
                p[rng.randrange(len(p))] = 0x00
                mut_payload = bytes(p)
            elif mkind == "ff":
                p[rng.randrange(len(p))] = 0xFF
                mut_payload = bytes(p)
            elif mkind == "bit":
                i = rng.randrange(len(p))
                p[i] ^= (1 << rng.randrange(8))
                mut_payload = bytes(p)
            else:  # run8
                start = rng.randrange(max(1, len(p) - 7))
                for i in range(min(8, len(p) - start)):
                    p[start + i] = rng.randrange(256)
                mut_payload = bytes(p)
        new_payloads = [r["payload"] if r["start"] != rec["start"] else mut_payload
                        for r in recs]
        mut_chunk = build_chunk(total, block_bytes, new_payloads)
        csA = total
        csM = total
        triple = chunk + mut_chunk + chunk
        spans = [dict(offs=0, size=len(chunk)),
                 dict(offs=len(chunk), size=len(mut_chunk)),
                 dict(offs=len(chunk) + len(mut_chunk), size=len(chunk))]
        layout = dict(csA=csA, csM=csM)
        return mkind, triple, spans, layout


# --------------------------------------------------------------------------
# named seeds
# --------------------------------------------------------------------------
def _byte_writer():
    class BW:
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
    return BW


def build_named_seeds(bases):
    """Return a list of job dicts for the named seeds (T06 and T07)."""
    bmap = {b["name"]: b for b in bases}
    BW = _byte_writer()

    def block_header(bsize, btype, last=True):
        return ((1 if last else 0) | (btype << 1) | (bsize << 3)).to_bytes(3, "little")

    def rle_literals(symbol, size, sf=1):
        bw = BW()
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

    def zero_seq_block(lit):
        return lit + b"\x00"

    def frame_header(content_size, single=True, fcs_flag=None):
        if fcs_flag is None:
            if content_size < 256:
                fcs_flag = 0
            elif content_size < 65536 + 256:
                fcs_flag = 1
            elif content_size < 1 << 32:
                fcs_flag = 2
            else:
                fcs_flag = 3
        desc = (0x20 if single else 0) | (fcs_flag << 6)
        out = struct.pack("<I", ZSTD_MAGIC) + bytes([desc])
        nb = {0: 1, 1: 2, 2: 4, 3: 8}[fcs_flag]
        if single and fcs_flag == 1:
            out += struct.pack("<H", (content_size - 256) & 0xFFFF)
        else:
            out += (content_size & ((1 << (8 * nb)) - 1)).to_bytes(nb, "little")
        return out

    jobs = []

    def add(label, base_name, mut, single=False, check_middle=True):
        b = bmap[base_name]
        if single:
            jobs.append(dict(label=label, kind="zstd-single", triple=mut,
                             spans=None, layout=dict(csA=0, csM=0),
                             check_middle=False, base_name=base_name))
            return
        csA = b["cs"]
        csM = frame_content_size(mut) or csA
        triple = b["src"] + mut + b["src"]
        spans = frame_spans(triple)
        jobs.append(dict(label=label, kind="zstd", triple=triple, spans=spans,
                         layout=dict(csA=csA, csM=csM), check_middle=check_middle,
                         base_name=base_name))

    # --- T06 seed 1: a malformed RLE-literals section whose regenerated size
    # (sf0 -> a 5-bit field, so 500 becomes 20) disagrees with the content size.
    size = 500
    lit = rle_literals(0x42, size, sf=0)
    body = zero_seq_block(lit)
    mut = frame_header(size, single=True) + block_header(len(body), 2, last=True) + body
    add("t06-rle-sf0", "v1.5.7-l3-text", mut, check_middle=False)

    # --- T06 seed 2: sequence-section compression-modes byte -> repeat(3), one
    # of LL/OF/ML at a time, on the first block (no previous table) and on a
    # later block.
    def mode_mutants(frame, block_index):
        blocks = [(b, s, l) for b, s, l in frame_blocks(frame) if b == 2]
        if block_index >= len(blocks):
            return []
        btype, body_start, body_len = blocks[block_index]
        body = frame[body_start:body_start + body_len]
        off = seq_modes_offset(body)
        if off is None:
            return []
        cur = body[off]
        ll = (cur >> 6) & 3
        of = (cur >> 4) & 3
        ml = (cur >> 2) & 3
        res = []
        for name, val in (("LL", ll), ("OF", of), ("ML", ml)):
            if val == 3:
                continue
            nv = (cur & 0x03)
            if name == "LL":
                nv |= (3 << 6) | (of << 4) | (ml << 2)
            elif name == "OF":
                nv |= (ll << 6) | (3 << 4) | (ml << 2)
            else:
                nv |= (ll << 6) | (of << 4) | (3 << 2)
            fr = bytearray(frame)
            fr[body_start + off] = nv
            res.append((name, bytes(fr)))
        return res

    for bname, blk in [("v1.5.7-l3-text", 0), ("v1.5.7-l19-text", 1)]:
        frame = bmap[bname]["src"]
        nblocks = len([1 for b in frame_blocks(frame) if b[0] == 2])
        if blk >= nblocks:
            continue
        cell = "first" if blk == 0 else "later"
        for mname, mut in mode_mutants(frame, blk):
            # the middle is a byte-mutation of the base frame, so its expected
            # output is the base's reference output
            add("t06-repeat-%s-%s" % (mname, cell), bname, mut,
                check_middle=True)

    # --- T07 seed: --patch-from frame (expected to fault the GPU), decoded alone.
    pf = os.path.join(ROOT, "tests", "data", "frame_bounds", "patchfrom_frame.zst")
    if os.path.exists(pf):
        src = open(pf, "rb").read()
        jobs.insert(0, dict(label="t07-patchfrom", kind="zstd-single", triple=src,
                            spans=None, layout=dict(csA=0, csM=0),
                            check_middle=False, base_name=None))
    return jobs


# --------------------------------------------------------------------------
# worker (one subprocess): decode one job, print a RESULT line
# --------------------------------------------------------------------------
def run_worker(job, lib=None):
    if lib is None:
        lib = load()
    triple = job["triple"]
    layout = job["layout"]
    kind = job["kind"]
    elem = job.get("elem", 0)
    if kind == "zstd-single":
        dc = make_dctx(lib)
        try:
            rc = lib.sbz_dctx_decompress(dc, triple, len(triple))
            errmsg = lib.sbz_dctx_error(dc).decode() if rc else ""
            return classify_single(rc, errmsg)
        finally:
            lib.sbz_dctx_destroy(dc)
    if kind == "zstd":
        spans = frame_spans(triple)
        arr = (Span * len(spans))(*[Span(o, s) for o, s in spans])
    else:
        spans = [(s["offs"], s["size"]) for s in job["spans"]]
        arr = (Span * len(spans))(*[Span(o, s) for o, s in spans])
    dc = make_dctx(lib)
    try:
        if kind == "zstd":
            rc = lib.sbz_dctx_decompress_frames(dc, triple, len(triple), arr, len(spans))
            outbuf = "UnCompressedFramesData"
        else:
            fn = ("sbz_dctx_bshuf_decompress" if kind == "bshuf"
                  else "sbz_dctx_bslz4_decompress")
            rc = getattr(lib, fn)(dc, triple, len(triple), arr, len(spans), elem)
            outbuf = "BshufOutput"
        errmsg = lib.sbz_dctx_error(dc).decode() if rc else ""
        # output is defined only after success
        out = read_buf(lib, dc, outbuf) if rc == 0 else b""
        # the reference for the good neighbours (precomputed per base)
        ref = job["ref"]
        return classify(layout, ref, job.get("check_middle", False), rc, errmsg, out)
    finally:
        lib.sbz_dctx_destroy(dc)


def classify_single(rc, errmsg):
    if rc != 0:
        if "CUresult" in errmsg or rc == SBZ_ERR_BACKEND:
            return "gpu-fault", rc, errmsg
        return "error-code", rc, errmsg
    return "ok-identical", 0, ""


def classify(layout, ref, check_middle, rc, errmsg, out):
    csA = layout["csA"]
    csM = layout["csM"]
    startB = csA + csM
    A = out[0:csA]
    M = out[csA:startB]
    B = out[startB:startB + csA]
    canary_ok = (A == ref[:csA]) and (B == ref[:csA])
    if rc != 0:
        if "CUresult" in errmsg or rc == SBZ_ERR_BACKEND:
            return "gpu-fault", rc, errmsg
        return "error-code", rc, errmsg
    if not canary_ok:
        return "canary-hit", 0, ""
    if check_middle and M == ref[:min(csM, len(ref))]:
        return "ok-identical", 0, ""
    return "wrong-output", 0, ""


# --------------------------------------------------------------------------
# parent orchestration
# --------------------------------------------------------------------------
def _write_job(job, seq):
    wd = os.path.join(JOBS, "job%05d" % seq)
    os.makedirs(wd, exist_ok=True)
    tp = os.path.join(wd, "triple.bin")
    open(tp, "wb").write(job["triple"])
    doc = dict(triple_path=tp, kind=job["kind"], elem=job.get("elem", 0),
               layout=job["layout"], spans=job.get("spans"),
               check_middle=job.get("check_middle", False),
               base_name=job.get("base_name"))
    # triples that need a neighbour reference always carry it (precomputed per
    # base in the parent); a zstd-single (patch-from) needs no reference.
    if job.get("ref") is not None:
        doc["ref_path"] = os.path.join(wd, "ref.bin")
        open(doc["ref_path"], "wb").write(job["ref"])
    jp = os.path.join(wd, "job.json")
    with open(jp, "w") as f:
        json.dump(doc, f)
    return jp, tp


def _spawn(job, seq):
    jp, tp = _write_job(job, seq)
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", jp]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, cwd=ROOT)
    return proc, jp, tp


def build_job_list(rng, args, bases, named):
    """Build the (named + random) job list deterministically from args.seed."""
    bmap = {b["name"]: b for b in bases}
    jobs = []
    for nd in named:
        job = dict(kind=nd["kind"], triple=nd["triple"], spans=nd.get("spans"),
                   layout=nd["layout"], check_middle=nd.get("check_middle", False),
                   label=nd["label"], elem=0, base_name=nd.get("base_name"))
        if nd["kind"] != "zstd-single" and nd.get("base_name"):
            job["ref"] = bmap[nd["base_name"]]["ref"]
        jobs.append(job)
    group_of = dict(BASE_TYPES)
    n_mut = 0
    while n_mut < args.n:
        # the brief's mix: 40% stock zstd, 30% pac-zstd, 30% bitshuffle-LZ4
        ctype = rng.choices(["stock", "pac-zstd", "pac-lz4"],
                            weights=[40, 30, 30])[0]
        group = group_of[ctype]
        tup = rng.choice(group)
        base = bmap[tup[0]]
        try:
            mkind, triple, spans, layout = build_mutant(base, rng)
        except ValueError:
            continue
        job = dict(kind=base["kind"], triple=triple, spans=spans, layout=layout,
                   check_middle=True, ref=base["ref"], elem=base["elem"],
                   base_name=base["name"], label="%s/%s" % (base["name"], mkind))
        jobs.append(job)
        n_mut += 1
    return jobs


def run_all(args, bases, named):
    rng = random.Random(args.seed)
    jobs = build_job_list(rng, args, bases, named)

    from collections import deque
    import time
    queue = deque(jobs)
    active = []
    seq = 0
    counts = {}
    saved = []
    results = []
    os.makedirs(JOBS, exist_ok=True)
    while queue or active:
        while queue and len(active) < PARALLEL:
            job = queue.popleft()
            proc, jp, tp = _spawn(job, seq)
            seq += 1
            active.append((proc, jp, tp, job, time.monotonic()))
        # reap finished children and kill children that exceeded the timeout
        finished = []
        timed_out = []
        now = time.monotonic()
        for item in list(active):
            proc, jp, tp, job, start = item
            if proc.poll() is not None:
                finished.append(item)
            elif now - start > TIME_BIN:
                timed_out.append(item)
        for item in finished + timed_out:
            active.remove(item)
            proc, jp, tp, job, start = item
            if (proc, jp, tp, job) in [x[:4] for x in timed_out]:
                proc.kill()
                res_kind = "hang"
                res = (res_kind, None, "")
            else:
                res = _collect(proc, jp)
                res_kind = res[0]
            counts[res_kind] = counts.get(res_kind, 0) + 1
            results.append((job["label"], res_kind, res))
            if res_kind in ("hang", "crash", "canary-hit", "gpu-fault"):
                fn = _save_failure(bases, job, res_kind, res, tp)
                saved.append((res_kind, fn, job["label"]))
        if not finished and not timed_out:
            time.sleep(0.05)
    return counts, saved, results


def _collect(proc, jp):
    # proc.poll() was non-None (the process has exited), so communicate()
    # returns immediately; a bare communicate(timeout=0) spuriously raises
    # TimeoutExpired on an already-finished child, which mislabels a result
    # as a hang.
    try:
        out, err = proc.communicate()
    except ValueError:
        # communicate already called once (should not happen) -- fall back
        out = err = ""
    if proc.returncode != 0 or not out.strip():
        return ("crash", proc.returncode, (err or "")[:300])
    line = out.strip().splitlines()[-1]
    if line.startswith("RESULT "):
        parts = line.split(maxsplit=2)
        kind = parts[1]
        rc = parts[2] if len(parts) > 2 else ""
        return (kind, rc, line)
    return ("crash", proc.returncode, line[:300])


def _save_failure(bases, job, res_kind, res, tp):
    os.makedirs(OUT, exist_ok=True)
    n = 0
    while os.path.exists(os.path.join(OUT, "%s-%d.bin" % (res_kind, n))):
        n += 1
    fn = os.path.join(OUT, "%s-%d.bin" % (res_kind, n))
    open(fn, "wb").write(job["triple"])
    meta = dict(label=job["label"], kind=job["kind"], elem=job.get("elem", 0),
                layout=job["layout"], spans=job.get("spans"),
                check_middle=job.get("check_middle", False),
                base_name=job.get("base_name"),
                result=res_kind, rc=str(res[1]), err=(res[2][:300] if len(res) > 2 else ""))
    # the base is committed; --replay re-derives the reference from it
    base = next((b for b in bases if b["name"] == job["base_name"]), None)
    if base is not None:
        meta["base_path"] = base["path"]
        meta["base_kind"] = base["kind"]
        meta["base_cs"] = base["cs"]
        meta["base_elem"] = base["elem"]
    with open(fn + ".json", "w") as f:
        json.dump(meta, f)
    with open(INDEX, "a") as f:
        f.write("%s\t%s\t%s\n" % (res_kind, os.path.basename(fn), job["label"]))
    return fn


def print_summary(counts, args):
    print("GPU corrupt-block run (seed %d)" % args.seed)
    total = 0
    for k in RESULT_ORDER:
        v = counts.get(k, 0)
        total += v
        print("  %-16s %d" % (k, v))
    for k in sorted(counts):
        if k not in RESULT_ORDER:
            print("  %-16s %d" % (k, counts[k]))
            total += counts[k]
    print("  %-16s %d" % ("TOTAL", total))
    return total


def sanitize_run(args, bases, named):
    """Single-process decode loop for use under compute-sanitizer: the sanitizer
    instruments THIS process, so the mutants are decoded here (no subprocess
    isolation) and the run stops at the first GPU fault to give the sanitizer a
    single faulting input.  Returns an exit code (1 if a fault tripped)."""
    rng = random.Random(args.seed)
    jobs = build_job_list(rng, args, bases, named)
    lib = load()
    counts = {}
    n = 0
    for job in jobs:
        kind, rc, errmsg = run_worker(job, lib)
        counts[kind] = counts.get(kind, 0) + 1
        n += 1
        print("  %-40s %s %s %s" % (job["label"], kind, rc,
                                    errmsg[:80].replace("\n", " ")))
        if kind in ("gpu-fault", "crash"):
            # a GPU fault poisons the process; the sanitizer wants the first one
            print("stopping at first %s (%s)" % (kind, job["label"]))
            print_summary(counts, args)
            return 1
    print_summary(counts, args)
    return 0


def main():
    global PARALLEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", dest="n", type=int, default=2000)
    ap.add_argument("--seed", dest="seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--long", action="store_true", help="soak run (20000 inputs)")
    ap.add_argument("--replay", metavar="FILE")
    ap.add_argument("--jobs", dest="jobs", type=int, default=PARALLEL)
    ap.add_argument("--no-named", action="store_true")
    ap.add_argument("--sanitize", dest="sanitize", type=int, metavar="N",
                    help=argparse.SUPPRESS)
    ap.add_argument("--worker", metavar="JOBJSON", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.long:
        args.n = 20000
    PARALLEL = args.jobs

    if args.worker:
        doc = json.load(open(args.worker))
        job = dict(kind=doc["kind"], elem=doc.get("elem", 0),
                   triple=open(doc["triple_path"], "rb").read(),
                   layout=doc.get("layout", dict(csA=0, csM=0)),
                   spans=doc.get("spans"), check_middle=doc.get("check_middle", False))
        if "ref_path" in doc:
            job["ref"] = open(doc["ref_path"], "rb").read()
        else:
            job["ref"] = None
        kind, rc, errmsg = run_worker(job)
        print("RESULT %s %s %s" % (kind, rc, errmsg))
        return 0

    if args.replay:
        return replay(args.replay)

    subprocess.run(["make", "-s", "-C", ROOT], check=True)
    lib = load()
    bases = load_bases()

    # cross-check each base against sbz_dctx_parse (the host walk accepts it)
    for b in bases:
        if b["kind"] == "zstd":
            if not parse_ok(lib, b):
                print("WARNING: base %s does not parse cleanly" % b["name"],
                      file=sys.stderr)
        b["ref"] = decode_ref(lib, b)

    named = build_named_seeds(bases) if not args.no_named else []
    if args.sanitize:
        args.n = args.sanitize
        return sanitize_run(args, bases, named)
    counts, saved, _ = run_all(args, bases, named)
    print_summary(counts, args)
    return 0


def replay(path):
    meta = json.load(open(path + ".json"))
    job = dict(kind=meta["kind"], elem=meta.get("elem", 0),
               triple=open(path, "rb").read(),
               layout=meta.get("layout", dict(csA=0, csM=0)),
               spans=meta.get("spans"), check_middle=meta.get("check_middle", False),
               base_name=meta.get("base_name"))
    lib = load()
    if meta.get("base_path") and meta.get("base_kind"):
        bp = os.path.join(ROOT, meta["base_path"])
        base = dict(kind=meta["base_kind"], src=open(bp, "rb").read(),
                    elem=meta.get("base_elem", 1), cs=meta["base_cs"],
                    name=meta.get("base_name"))
        job["ref"] = decode_ref(lib, base)
    kind, rc, errmsg = run_worker(job)
    print("RESULT %s %s %s  (input %s)" % (kind, rc, errmsg, path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
