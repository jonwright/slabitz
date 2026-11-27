#!/usr/bin/env python3
"""T09: every frame stays inside its own output range, and rejects leave the
context usable.

The decoder owns its output buffer, so a caller cannot pad it with canary
bytes. Instead each bad frame is decoded in one call between good frames:

    [A: zstd text] [BAD] [C: canary, one raw block of 0xC5] [B: zstd text]

C is written by MemsetMemcpy, which runs before ExecuteSequences, so a
compressed BAD frame that overflowed its content size would overwrite C's
first bytes deterministically. A, C and B must be byte-exact after the call,
and the call must return the stated code and message.

Cases (each in two modes):
  cmp-over     a compressed frame whose content size is 1000 bytes smaller than
               what it decodes to -> CORRUPT "decodes past its content size"
  cmp-short    the same frame with a content size 1000 bytes larger ->
               CORRUPT "decodes short of its content size"
  raw-over     a raw block of 150 bytes in a frame declaring 100 -> CORRUPT
  rle-over     an RLE block of 150 bytes in a frame declaring 100 -> CORRUPT
  patch-from   a `zstd --patch-from` frame (T07): its first match reaches
               before the frame start -> CORRUPT "match offset before frame start"

Modes: "exact" (a fresh context: its first call reads the counters after each
stage) and "queued" (a context that has decoded before, so the call is queued
and checked at the wait, which replays it exactly; the replay count must rise).

Then one context runs a good call, each reject case (the bad frames above, a
frame without content size, a 2^29 window, a dictionary, more than 1 GiB of
output), and a good call after each: every good call must be byte-exact.

Usage: timeout 600 python3 tests/run_frame_bounds.py
"""
import ctypes as ct
import os
import struct
import sys

import zstandard

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = os.path.join(ROOT, "build", "libslabitz.so")
KDIR = os.environ.get("SLABITZ_KERNEL_DIR", os.path.join(ROOT, "build"))
PATCHFROM = os.path.join(ROOT, "tests", "data", "frame_bounds", "patchfrom_frame.zst")

SBZ_ERR_BACKEND, SBZ_ERR_CORRUPT, SBZ_ERR_LIMIT, SBZ_ERR_UNSUPPORTED = -1, -5, -6, -7
ZSTD_MAGIC = 0xFD2FB528


class FrameSpan(ct.Structure):
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
                                               ct.POINTER(FrameSpan), ct.c_uint32]
    lib.sbz_dctx_read_buffer.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_void_p,
                                         ct.POINTER(ct.c_size_t)]
    u32p = ct.POINTER(ct.c_uint32)
    lib.sbz_dctx_zstd_stats.argtypes = [ct.c_void_p, u32p, u32p, u32p, u32p]
    return lib


def make_dctx(lib):
    err = ct.create_string_buffer(512)
    dc = lib.sbz_dctx_create_cuda(0, KDIR.encode(), err, len(err))
    if not dc:
        raise SystemExit("sbz_dctx_create_cuda: %s" % err.value.decode())
    return dc


def read_output(lib, dc):
    n = ct.c_size_t(0)
    assert lib.sbz_dctx_read_buffer(dc, b"UnCompressedFramesData", None, ct.byref(n)) == 0
    b = ct.create_string_buffer(max(n.value, 1))
    rc = lib.sbz_dctx_read_buffer(dc, b"UnCompressedFramesData", b, ct.byref(n))
    if rc != 0:
        raise SystemExit("read output: %s" % lib.sbz_dctx_error(dc).decode())
    return b.raw[:n.value]


def replays(lib, dc):
    e, r = ct.c_uint32(0), ct.c_uint32(0)
    lib.sbz_dctx_zstd_stats(dc, ct.byref(e), ct.byref(r), None, None)
    return r.value


def err(lib, dc):
    return (lib.sbz_dctx_error(dc) or b"").decode(errors="replace")


# ---------------------------------------------------------------- frames
def text(seed, n):
    words = [b"slabitz", b"frame", b"bounds", b"detector", b"pixel", b"zstd", b"gpu",
             b"offset", b"block", b"canary"]
    out, x = bytearray(), seed
    while len(out) < n:
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        out += words[x % len(words)] + b" %d " % (x % 997)
    return bytes(out[:n])


def zframe(data, level=3):
    return zstandard.ZstdCompressor(level=level, write_content_size=True).compress(data)


def block_header(btype, size, last=True):
    return struct.pack("<I", (1 if last else 0) | (btype << 1) | (size << 3))[:3]


def single_raw_frame(payload):
    """Single segment, 4-byte content size, one raw block."""
    return (struct.pack("<I", ZSTD_MAGIC) + bytes([0xA0]) + struct.pack("<I", len(payload))
            + block_header(0, len(payload)) + payload)


def lying_frame(btype, size, fcs, payload):
    """1 KiB window, 4-byte content size `fcs`, one raw/RLE block of `size` bytes."""
    return (struct.pack("<I", ZSTD_MAGIC) + bytes([0x80, 0x00]) + struct.pack("<I", fcs)
            + block_header(btype, size) + payload)


def set_fcs(frame, fcs):
    """Rewrite the content size of a single-segment frame in place (same field size)."""
    desc = frame[4]
    assert desc & 0x20 and not desc & 0x03, "single segment, no dictionary expected"
    flag = desc >> 6
    size = {0: 1, 1: 2, 2: 4, 3: 8}[flag]
    if flag == 1:
        field = struct.pack("<H", fcs - 256)
    else:
        field = (fcs).to_bytes(size, "little")
    return frame[:5] + field + frame[5 + size:]


def one_compressed_block(frame):
    desc = frame[4]
    hdr = 5 + {0: 1, 1: 2, 2: 4, 3: 8}[desc >> 6]
    bh = int.from_bytes(frame[hdr:hdr + 3], "little")
    return (bh >> 1) & 3 == 2 and bh & 1 == 1


def big_rle_frame(fcs_log=27):
    """Single segment, content size 2^fcs_log, made of 128 KiB RLE blocks."""
    n = (1 << fcs_log) // (128 * 1024)
    out = struct.pack("<I", ZSTD_MAGIC) + bytes([0xA0]) + struct.pack("<I", 1 << fcs_log)
    for i in range(n):
        out += block_header(1, 128 * 1024, last=(i == n - 1)) + b"\x5a"
    return out


A_DATA, B_DATA, BAD_DATA = text(1, 7000), text(2, 9000), text(3, 5000)
C_DATA = b"\xc5" * 2000
A, B, C = zframe(A_DATA), zframe(B_DATA), single_raw_frame(C_DATA)
CMP = zframe(BAD_DATA)
assert one_compressed_block(CMP), "the bad zstd frame must be one compressed block"

BAD = {
    "cmp-over": (set_fcs(CMP, len(BAD_DATA) - 1000), "decodes past its content size"),
    "cmp-short": (set_fcs(CMP, len(BAD_DATA) + 1000), "decodes short of its content size"),
    "raw-over": (lying_frame(0, 150, 100, b"r" * 150), "decodes past its content size"),
    "rle-over": (lying_frame(1, 150, 100, b"q"), "decodes past its content size"),
    "patch-from": (open(PATCHFROM, "rb").read(), "match offset before frame start"),
}
REJECTS = {
    "no-fcs": (struct.pack("<I", ZSTD_MAGIC) + bytes([0x00, 0x00]) + block_header(0, 5)
               + b"hello", SBZ_ERR_UNSUPPORTED, "frame 0: no content size"),
    "window-2^29": (struct.pack("<I", ZSTD_MAGIC) + bytes([0x80, 19 << 3])
                    + struct.pack("<I", 5) + block_header(0, 5) + b"hello",
                    SBZ_ERR_UNSUPPORTED, "frame 0: window 2^29 exceeds 2^29"),
    "dictionary": (struct.pack("<I", ZSTD_MAGIC) + bytes([0x21, 0x00, 0x05])
                   + block_header(0, 5) + b"hello", SBZ_ERR_UNSUPPORTED,
                   "frame 0: dictionary"),
    "over-1GiB": (big_rle_frame() * 9, SBZ_ERR_LIMIT, "exceeds 1 GiB"),
}


def bad_call(lib, dc, bad):
    """[A][bad][C][B] in one decompress_frames call -> (rc, message, output)."""
    parts = [A, bad, C, B]
    buf, spans, o = b"".join(parts), [], 0
    for p in parts:
        spans.append(FrameSpan(o, len(p)))
        o += len(p)
    arr = (FrameSpan * len(spans))(*spans)
    rc = lib.sbz_dctx_decompress_frames(dc, buf, len(buf), arr, len(spans))
    return rc, err(lib, dc), read_output(lib, dc)


def neighbours_ok(out, bad_fcs):
    """A, C and B at their offsets (frame i starts at the sum of the content sizes
    before it); returns a list of problems."""
    probs = []
    a0, c0 = 0, len(A_DATA) + bad_fcs
    b0 = c0 + len(C_DATA)
    if out[a0:a0 + len(A_DATA)] != A_DATA:
        probs.append("frame A damaged")
    got_c = out[c0:c0 + len(C_DATA)]
    changed = sum(1 for x, y in zip(got_c, C_DATA) if x != y) + len(C_DATA) - len(got_c)
    if changed:
        probs.append("%d canary bytes changed" % changed)
    if out[b0:b0 + len(B_DATA)] != B_DATA:
        probs.append("frame B damaged")
    return probs


def fcs_of(frame):
    desc = frame[4]
    single, flag = desc & 0x20, desc >> 6
    pos = 5 + (0 if single else 1) + {0: 0, 1: 1, 2: 2, 3: 4}[desc & 3]
    size = {0: 1 if single else 0, 1: 2, 2: 4, 3: 8}[flag]
    v = int.from_bytes(frame[pos:pos + size], "little")
    return v + 256 if size == 2 else v


def good_call(lib, dc):
    rc = lib.sbz_dctx_decompress(dc, A, len(A))
    if rc != 0:
        return "good call rc=%d (%s)" % (rc, err(lib, dc))
    if read_output(lib, dc)[:len(A_DATA)] != A_DATA:
        return "good call output differs"
    return None


def main():
    lib = load()
    fails = []

    for mode in ("exact", "queued"):
        for name, (bad, msg) in BAD.items():
            dc = make_dctx(lib)
            try:
                if mode == "queued":
                    g = good_call(lib, dc)  # primes the capacities: next call is queued
                    if g:
                        fails.append("%s/%s: priming %s" % (mode, name, g))
                        continue
                r0 = replays(lib, dc)
                rc, m, out = bad_call(lib, dc, bad)
                r1 = replays(lib, dc)
            finally:
                lib.sbz_dctx_destroy(dc)
            probs = []
            if rc != SBZ_ERR_CORRUPT:
                probs.append("rc=%d, want %d" % (rc, SBZ_ERR_CORRUPT))
            if msg not in m:
                probs.append("message %r lacks %r" % (m, msg))
            if mode == "queued" and r1 != r0 + 1:
                probs.append("not queued (replays %d -> %d)" % (r0, r1))
            probs += neighbours_ok(out, fcs_of(bad))
            print("%-6s %-10s %s  rc=%d  %s" % (mode, name, "ok  " if not probs else "FAIL",
                                               rc, m))
            for p in probs:
                fails.append("%s/%s: %s" % (mode, name, p))
                print("       " + p)

    # one context: every reject leaves it usable
    dc = make_dctx(lib)
    try:
        g = good_call(lib, dc)
        print("reuse  first good call  %s" % ("ok" if not g else "FAIL " + g))
        if g:
            fails.append("reuse: first good call: " + g)
        cases = [(n, f, SBZ_ERR_CORRUPT, m) for n, (f, m) in BAD.items()]
        cases += [(n, f, c, m) for n, (f, c, m) in REJECTS.items()]
        for name, frame, code, msg in cases:
            rc = lib.sbz_dctx_decompress(dc, frame, len(frame))
            m = err(lib, dc)
            g = good_call(lib, dc)
            probs = []
            if rc != code or msg not in m:
                probs.append("rc=%d (%s), want %d with %r" % (rc, m, code, msg))
            if g:
                probs.append("good call after it: " + g)
            print("reuse  %-10s %s  rc=%d  %s" % (name, "ok  " if not probs else "FAIL", rc, m))
            for p in probs:
                fails.append("reuse/%s: %s" % (name, p))
                print("       " + p)
    finally:
        lib.sbz_dctx_destroy(dc)

    print("PASS" if not fails else "FAIL: %d problems" % len(fails))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
