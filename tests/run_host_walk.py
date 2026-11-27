#!/usr/bin/env python3
"""Verify the bounded host frame walk (P2 Stage 0) through libslabitz.so.

Compares against a DS-mirroring oracle and libzstd. Valid inputs (single and
concatenated zstd frames, raw/RLE/compressed blocks, a skippable frame and
checksummed frames) are decoded with sbz_dctx_decompress and checked byte-exact
against `zstandard`; frame/block counts come from sbz_dctx_counters and are
checked against the Python structural oracle (_walk); the GPU literal/sequence
totals are checked against the edu_oracle. Every check prints the number of
values compared and fails if that is zero.

Corrupt inputs, one per host-walk check, must return SBZ_ERR_CORRUPT (or
SBZ_ERR_LIMIT for the RLE-over-4 GiB case) and must not hang. Valid frames this
decoder does not handle (dictionary, window >= 2^29, no content size) must return
SBZ_ERR_UNSUPPORTED with a named message (T09).

Finally it builds and runs the ASan/UBSan fuzz binary (tests/fuzz_host_walk.c)
on a fixed-seed mutation corpus and requires that OK, CORRUPT and LIMIT are all
reached with no sanitizer report (UNSUPPORTED is counted too).
"""
import ctypes as ct
import os
import random
import re
import struct
import platform
import shutil
import subprocess
import sys
import zstandard as zstd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
from tools.edu_oracle import run_oracle

LIB = os.path.join(ROOT, "build", "libslabitz.so")
FUZZ = os.path.join(ROOT, "build", "fuzz_host_walk")
CORPUS = os.path.join(ROOT, "build", "fuzz_corpus.bin")
KDIR = os.environ.get("SLABITZ_KERNEL_DIR", os.path.join(ROOT, "build"))
TIMEOUT = 120
FUZZ_TIMEOUT = 60  # a healthy 200000-mutant run takes about 1 s
FUZZ_MUTANTS = 200000

SBZ_OK, SBZ_ERR_CORRUPT, SBZ_ERR_LIMIT, SBZ_ERR_UNSUPPORTED = 0, -5, -6, -7
ZSTD_MAGIC = 0xFD2FB528
BLOCK_MAX = 128 * 1024


class Counters(ct.Structure):
    _fields_ = [(n, ct.c_uint32) for n in (
        "FseHufW FseLLen FseOffs FseMLen DecompressLiteralsGroups HUF_WgtStreams "
        "Seq_Streams_DecodedItems HUF_Streams_DecodedBytes Seq_Streams HUF_Streams HufLit "
        "RAW_Streams RLE_Streams Blocks_RAW Blocks_RLE Blocks_CMP BlocksBytes_RAW "
        "BlocksBytes_RLE Frames Frames_UncompressedByteSize "
        "Frames_AcceptedUncompressedByteSize Frames_RequestTooLarge").split()]


def _u32(buf, off):
    return struct.unpack_from("<I", buf, off)[0]


def _walk(buf, content_size):
    """DS-mirroring structural oracle. Returns (frames, frame_infos, raw, rle, cmp)."""
    n = content_size
    pos = 0
    frames, finfos = [], []
    raw, rle, cmp = [], [], []
    run_r = run_l = run_c = 0
    byter = bytel = 0
    while pos < n:
        magic = _u32(buf, pos)
        while 0x184D2A50 <= magic <= 0x184D2A5F:
            fsz = _u32(buf, pos + 4)
            pos += 8 + fsz
            magic = _u32(buf, pos)
        if magic != ZSTD_MAGIC:
            break
        offs = pos
        pos += 4
        desc = buf[pos]
        pos += 1
        single_seg = bool(desc & 0x20)
        winsz = 0
        if not single_seg:
            wd = buf[pos]
            pos += 1
            wbase = 1 << (10 + (wd >> 3))
            winsz = wbase + (wbase >> 3) * (wd & 7)
        dict_flag = desc & 0x03
        dict_val = 0
        if dict_flag:
            nb = {1: 1, 2: 2, 3: 4}[dict_flag]
            dict_val = int.from_bytes(buf[pos:pos + nb], "little")
            pos += nb
        fcs_flag = (desc >> 6) & 0x3
        fcs_bytes = {0: (1 if single_seg else 0), 1: 2, 2: 4, 3: 8}[fcs_flag]
        if fcs_bytes:
            if fcs_bytes == 8:
                uncomp = struct.unpack_from("<Q", buf, pos)[0]
            else:
                uncomp = int.from_bytes(buf[pos:pos + fcs_bytes], "little")
                if fcs_bytes == 2:
                    uncomp += 256
            pos += fcs_bytes
        else:
            uncomp = 0
        if single_seg:
            winsz = uncomp
        # DS *Start / *BytesStart are PREFIXES (counts before this frame).
        pre_run_r, pre_run_l, pre_run_c = run_r, run_l, run_c
        pre_byter, pre_bytel = byter, bytel
        while True:
            bh = _u32(buf, pos) & 0xFFFFFF
            last = bh & 1
            btype = (bh >> 1) & 0x3
            bsize = (bh >> 3) & 0x3FFFFF
            pos += 3
            block_base = pos
            if btype == 2:
                cmp.append((block_base, bsize))
                run_c += 1
                pos += bsize
            elif btype == 0:
                raw.append((block_base, bsize))
                run_r += 1
                byter += bsize
                pos += bsize
            elif btype == 1:
                sym = buf[pos]
                pos += 1
                rle.append((sym, bsize))
                run_l += 1
                bytel += bsize
            else:
                raise AssertionError("reserved block type")
            if last:
                break
        if desc & 0x04:
            pos += 4
        frames.append((offs, pos - offs))
        finfos.append(dict(windowSize=winsz, uncompSize=uncomp, dictionary=dict_val,
                           rawBlockStart=pre_run_r, rleBlockStart=pre_run_l,
                           cmpBlockStart=pre_run_c,
                           rawBlockBytesStart=pre_byter, rleBlockBytesStart=pre_bytel))
    return frames, finfos, raw, rle, cmp


def load():
    lib = ct.CDLL(LIB)
    lib.sbz_dctx_create_cuda.restype = ct.c_void_p
    lib.sbz_dctx_create_cuda.argtypes = [ct.c_int, ct.c_char_p, ct.c_char_p, ct.c_size_t]
    lib.sbz_dctx_destroy.argtypes = [ct.c_void_p]
    lib.sbz_dctx_error.restype = ct.c_char_p
    lib.sbz_dctx_error.argtypes = [ct.c_void_p]
    lib.sbz_dctx_decompress.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_size_t]
    lib.sbz_dctx_counters.argtypes = [ct.c_void_p, ct.POINTER(Counters)]
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


def make(lib, err):
    dc = lib.sbz_dctx_create_cuda(0, KDIR.encode(), err, len(err))
    return dc


# ---------------------------------------------------------------- test data
def payload(seed):
    random.seed(seed)
    words = "alpha beta gamma delta epsilon zeta eta theta iota kappa".split()
    parts = []
    for _ in range(6):
        parts.append((" ".join(random.choice(words) for _ in range(15000)) + " ").encode())
        parts.append(bytes(random.randrange(0, 256) for _ in range(60000)))
    return b"".join(parts)


def make_raw_rle_frame():
    """Single-segment frame with a Raw block and an RLE block (no compressed)."""
    total = 5 + 300
    desc = 0x20 | (0b10 << 6)  # single_segment, FCS flag=2 (4 bytes)
    b1 = 0 | (0 << 1) | (5 << 3)            # last=0, type=0 raw, size=5
    b2 = 1 | (1 << 1) | (300 << 3)          # last=1, type=1 rle, size=300
    return struct.pack("<I", ZSTD_MAGIC) + bytes([desc]) + struct.pack("<I", total) + \
        b1.to_bytes(3, "little") + b"HELLO" + b2.to_bytes(3, "little") + b"B"


def make_skippable(payload_bytes):
    """A skippable frame wrapping payload_bytes."""
    return struct.pack("<I", 0x184D2A50) + struct.pack("<I", len(payload_bytes)) + \
        payload_bytes


def oracle_lit_seq(comp, frames):
    """Aggregate the edu_oracle's literal bytes / sequence items over each frame."""
    lit = 0
    seq = 0
    for off, size in frames:
        o = run_oracle(comp[off:off + size])
        lit += sum(len(v) for v in o.get("LITS", []))
        seq += sum(len(v) // 3 for v in o.get("SEQ", []))
    return lit, seq


# ---------------------------------------------------------------- valid cases
def check_valid(lib, name, comp, expected, compared):
    content_size = len(comp)
    buf = comp + b"\x00" * ((-len(comp)) % 4)
    frames, finfos, raw, rle, cmp = _walk(buf, content_size)

    dc = make(lib, ct.create_string_buffer(512))
    if not dc:
        print("[%s] FAIL: create context" % name)
        return compared, False
    try:
        rc = lib.sbz_dctx_decompress(dc, comp, len(comp))
        if rc != 0:
            print("[%s] FAIL: decompress rc=%d: %s" %
                  (name, rc, lib.sbz_dctx_error(dc).decode()))
            return compared, False
        got = read_buf(lib, dc, "UnCompressedFramesData")[:len(expected)]
        cnt = Counters()
        assert lib.sbz_dctx_counters(dc, ct.byref(cnt)) == 0
    finally:
        lib.sbz_dctx_destroy(dc)

    ok = got == expected
    compared += len(expected)  # one value per output byte

    failures = []

    def chk(what, got_v, exp_v):
        nonlocal compared, ok
        compared += 1
        if got_v != exp_v:
            ok = False
            failures.append("%s: got %r exp %r" % (what, got_v, exp_v))

    chk("Frames", cnt.Frames, len(frames))
    chk("Blocks_RAW", cnt.Blocks_RAW, len(raw))
    chk("Blocks_RLE", cnt.Blocks_RLE, len(rle))
    chk("Blocks_CMP", cnt.Blocks_CMP, len(cmp))
    chk("BlocksBytes_RAW", cnt.BlocksBytes_RAW, sum(s for _, s in raw))
    chk("BlocksBytes_RLE", cnt.BlocksBytes_RLE, sum(s for _, s in rle))
    lit, seq = oracle_lit_seq(comp, frames)
    chk("HUF_Streams_DecodedBytes (oracle)", cnt.HUF_Streams_DecodedBytes, lit)
    chk("Seq_Streams_DecodedItems (oracle)", cnt.Seq_Streams_DecodedItems, seq)
    for f in failures:
        print("    " + f)

    tag = "OK" if ok else "FAIL"
    print("[%s] %s: frames=%d raw/rle/cmp=%d/%d/%d outBytes=%d lit=%d seq=%d worth %d values"
          % (name, tag, cnt.Frames, cnt.Blocks_RAW, cnt.Blocks_RLE, cnt.Blocks_CMP,
             len(got), cnt.HUF_Streams_DecodedBytes, cnt.Seq_Streams_DecodedItems, compared))
    return compared, ok


# ---------------------------------------------------------------- corrupt cases
def raw_frame(payload, fcs_flag=2, last=True, desc_or=0, corrupt_block=None,
              checksum_bytes=0, not_last=False, window=None, block_size=None,
              dict_id=None):
    """Single zstd frame with one raw block. Hand-crafted to hit specific checks."""
    total = len(payload) if fcs_flag in (0, 2) else payload
    desc = 0x20 | (fcs_flag << 6) | desc_or
    if dict_id is not None:
        desc |= 0x01
    out = struct.pack("<I", ZSTD_MAGIC) + bytes([desc])
    if not (desc & 0x20):
        out += bytes([window if window is not None else 0])
    if dict_id is not None:
        out += struct.pack("<B", dict_id)
    # content size
    if desc & 0x20:
        if fcs_flag == 0:
            out += struct.pack("<B", len(payload) & 0xFF)
        elif fcs_flag == 1:
            out += struct.pack("<H", (len(payload) - 256) & 0xFFFF)
        elif fcs_flag == 2:
            out += struct.pack("<I", len(payload))
        else:
            out += struct.pack("<Q", len(payload))
    # block header
    if block_size is None:
        block_size = len(payload)
    blk_type = 0
    bsize = block_size
    if corrupt_block == "type3":
        blk_type = 3
        bsize = 0
    elif corrupt_block == "rle_size0":
        blk_type = 2
        bsize = 0
    lastbit = 0 if not_last else (1 if last else 0)
    bh = lastbit | (blk_type << 1) | (bsize << 3)
    out += bh.to_bytes(3, "little")
    if corrupt_block in ("type3", "rle_size0"):
        return out
    if blk_type == 2:
        return out + payload
    if blk_type == 1:
        return out + payload[:1]
    out += payload
    if checksum_bytes:
        out += b"\x00" * checksum_bytes
    return out


CORRUPT_CASES = []


def add_corrupt(name, comp, expect, msg=None):
    CORRUPT_CASES.append((name, comp, expect, msg))


def hand_frame(blocks, window_log=None, fcs=None, dict_id=None):
    """One zstd frame with explicit header fields. window_log=None: single
    segment (fcs required, 4-byte field); otherwise a window descriptor of
    2^window_log and, if fcs is not None, a 4-byte content size. dict_id: a
    1-byte Dictionary_ID field. blocks: list of (type, size, payload)."""
    desc = 0
    if window_log is None:
        desc |= 0x20
    if fcs is not None:
        desc |= 2 << 6
    if dict_id is not None:
        desc |= 0x01
    out = struct.pack("<I", ZSTD_MAGIC) + bytes([desc])
    if window_log is not None:
        out += bytes([(window_log - 10) << 3])
    if dict_id is not None:
        out += bytes([dict_id])
    if fcs is not None:
        out += struct.pack("<I", fcs)
    for i, (btype, size, payload) in enumerate(blocks):
        hdr = (1 if i == len(blocks) - 1 else 0) | (btype << 1) | (size << 3)
        out += struct.pack("<I", hdr)[:3] + payload
    return out


def rle_limit_frame():
    """Non-single-segment frame of RLE blocks; decoded bytes exceed 32 bits.

    A 128 KiB window and a (lying) 4-byte content size of 16, then 32768 RLE
    blocks of BLOCK_MAX decoded bytes each: the running raw/RLE byte total passes
    4 GiB, which the walk reports as SBZ_ERR_LIMIT.
    """
    nblocks = 32768  # 32768 x 128 KiB = 4 GiB > 0xFFFFFFF0
    # magic, desc (FCS flag 2: 4 bytes), window descriptor 2^17, content size 16
    pre = struct.pack("<I", ZSTD_MAGIC) + bytes([0x80, (17 - 10) << 3]) + struct.pack("<I", 16)
    mid = b"\x02\x00\x10\x42" * (nblocks - 1)      # RLE, size=131072, last=0, symbol 'B'
    last = b"\x03\x00\x10\x42"                     # RLE, size=131072, last=1, symbol 'B'
    return pre + mid + last


def build_corrupt_cases():
    tiny = b"TRUNC-TEST" * 3

    # A. fewer than 4 bytes left where a frame should start
    add_corrupt("A-fewer-than-4-bytes", b"\xFD", SBZ_ERR_CORRUPT)
    # B. magic neither zstd nor skippable
    add_corrupt("B-bad-magic", b"\x11\x22\x33\x44" + raw_frame(tiny)[4:], SBZ_ERR_CORRUPT)
    # C. skippable frame: size field or skipped bytes run past content
    add_corrupt("C-skippable-size-field", struct.pack("<I", 0x184D2A50), SBZ_ERR_CORRUPT)
    add_corrupt("C-skippable-past", struct.pack("<I", 0x184D2A50) + struct.pack("<I", 256),
                SBZ_ERR_CORRUPT)
    # D. frame header runs past content (8-byte content size declared, none present)
    add_corrupt("D-header-past", struct.pack("<I", ZSTD_MAGIC) + bytes([0xE0, 0x01]),
                SBZ_ERR_CORRUPT)
    # E. reserved descriptor bit (bit 3) set
    fr = raw_frame(tiny)
    add_corrupt("E-reserved-desc-bit", fr[:4] + bytes([fr[4] | 0x08]) + fr[5:], SBZ_ERR_CORRUPT)
    # F. Dictionary_ID field (T09: UNSUPPORTED, any value, including 0)
    add_corrupt("F-dict-id", struct.pack("<I", ZSTD_MAGIC) + bytes([0x21, 0x07, 0x05]),
                SBZ_ERR_UNSUPPORTED, "frame 0: dictionary")
    # G. block header (3 bytes) runs past content (frame not terminated)
    add_corrupt("G-no-block", struct.pack("<I", ZSTD_MAGIC) + bytes([0x20, 0x00]),
                SBZ_ERR_CORRUPT)
    # H. reserved block type 3
    add_corrupt("H-reserved-block-type", raw_frame(tiny, corrupt_block="type3"),
                SBZ_ERR_CORRUPT)
    # I. block size above 128 KiB
    add_corrupt("I-block-size-too-big", raw_frame(b"", block_size=BLOCK_MAX + 1),
                SBZ_ERR_CORRUPT)
    # J. raw block bytes run past content
    add_corrupt("J-raw-past", raw_frame(b"ABCDE", block_size=100), SBZ_ERR_CORRUPT)
    # K. compressed block of size 0
    add_corrupt("K-cmp-size-0", raw_frame(tiny, corrupt_block="rle_size0"), SBZ_ERR_CORRUPT)
    # L. content ends before a block with the last-block flag
    add_corrupt("L-not-terminated", raw_frame(tiny, not_last=True), SBZ_ERR_CORRUPT)
    # M. content checksum runs past content
    add_corrupt("M-checksum-past", raw_frame(tiny, desc_or=0x04, checksum_bytes=1),
                SBZ_ERR_CORRUPT)
    # N. SBZ_ERR_LIMIT: RLE blocks declare over 4 GiB decoded
    add_corrupt("N-limit-rle-4gib", rle_limit_frame(), SBZ_ERR_LIMIT)
    # T09: frame-level rejects (UNSUPPORTED) and Block_Maximum_Size = min(window, 128 KiB)
    add_corrupt("T09-dict-id-0", hand_frame([(0, 5, b"ABCDE")], fcs=5, window_log=10, dict_id=0),
                SBZ_ERR_UNSUPPORTED, "frame 0: dictionary (id 0)")
    add_corrupt("T09-window-2^29", hand_frame([(0, 5, b"ABCDE")], window_log=29, fcs=5),
                SBZ_ERR_UNSUPPORTED, "frame 0: window 2^29 exceeds 2^29")
    add_corrupt("T09-window-2^31", hand_frame([(0, 5, b"ABCDE")], window_log=31, fcs=5),
                SBZ_ERR_UNSUPPORTED, "frame 0: window 2^31 exceeds 2^29")
    add_corrupt("T09-no-fcs", hand_frame([(0, 5, b"ABCDE")], window_log=10),
                SBZ_ERR_UNSUPPORTED, "frame 0: no content size")
    add_corrupt("T09-raw-block-over-window",
                hand_frame([(0, 1500, b"x" * 1500)], window_log=10, fcs=1500),
                SBZ_ERR_CORRUPT, "block size 1500 exceeds the block maximum 1024")
    add_corrupt("T09-rle-block-over-window", hand_frame([(1, 1500, b"y")], window_log=10, fcs=1500),
                SBZ_ERR_CORRUPT, "block size 1500 exceeds the block maximum 1024")
    add_corrupt("T09-cmp-block-over-fcs-window",
                hand_frame([(2, 40, bytes(40))], fcs=32),
                SBZ_ERR_CORRUPT, "block size 40 exceeds the block maximum 32")


def check_corrupt(lib, name, comp, expect, msg=None):
    dc = make(lib, ct.create_string_buffer(512))
    if not dc:
        print("[%s] FAIL: create context" % name)
        return False
    try:
        rc = lib.sbz_dctx_decompress(dc, comp, len(comp))
        err = (lib.sbz_dctx_error(dc) or b"").decode(errors="replace")
    finally:
        lib.sbz_dctx_destroy(dc)
    ok = rc == expect and (msg is None or msg in err)
    print("[%s] %s: rc=%d (want %d)%s" % (name, "OK" if ok else "FAIL", rc, expect,
                                         (" msg %r" % err) if msg is not None else ""))
    return ok


# ---------------------------------------------------------------- fuzz
def fuzz_seeds():
    """Small valid inputs (each well under 10 kB, so most mutations land in
    headers rather than in skipped compressed payload) covering every structure
    the walk parses: raw, RLE and compressed blocks, multi-block frames, content
    size present/absent, checksums, skippable frames, concatenated frames."""
    rnd = random.Random(1234)
    text = b"hello-host-walk-fuzz " * 40
    noise = bytes(rnd.getrandbits(8) for _ in range(700))       # incompressible -> raw block
    zeros = bytes(300000)                                       # several RLE-ish 128 KiB blocks
    mixed = (bytes(rnd.getrandbits(8) for _ in range(200)) + bytes(140000)) * 2
    c = lambda lvl, **kw: zstd.ZstdCompressor(level=lvl, **kw)
    skip = struct.pack("<II", 0x184D2A53, 5) + b"skip!"
    seeds = [
        c(3).compress(b"hello-host-walk-fuzz" * 4),
        c(1).compress(text),
        c(19, write_checksum=True).compress(text),
        c(3, write_content_size=False).compress(text),
        c(3).compress(noise),
        c(3).compress(zeros),
        c(3, write_checksum=True).compress(mixed),
        c(3).compress(text) + c(3).compress(noise) + c(3).compress(zeros),
        skip + c(3).compress(text) + skip,
        c(3).compress(b""),
    ]
    for sd in seeds:
        assert len(sd) < 10000, len(sd)
    return seeds


def run_fuzz():
    subprocess.run(["make", "-s", "-C", ROOT, "build/fuzz_host_walk"], check=True)
    with open(CORPUS, "wb") as f:
        for comp in fuzz_seeds():
            f.write(struct.pack("<I", len(comp)))
            f.write(comp)
    # ASan (the pixi gcc 13.4 runtime) on Linux 6.x kernels with high mmap
    # randomisation (vm.mmap_rnd_bits = 32) sometimes cannot map its shadow memory
    # and loops on "AddressSanitizer:DEADLYSIGNAL" before main: 7 of 30 runs here
    # (2026-11-20), 0 of 30 with randomisation off. So run it under `setarch -R`
    # (no root needed); without setarch, retry a run that died that way.
    cmd = [FUZZ, CORPUS, str(FUZZ_MUTANTS)]
    if shutil.which("setarch"):
        cmd = ["setarch", platform.machine(), "-R"] + cmd
    for attempt in range(3):
        try:
            # cwd build/: a mutant that trips the binary's 10 s watchdog is saved
            # there as fuzz_hang.bin (exit 3)
            p = subprocess.run(cmd, timeout=FUZZ_TIMEOUT, capture_output=True, text=True,
                               cwd=os.path.dirname(FUZZ),
                               env=dict(os.environ,
                                        ASAN_OPTIONS="detect_leaks=0:abort_on_error=1"))
        except subprocess.TimeoutExpired as ex:
            err = ex.stderr if isinstance(ex.stderr, str) else (ex.stderr or b"").decode(
                errors="replace")
            if "DEADLYSIGNAL" in err and "mutants" not in err and attempt < 2:
                print("[fuzz] ASan failed to start (DEADLYSIGNAL, address randomisation); "
                      "retrying")
                continue
            print("[fuzz] FAIL: timeout after %ds; partial output:" % FUZZ_TIMEOUT)
            for part in (ex.stdout, ex.stderr):
                if part:
                    print((part if isinstance(part, str) else part.decode(errors="replace"))
                          [-3000:])
            return False
        break
    print("fuzz exit=%d" % p.returncode)
    print(p.stdout.strip())
    if p.stderr.strip():
        print("fuzz stderr:\n" + p.stderr.strip())
    if p.returncode != 0:
        print("[fuzz] FAIL: nonzero exit (sanitizer or out-of-set return)")
        return False
    m = re.search(r"mutants=(\d+) OK=(\d+) CORRUPT=(\d+) LIMIT=(\d+) UNSUPPORTED=(\d+)",
                  p.stdout)
    if not m:
        print("[fuzz] FAIL: could not parse fuzz histogram")
        return False
    n, ok, corrupt, limit, unsup = map(int, m.groups())
    print("[fuzz] %s: OK=%d CORRUPT=%d LIMIT=%d (all nonzero) UNSUPPORTED=%d"
          % ("OK" if ok and corrupt and limit else "FAIL", ok, corrupt, limit, unsup))
    return ok > 0 and corrupt > 0 and limit > 0


def main():
    subprocess.run(["make", "-s", "-C", ROOT], check=True)
    lib = load()
    compared = 0
    ok = True

    # --- valid inputs ---
    for lev in (1, 3, 6, 12):
        exp = payload(10 + lev)
        comp = zstd.ZstdCompressor(level=lev).compress(exp)
        compared, o = check_valid(lib, "single lev%d" % lev, comp, exp, compared)
        ok &= o
    a = zstd.ZstdCompressor(level=5).compress(payload(1))
    b = zstd.ZstdCompressor(level=9).compress(payload(2))
    compared, o = check_valid(lib, "concatenated", a + b, payload(1) + payload(2), compared)
    ok &= o
    syn = make_raw_rle_frame()
    expected_syn = b"HELLO" + b"B" * 300
    compared, o = check_valid(lib, "raw+rle synthetic", syn, expected_syn, compared)
    ok &= o
    # a zstd frame, a skippable frame between two zstd frames (walk skips it)
    sk = a + make_skippable(b"Skippable metadata goes here.") + b
    compared, o = check_valid(lib, "zstd+skippable+zstd", sk, payload(1) + payload(2), compared)
    ok &= o
    # checksummed frame
    csum = zstd.ZstdCompressor(level=3, write_checksum=True).compress(payload(3))
    compared, o = check_valid(lib, "checksum frame", csum, payload(3), compared)
    ok &= o

    if compared == 0:
        print("FAIL: no values compared")
        ok = False
    else:
        print("compared %d values" % compared)

    # --- corrupt inputs ---
    build_corrupt_cases()
    for name, comp, expect, msg in CORRUPT_CASES:
        o = check_corrupt(lib, name, comp, expect, msg)
        ok &= o

    # --- fuzz ---
    f_ok = run_fuzz()
    ok &= f_ok

    print("OVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
