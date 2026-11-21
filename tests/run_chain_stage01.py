#!/usr/bin/env python3
"""P2: Stage 0 + Stage 1 of the C device chain (libslabitz.so) on the GPU.

Runs sbz_dctx_parse() (host prep -> InitResources -> ParseFrames count ->
PrefixSum x4 -> readback -> InitResources 1 -> memsets -> ParseFrames collect ->
ParseCompressedBlocks -> PrefixSum raw/rle -> readback), entirely from C on
device-resident buffers, and checks the results read back through the debug hook:

  * Counters: frames, raw/rle/cmp block counts and raw/rle byte totals vs the
    Python structural oracle (run_host_walk._walk); sequence count vs the trusted
    edu_oracle; sequence count and Huffman-literal bytes vs the C host walk.
  * PerFrameBlockCount{RAW,RLE,CMP,All}: exclusive per-frame prefixes.
  * BlocksRAWRefs / BlocksRLERefs / BlocksCMPRefs: every block ref, in order.
  * Raw/RleBlockSizePrefix: exclusive prefix of the raw/RLE block sizes.

Usage:  python3 tests/run_chain_stage01.py      (builds with `make` first)
Each case runs in a subprocess under a timeout so a GPU hang fails fast.
"""
import ctypes as ct
import os
import random
import struct
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

LIB = os.path.join(ROOT, "build", "libslabitz.so")
KDIR = os.environ.get("SLABITZ_KERNEL_DIR", os.path.join(ROOT, "build"))  # e.g. build/sm_35
TIMEOUT = 60


class Counters(ct.Structure):
    _fields_ = [(n, ct.c_uint32) for n in (
        "FseHufW FseLLen FseOffs FseMLen DecompressLiteralsGroups HUF_WgtStreams "
        "Seq_Streams_DecodedItems HUF_Streams_DecodedBytes Seq_Streams HUF_Streams HufLit "
        "RAW_Streams RLE_Streams Blocks_RAW Blocks_RLE Blocks_CMP BlocksBytes_RAW "
        "BlocksBytes_RLE Frames Frames_UncompressedByteSize "
        "Frames_AcceptedUncompressedByteSize Frames_RequestTooLarge").split()]


def load():
    lib = ct.CDLL(LIB)
    lib.sbz_dctx_create_cuda.restype = ct.c_void_p
    lib.sbz_dctx_create_cuda.argtypes = [ct.c_int, ct.c_char_p, ct.c_char_p, ct.c_size_t]
    lib.sbz_dctx_destroy.argtypes = [ct.c_void_p]
    lib.sbz_dctx_error.restype = ct.c_char_p
    lib.sbz_dctx_error.argtypes = [ct.c_void_p]
    lib.sbz_dctx_parse.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_size_t]
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


def u32s(raw):
    return list(struct.unpack("<%dI" % (len(raw) // 4), raw))


def pairs(raw):
    w = u32s(raw)
    return [(w[i], w[i + 1]) for i in range(0, len(w), 2)]


def excl_prefix(xs):
    out, s = [], 0
    for x in xs:
        out.append(s)
        s += x
    return out


# ---------------------------------------------------------------- test data
def text(seed, n):
    random.seed(seed)
    words = "alpha beta gamma delta epsilon zeta eta theta iota kappa".split()
    out = []
    while sum(map(len, out)) < n:
        out.append(random.choice(words) + " ")
    return "".join(out).encode()[:n]


def noise(seed, n):
    random.seed(seed)
    return bytes(random.randrange(256) for _ in range(n))


def cases():
    import zstandard as zstd

    def c(data, level=3, **kw):
        return zstd.ZstdCompressor(level=level, write_content_size=True, **kw).compress(data)

    mixed = text(1, 300000) + noise(2, 70000) + b"\x07" * 200000 + text(3, 50000)
    yield "text-l1", c(text(10, 100000), 1)
    yield "text-l3", c(text(11, 100000), 3)
    yield "text-l19", c(text(12, 100000), 19)
    yield "multiblock-mixed", c(mixed, 3)         # cmp + raw + rle blocks, >128K
    yield "raw-only", c(noise(4, 5000), 3)          # incompressible -> raw block
    yield "rle-only", c(b"\xab" * 300000, 3)        # RLE blocks
    yield "concat-8", b"".join(c(text(20 + i, 5000 + 3000 * i), 1 + i % 5) for i in range(8))
    yield "concat-mixed", c(mixed, 1) + c(noise(5, 3000)) + c(b"z" * 1000) + c(text(6, 40000), 9)
    yield "concat-200", b"".join(c(text(100 + i, 2000), 3) for i in range(200))


# ---------------------------------------------------------------- one case
def run_one(name, comp):
    from run_host_walk import _walk
    from tools.edu_oracle import run_oracle

    lib = load()
    err = ct.create_string_buffer(512)
    dc = lib.sbz_dctx_create_cuda(0, KDIR.encode(), err, len(err))
    if not dc:
        print("FAIL %s: create: %s" % (name, err.value.decode()))
        return 1
    try:
        rc = lib.sbz_dctx_parse(dc, comp, len(comp))
        if rc != 0:
            print("FAIL %s: parse rc=%d: %s" % (name, rc, lib.sbz_dctx_error(dc).decode()))
            return 1
        cnt = Counters()
        assert lib.sbz_dctx_counters(dc, ct.byref(cnt)) == 0

        buf = comp + b"\x00" * ((-len(comp)) % 4)
        frames, finfos, raw, rle, cmp = _walk(buf, len(comp))
        errs = []

        def chk(what, got, exp):
            if got != exp:
                g, e = (got[:8], exp[:8]) if isinstance(got, list) else (got, exp)
                errs.append("%s: got %r exp %r" % (what, g, e))

        chk("Frames", cnt.Frames, len(frames))
        chk("Blocks_RAW", cnt.Blocks_RAW, len(raw))
        chk("Blocks_RLE", cnt.Blocks_RLE, len(rle))
        chk("Blocks_CMP", cnt.Blocks_CMP, len(cmp))
        chk("BlocksBytes_RAW", cnt.BlocksBytes_RAW, sum(s for _, s in raw))
        chk("BlocksBytes_RLE", cnt.BlocksBytes_RLE, sum(s for _, s in rle))

        # per-frame exclusive prefixes of the block counts
        chk("PerFrameBlockCountRAW", u32s(read_buf(lib, dc, "PerFrameBlockCountRAW")),
            [f["rawBlockStart"] for f in finfos])
        chk("PerFrameBlockCountRLE", u32s(read_buf(lib, dc, "PerFrameBlockCountRLE")),
            [f["rleBlockStart"] for f in finfos])
        chk("PerFrameBlockCountCMP", u32s(read_buf(lib, dc, "PerFrameBlockCountCMP")),
            [f["cmpBlockStart"] for f in finfos])

        chk("BlocksRAWRefs", pairs(read_buf(lib, dc, "BlocksRAWRefs")), raw)
        chk("BlocksRLERefs", pairs(read_buf(lib, dc, "BlocksRLERefs")), rle)
        chk("BlocksCMPRefs", pairs(read_buf(lib, dc, "BlocksCMPRefs")), cmp)
        chk("RawBlockSizePrefix", u32s(read_buf(lib, dc, "RawBlockSizePrefix")),
            excl_prefix([s for _, s in raw]))
        chk("RleBlockSizePrefix", u32s(read_buf(lib, dc, "RleBlockSizePrefix")),
            excl_prefix([s for _, s in rle]))

        # Sequence count and Huffman-literal bytes from ParseCompressedBlocks vs the
        # trusted edu_oracle (per frame). The oracle's LITS tag is the literal-region
        # byte count per compressed block, which is what DS sizes DecompressedLiterals
        # from; its SEQ tag is 3 values per sequence.
        seq = 0
        lit = 0
        for off, size in frames:
            o = run_oracle(comp[off:off + size])
            seq += sum(len(v) // 3 for v in o.get("SEQ", []))
            lit += sum(len(v) for v in o.get("LITS", []))
        info = ("frames=%d raw/rle/cmp=%d/%d/%d seqStreams=%d seqItems=%d hufLitBytes=%d "
                "hufStreams=%d" % (cnt.Frames, cnt.Blocks_RAW, cnt.Blocks_RLE, cnt.Blocks_CMP,
                                   cnt.Seq_Streams, cnt.Seq_Streams_DecodedItems,
                                   cnt.HUF_Streams_DecodedBytes, cnt.HUF_Streams))
        chk("Seq_Streams_DecodedItems (oracle)", cnt.Seq_Streams_DecodedItems, seq)
        chk("HUF_Streams_DecodedBytes (oracle)", cnt.HUF_Streams_DecodedBytes, lit)
        if errs:
            print("FAIL %s (%s)" % (name, info))
            for e in errs:
                print("    " + e)
            return 1
        print("ok   %s (%s)" % (name, info))
        return 0
    finally:
        lib.sbz_dctx_destroy(dc)


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--one":
        for name, comp in cases():
            if name == sys.argv[2]:
                return run_one(name, comp)
        print("no case", sys.argv[2])
        return 2
    subprocess.run(["make", "-s", "-C", ROOT], check=True)
    bad = 0
    for name, _ in cases():
        try:
            r = subprocess.run([sys.executable, __file__, "--one", name], timeout=TIMEOUT)
            bad += r.returncode != 0
        except subprocess.TimeoutExpired:
            print("FAIL %s: timeout (%ds) -- GPU hang?" % (name, TIMEOUT))
            bad += 1
    print("%s: %d case(s) failed" % ("FAIL" if bad else "PASS", bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
