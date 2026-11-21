#!/usr/bin/env python3
"""P2: Stage 2 of the C device chain (full decode) on the GPU.

Runs sbz_dctx_decompress() (Stage 0+1+2: parse -> FSE/Huffman -> literals -> sequences
-> dest offsets -> execute) entirely from C on device-resident buffers and compares the
decompressed output (the debug hook on "UnCompressedFramesData") against libzstd.

Usage:  python3 tests/run_chain_stage2.py      (builds with `make` first)
Each case runs in a subprocess under a timeout so a GPU hang fails fast.
"""
import ctypes as ct
import os
import random
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

LIB = os.path.join(ROOT, "build", "libslabitz.so")
KDIR = os.environ.get("SLABITZ_KERNEL_DIR", os.path.join(ROOT, "build"))  # e.g. build/sm_35
TIMEOUT = 120


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
    lib.sbz_dctx_decompress.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_size_t]
    lib.sbz_dctx_counters.argtypes = [ct.c_void_p, ct.POINTER(Counters)]
    lib.sbz_dctx_read_buffer.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_void_p,
                                        ct.POINTER(ct.c_size_t)]
    return lib


def read_buf(lib, dc, name):
    n = ct.c_size_t(0)
    rc = lib.sbz_dctx_read_buffer(dc, name.encode(), None, ct.byref(n))
    assert rc == 0, lib.sbz_dctx_error(dc)
    b = ct.create_string_buffer(max(n.value, 1))
    rc = lib.sbz_dctx_read_buffer(dc, name.encode(), b, ct.byref(n))
    assert rc == 0, lib.sbz_dctx_error(dc)
    return b.raw[:n.value]


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
    yield "empty-rle", c(b"", 3)
    yield "short", c(b"hello", 3)


# ---------------------------------------------------------------- one case
def run_one(name, comp):
    import zstandard as zstd

    want = zstd.ZstdDecompressor().decompress(comp)

    lib = load()
    err = ct.create_string_buffer(512)
    dc = lib.sbz_dctx_create_cuda(0, KDIR.encode(), err, len(err))
    if not dc:
        print("FAIL %s: create: %s" % (name, err.value.decode()))
        return 1
    try:
        rc = lib.sbz_dctx_decompress(dc, comp, len(comp))
        if rc != 0:
            print("FAIL %s: decompress rc=%d: %s" % (name, rc, lib.sbz_dctx_error(dc).decode()))
            return 1
        cnt = Counters()
        assert lib.sbz_dctx_counters(dc, ct.byref(cnt)) == 0
        got = read_buf(lib, dc, "UnCompressedFramesData")[:len(want)]
        if got != want:
            i = next((k for k, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
            print("FAIL %s: output mismatch at byte %d: got len=%d want len=%d "
                  "(frames=%d uncomp=%d seqItems=%d cmpBlocks=%d)" % (
                      name, i, len(got), len(want), cnt.Frames,
                      cnt.Frames_UncompressedByteSize, cnt.Seq_Streams_DecodedItems,
                      cnt.Blocks_CMP))
            return 1
        print("ok   %s (frames=%d blocks=%d/%d/%d seqItems=%d out=%d)" % (
            name, cnt.Frames, cnt.Blocks_RAW, cnt.Blocks_RLE, cnt.Blocks_CMP,
            cnt.Seq_Streams_DecodedItems, cnt.Frames_UncompressedByteSize))
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
