#!/usr/bin/env python3
"""P4 step 2b: zstd calls without a host wait inside, on a reused context.

The first zstd call of a context runs exactly (counters read back after each
stage) and learns literal/sequence capacities; later calls run on host bounds and
those capacities and are checked once by sbz_dctx_wait, which replays the call
exactly if the capacities overflowed. Checked here, every output byte-exact:

  bitshuffle-zstd, one context:
    1 small call (exact), 2 more small calls (not exact, no replay),
    1 bigger and denser call (overflows the capacities: one replay),
    1 more bigger call (not exact: the capacities grew), then small again;
  bitshuffle-zstd async: two contexts alternated from one thread with
    sbz_dctx_bshuf_decompress_async + sbz_dctx_wait, 3 rounds;
  plain zstd frames (sbz_dctx_decompress_frames), one context, frames of growing
    size and level vs python zstandard;
  SLABITZ_CHECK_SYNC=1 forces the exact path on a warm context.
"""
import ctypes as ct
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests"))
from run_bshuf_chain import load, read_buf, make_h5, Span, KDIR  # noqa: E402


def stats(lib, dc):
    v = [ct.c_uint32() for _ in range(4)]
    lib.sbz_dctx_zstd_stats(dc, *[ct.byref(x) for x in v])
    return tuple(x.value for x in v)


def bshuf_call(lib, dc, raws, e, async_=False):
    src = b"".join(raws)
    offs = [0]
    for r in raws:
        offs.append(offs[-1] + len(r))
    spans = (Span * len(raws))(*[Span(offs[i], len(r)) for i, r in enumerate(raws)])
    if async_:
        return lib.sbz_dctx_bshuf_decompress_async(dc, src, len(src), spans, len(raws), e), \
            (src, spans)
    return lib.sbz_dctx_bshuf_decompress(dc, src, len(src), spans, len(raws), e), None


def main():
    import numpy as np
    import zstandard as zstd
    lib = load()
    lib.sbz_dctx_zstd_stats.argtypes = [ct.c_void_p] + [ct.POINTER(ct.c_uint32)] * 4
    lib.sbz_dctx_bshuf_decompress_async.argtypes = lib.sbz_dctx_bshuf_decompress.argtypes
    lib.sbz_dctx_wait.argtypes = [ct.c_void_p]
    lib.sbz_dctx_decompress_frames.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_size_t,
                                              ct.POINTER(Span), ct.c_uint32]
    bad = 0
    compared = 0

    def check(name, ok, detail=""):
        nonlocal bad
        bad += not ok
        print("%s %s%s" % ("ok  " if ok else "FAIL", name, (": " + detail) if detail else ""))

    with tempfile.TemporaryDirectory(dir=os.path.join(ROOT, "build")) as td:
        # small sparse chunks and bigger, denser ones (more literals and sequences)
        small = make_h5([("s%d" % i, "uint16", 200000, 0, 0, 101 + i) for i in range(3)], td)
        os.makedirs(os.path.join(td, "b"))
        big = make_h5([("b%d" % i, "uint16", 2000000, 0, 0, 201 + i) for i in range(4)],
                      os.path.join(td, "b"))

        def inputs(entries):
            raws = [x[2][0] for x in entries]
            return raws, b"".join(x[3] for x in entries)

        sraw, swant = inputs(small)
        braw, bwant = inputs(big)
        err = ct.create_string_buffer(512)
        dc = lib.sbz_dctx_create_cuda(0, KDIR.encode(), err, len(err))
        if not dc:
            print("FAIL create: %s" % err.value.decode())
            return 1
        try:
            plan = [("small 1 (exact)", sraw, swant, (1, 0)),
                    ("small 2", sraw, swant, (1, 0)),
                    ("small 3", sraw, swant, (1, 0)),
                    ("big 1 (overflow, replay)", braw, bwant, (2, 1)),
                    ("big 2", braw, bwant, (2, 1)),
                    ("small 4", sraw, swant, (2, 1))]
            for name, raws, want, (ne, nr) in plan:
                rc, _ = bshuf_call(lib, dc, raws, 2)
                got = read_buf(lib, dc, "BshufOutput")[:len(want)]
                st = stats(lib, dc)
                check("bshuf %s" % name, rc == 0 and got == want and st[:2] == (ne, nr),
                      "rc=%d, %s, exact calls %d (want %d), replays %d (want %d), caps lit %d "
                      "seq %d" % (rc, "match" if got == want else "DIFFER", st[0], ne, st[1],
                                  nr, st[2], st[3]))
                compared += len(want)
            os.environ["SLABITZ_CHECK_SYNC"] = "1"
            rc, _ = bshuf_call(lib, dc, sraw, 2)
            os.environ.pop("SLABITZ_CHECK_SYNC")
            got = read_buf(lib, dc, "BshufOutput")[:len(swant)]
            st = stats(lib, dc)
            check("bshuf SLABITZ_CHECK_SYNC=1 on a warm context", rc == 0 and got == swant
                  and st[0] == 3, "exact calls %d (want 3)" % st[0])
            compared += len(swant)
        finally:
            lib.sbz_dctx_destroy(dc)

        # async: two contexts alternated from one thread
        dcs = [lib.sbz_dctx_create_cuda(0, KDIR.encode(), err, len(err)) for _ in range(2)]
        try:
            jobs = [(sraw, swant), (braw, bwant)]
            keep = [None, None]
            for rnd in range(3):
                for k in (0, 1):
                    rc, keep[k] = bshuf_call(lib, dcs[k], jobs[k][0], 2, async_=True)
                    check("async round %d ctx %d queued" % (rnd, k), rc == 0, "rc=%d" % rc)
                for k in (0, 1):
                    rc = lib.sbz_dctx_wait(dcs[k])
                    want = jobs[k][1]
                    got = read_buf(lib, dcs[k], "BshufOutput")[:len(want)]
                    check("async round %d ctx %d" % (rnd, k), rc == 0 and got == want,
                          "rc=%d, %s, stats %s" % (rc, "match" if got == want else "DIFFER",
                                                   stats(lib, dcs[k])))
                    compared += len(want)
        finally:
            for d in dcs:
                lib.sbz_dctx_destroy(d)

        # plain zstd frames, one context, growing size and level
        rng = np.random.default_rng(5)
        dc = lib.sbz_dctx_create_cuda(0, KDIR.encode(), err, len(err))
        try:
            for i, (n, lvl) in enumerate([(50000, 3), (50000, 3), (400000, 9), (1500000, 1),
                                          (1500000, 19), (60000, 3)]):
                data = (rng.poisson(2 + 3 * i, n).astype(np.uint8)).tobytes()
                frames = [zstd.ZstdCompressor(level=lvl).compress(data[j:j + n // 3])
                          for j in range(0, n, n // 3)]
                want = b"".join(data[j:j + n // 3] for j in range(0, n, n // 3))
                src = b"".join(frames)
                offs = [0]
                for f in frames:
                    offs.append(offs[-1] + len(f))
                spans = (Span * len(frames))(*[Span(offs[k], len(f))
                                               for k, f in enumerate(frames)])
                rc = lib.sbz_dctx_decompress_frames(dc, src, len(src), spans, len(frames))
                got = read_buf(lib, dc, "UnCompressedFramesData")[:len(want)]
                check("frames %d (%d B, level %d)" % (i, n, lvl), rc == 0 and got == want,
                      "rc=%d, %s, stats %s" % (rc, "match" if got == want else "DIFFER",
                                               stats(lib, dc)))
                compared += len(want)
        finally:
            lib.sbz_dctx_destroy(dc)

    ok = bad == 0 and compared > 0
    print("%s: %d B compared, %d failed" % ("PASS" if ok else "FAIL", compared, bad))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
