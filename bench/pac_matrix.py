#!/usr/bin/env python3
"""12-case detector decode matrix: silicon, thaumatin, CeO2 x {u16, u32} x {mask=0, mask=max-1}.

Encodes one frame per case in memory using both pac-zstd (if libpac.so is available) and
stock bitshuffle-zstd (hdf5plugin level 2), tiles it N times (default 4, so 16M u32 fits
on 4 GB GPUs), verifies every case byte-exact on the GPU, and times the decode call.

Usage:
  SLABITZ_EXAMPLES=/path/to/examples python3 bench/pac_matrix.py [--frames 4] [--reps 3]
"""
import argparse
import ctypes as ct
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = os.path.join(ROOT, "build", "libslabitz.so")
KDIR = os.environ.get("SLABITZ_KERNEL_DIR", os.path.join(ROOT, "build"))


class Span(ct.Structure):
    _fields_ = [("offs", ct.c_uint64), ("size", ct.c_uint64)]


def load_slabitz():
    lib = ct.CDLL(LIB)
    lib.sbz_dctx_create_cuda.restype = ct.c_void_p
    lib.sbz_dctx_create_cuda.argtypes = [ct.c_int, ct.c_char_p, ct.c_char_p, ct.c_size_t]
    lib.sbz_dctx_destroy.argtypes = [ct.c_void_p]
    lib.sbz_dctx_error.restype = ct.c_char_p
    lib.sbz_dctx_error.argtypes = [ct.c_void_p]
    lib.sbz_dctx_bshuf_decompress.argtypes = [
        ct.c_void_p, ct.c_char_p, ct.c_size_t, ct.POINTER(Span), ct.c_uint32, ct.c_uint32
    ]
    lib.sbz_dctx_read_buffer.argtypes = [
        ct.c_void_p, ct.c_char_p, ct.c_void_p, ct.POINTER(ct.c_size_t)
    ]
    return lib


def load_pac(pac_path):
    if not pac_path or not os.path.isfile(pac_path):
        return None
    pac = ct.CDLL(pac_path)
    vp, sz, ci = ct.c_void_p, ct.c_size_t, ct.c_int
    pac.pac_create.restype = vp
    pac.pac_create.argtypes = [ci]
    pac.pac_free.argtypes = [vp]
    pac.pac_set_raw_literals.argtypes = [vp, ci]
    pac.pac_bound.restype = sz
    pac.pac_bound.argtypes = [sz, sz, sz]
    pac.pac_compress_chunk.restype = sz
    pac.pac_compress_chunk.argtypes = [vp, vp, sz, vp, sz, sz, sz]
    return pac


def encode_stock_zstd(fr, block_bytes):
    import h5py
    import hdf5plugin
    e = fr.dtype.itemsize
    with h5py.File("mem.h5", "w", driver="core", backing_store=False) as f:
        ds = f.create_dataset(
            "d",
            data=fr,
            chunks=fr.shape,
            **hdf5plugin.Bitshuffle(nelems=block_bytes // e, cname="zstd", clevel=2),
        )
        _, raw = ds.id.read_direct_chunk((0,) * fr.ndim)
        return bytes(raw)


def encode_pac_zstd(pac, enc, fr, block_bytes):
    import numpy as np
    e = fr.dtype.itemsize
    fb = fr.nbytes
    cap = pac.pac_bound(fb, e, block_bytes)
    buf = np.empty(cap, dtype=np.uint8)
    m = pac.pac_compress_chunk(enc, buf.ctypes.data, cap, fr.ctypes.data, fb, e, block_bytes)
    if m <= 0:
        raise RuntimeError("pac_compress_chunk failed (rc=%d)" % m)
    return bytes(buf[:m].tobytes())


def read_buf(lib, dc, name):
    n = ct.c_size_t(0)
    assert lib.sbz_dctx_read_buffer(dc, name.encode(), None, ct.byref(n)) == 0
    b = ct.create_string_buffer(max(n.value, 1))
    assert lib.sbz_dctx_read_buffer(dc, name.encode(), b, ct.byref(n)) == 0
    return b.raw[:n.value]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--block-kb", type=int, default=248)
    ap.add_argument("--codec", choices=("both", "pac", "stock"), default="both")
    ap.add_argument("--raw", type=int, default=0, metavar="PCT")
    ap.add_argument(
        "--pac-lib",
        default=os.environ.get(
            "PAC_LIB", os.path.join(os.path.dirname(ROOT), "pac-zstd", "build", "libpac.so")
        ),
    )
    ap.add_argument("--only", default="", help="substring filter on case name")
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()

    ex = os.environ.get("SLABITZ_EXAMPLES", "")
    needed = [
        "silicon_u16_zstd2_256k.h5",
        "ceo2_u32_zstd2_256k.h5",
        "thaumatin_9_1_000001.h5",
        "mask.h5",
    ]
    if not ex or not all(os.path.isfile(os.path.join(ex, fn)) for fn in needed):
        print("skip pac_matrix: SLABITZ_EXAMPLES missing required files (%s)" % ex)
        return 0

    import h5py
    import hdf5plugin  # noqa: F401
    import numpy as np

    pac = None
    enc = None
    if a.codec in ("both", "pac"):
        pac = load_pac(a.pac_lib)
        if pac is None:
            print("note: libpac.so not found at %s; skipping pac cases" % a.pac_lib)
            if a.codec == "pac":
                return 0
        else:
            enc = pac.pac_create(2)  # PAC_ZSTD
            pac.pac_set_raw_literals(enc, a.raw)

    with h5py.File(os.path.join(ex, "mask.h5"), "r") as f:
        mask4m = f["mask"][:].astype(bool)

    frames = {}
    with h5py.File(os.path.join(ex, "silicon_u16_zstd2_256k.h5"), "r") as f:
        frames["silicon"] = (f["entry_0000/ESRF-ID11/eiger/data"][0], mask4m)
    with h5py.File(os.path.join(ex, "ceo2_u32_zstd2_256k.h5"), "r") as f:
        frames["ceo2"] = (f["entry_0000/ESRF-ID11/eiger/data"][0], mask4m)
    with h5py.File(os.path.join(ex, "thaumatin_9_1_000001.h5"), "r") as f:
        th = f["data"][0]
        frames["thaumatin"] = (th, th == 65535)

    lib = load_slabitz()
    err = ct.create_string_buffer(512)
    dc = lib.sbz_dctx_create_cuda(0, KDIR.encode(), err, len(err))
    if not dc:
        raise SystemExit("create: %s" % err.value.decode())

    block_bytes = a.block_kb * 1024
    codecs = []
    if a.codec in ("both", "pac") and pac is not None:
        codecs.append("pac")
    if a.codec in ("both", "stock"):
        codecs.append("stock")

    total_compared = 0
    print(
        "%-24s %-6s %7s %8s %7s %4s" % ("case", "codec", "ratio", "ms/fr", "GB/s", "ok"),
        flush=True,
    )
    try:
        for name in ("silicon", "thaumatin", "ceo2"):
            raw_fr, msk = frames[name]
            for dt_name, dt in (("u16", np.uint16), ("u32", np.uint32)):
                e = np.dtype(dt).itemsize
                dmax = int(np.iinfo(dt).max)
                for mlabel, mval in (("mask0", 0), ("maskmax-1", dmax - 1)):
                    case = "%s_%s_%s" % (name, dt_name, mlabel)
                    if a.only and a.only not in case:
                        continue
                    fr = np.clip(raw_fr.astype(np.uint64), 0, dmax).astype(dt)
                    fr[msk] = mval
                    fr = np.ascontiguousarray(fr)
                    fb = fr.nbytes
                    want = fr.tobytes() if not a.no_verify else None

                    for cname in codecs:
                        if cname == "pac":
                            ch = encode_pac_zstd(pac, enc, fr, block_bytes)
                        else:
                            ch = encode_stock_zstd(fr, block_bytes)

                        if want is not None:
                            sp1 = (Span * 1)(Span(0, len(ch)))
                            rc = lib.sbz_dctx_bshuf_decompress(dc, ch, len(ch), sp1, 1, e)
                            if rc != 0:
                                raise SystemExit(
                                    "%s %s verify failed: %s"
                                    % (case, cname, lib.sbz_dctx_error(dc).decode())
                                )
                            got = read_buf(lib, dc, "BshufOutput")
                            if len(got) < fb or got[:fb] != want:
                                raise SystemExit("%s %s byte mismatch!" % (case, cname))
                            total_compared += fb
                            ok_str = "ok"
                        else:
                            ok_str = "skip"

                        src = ch * a.frames
                        spans = (Span * a.frames)(
                            *[Span(i * len(ch), len(ch)) for i in range(a.frames)]
                        )
                        for _ in range(2):
                            rc = lib.sbz_dctx_bshuf_decompress(
                                dc, src, len(src), spans, a.frames, e
                            )
                            if rc != 0:
                                raise SystemExit(
                                    "%s %s warmup failed: %s"
                                    % (case, cname, lib.sbz_dctx_error(dc).decode())
                                )
                        t0 = time.perf_counter()
                        for _ in range(a.reps):
                            rc = lib.sbz_dctx_bshuf_decompress(
                                dc, src, len(src), spans, a.frames, e
                            )
                            if rc != 0:
                                raise SystemExit(
                                    "%s %s timed failed: %s"
                                    % (case, cname, lib.sbz_dctx_error(dc).decode())
                                )
                        dt_s = (time.perf_counter() - t0) / a.reps / a.frames
                        print(
                            "%-24s %-6s %7.2f %8.2f %7.2f %4s"
                            % (case, cname, fb / len(ch), dt_s * 1e3, fb / dt_s / 1e9, ok_str),
                            flush=True,
                        )
    finally:
        lib.sbz_dctx_destroy(dc)
        if enc is not None and pac is not None:
            pac.pac_free(enc)

    if not a.no_verify:
        assert total_compared > 0, "0 bytes compared"
    print("TOTAL compared: %d bytes" % total_compared)
    return 0


if __name__ == "__main__":
    sys.exit(main())
