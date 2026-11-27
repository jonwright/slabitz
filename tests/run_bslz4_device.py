#!/usr/bin/env python3
"""P4 step 2c: bitshuffle-LZ4 from device memory, no host pass over the bytes.

sbz_dctx_bslz4_decompress_device takes the chunks in device memory, their spans
(host), and the stated geometry (element size, bytes per chunk, block bytes); a
kernel walks the block records on the device. Checked, every output vs h5py:

  - synthetic chunks of one geometry per call: 8 kB blocks with raw tails (the
    fused path plus the partial last block), 248 kB blocks (unfused), u64;
  - the real files (first 16 frames), if readable;
  - a stated geometry that does not match the data: the wait replays the call
    on the host and the output is still right (one replay counted);
  - a corrupt chunk (a block record past the end): SBZ_ERR_CORRUPT after the
    replay, and the context works afterwards.
"""
import ctypes as ct
import os
import struct
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests"))
from run_bslz4_chain import (load, read_buf, make_h5, Span, KDIR, find_chunked,  # noqa: E402
                             SBZ_ERR_CORRUPT)


class Geom(ct.Structure):
    _fields_ = [("elemSize", ct.c_uint32), ("chunkBytes", ct.c_uint64),
                ("blockBytes", ct.c_uint32)]


def cuda():
    cu = ct.CDLL("libcuda.so.1")
    cu.cuInit(0)
    dev = ct.c_int()
    cu.cuDeviceGet(ct.byref(dev), 0)
    ctx = ct.c_void_p()
    if hasattr(cu, "cuDevicePrimaryCtxRetain"):
        rc = cu.cuDevicePrimaryCtxRetain(ct.byref(ctx), dev)
    else:  # driver 340 (CUDA 6.5)
        rc = cu.cuCtxCreate_v2(ct.byref(ctx), 0, dev)
    assert rc == 0 and cu.cuCtxSetCurrent(ctx) == 0
    cu.cuMemAlloc_v2.argtypes = [ct.POINTER(ct.c_uint64), ct.c_size_t]
    cu.cuMemcpyHtoD_v2.argtypes = [ct.c_uint64, ct.c_char_p, ct.c_size_t]
    cu.cuMemFree_v2.argtypes = [ct.c_uint64]
    return cu


def main():
    import numpy as np
    lib = load()
    lib.sbz_dctx_bslz4_decompress_device.argtypes = [
        ct.c_void_p, ct.c_uint64, ct.c_size_t, ct.POINTER(Span), ct.c_uint32,
        ct.POINTER(Geom)]
    lib.sbz_dctx_wait.argtypes = [ct.c_void_p]
    lib.sbz_dctx_zstd_stats.argtypes = [ct.c_void_p] + [ct.POINTER(ct.c_uint32)] * 4
    cu = cuda()
    err = ct.create_string_buffer(512)
    dc = lib.sbz_dctx_create_cuda(0, KDIR.encode(), err, len(err))
    if not dc:
        print("FAIL create: %s" % err.value.decode())
        return 1
    bad = 0
    compared = 0

    def replays():
        r = ct.c_uint32()
        lib.sbz_dctx_zstd_stats(dc, None, ct.byref(r), None, None)
        return r.value

    def run(name, raws, want, e, chunk_bytes, block_bytes, expect_rc=0, expect_replay=0):
        nonlocal bad, compared
        src = b"".join(raws)
        offs = [0]
        for r in raws:
            offs.append(offs[-1] + len(r))
        spans = (Span * len(raws))(*[Span(offs[i], len(r)) for i, r in enumerate(raws)])
        d = ct.c_uint64()
        assert cu.cuMemAlloc_v2(ct.byref(d), len(src)) == 0
        assert cu.cuMemcpyHtoD_v2(d.value, src, len(src)) == 0
        # From pageable memory the copy can return before its DMA lands, and the
        # context's stream is non-blocking (it does not wait for the legacy stream):
        # without this sync the decode can read the input's last bytes too early, flag
        # those chunks and replay (seen 2026-11-27: "u8 8k", replays 1, 6 of 7 runs).
        assert cu.cuCtxSynchronize() == 0
        g = Geom(e, chunk_bytes, block_bytes)
        r0 = replays()
        rc = lib.sbz_dctx_bslz4_decompress_device(dc, d.value, len(src), spans, len(raws),
                                                 ct.byref(g))
        if rc == 0:
            rc = lib.sbz_dctx_wait(dc)
        nrep = replays() - r0
        ok = rc == expect_rc and nrep == expect_replay
        detail = "rc=%d (want %d), replays %d (want %d)" % (rc, expect_rc, nrep, expect_replay)
        if rc == 0:
            got = read_buf(lib, dc, "BshufOutput")[:len(want)]
            ok = ok and got == want and len(want) > 0
            compared += len(want)
            detail += ", %d B %s" % (len(want), "match" if got == want else "DIFFER")
        else:
            detail += ": " + lib.sbz_dctx_error(dc).decode()
        cu.cuMemFree_v2(d.value)
        bad += not ok
        print("%s %s: %s" % ("ok  " if ok else "FAIL", name, detail))

    with tempfile.TemporaryDirectory(dir=os.path.join(ROOT, "build")) as td:
        cases = [("u16 8k tails", "uint16", 300003, 0),
                 ("u32 248k", "uint32", 150000, 248 * 1024 // 4),
                 ("u64 8k tails", "uint64", 50005, 0),
                 ("u8 8k", "uint8", 400000, 0)]
        for k, (name, dt, n, nel) in enumerate(cases):
            sub = os.path.join(td, "c%d" % k)
            os.makedirs(sub)
            ent = make_h5([("%s-%d" % (name.replace(" ", "_"), i), dt, n, nel, 0, 300 + 7 * k + i,
                            "repeat") for i in range(5)], sub)
            raws = [x[2][0] for x in ent]
            want = b"".join(x[3] for x in ent)
            e = np.dtype(dt).itemsize
            block = struct.unpack(">I", raws[0][8:12])[0]
            run(name, raws, want, e, n * e, block)
            if k == 0:
                # stated geometry wrong (block size): the walk flags every chunk, the wait
                # replays on the host, the output is still right
                run("u16 8k, wrong block size stated", raws, want, e, n * e, block * 2,
                    expect_replay=1)
                # a corrupt chunk: the first block record says more bytes than the chunk has
                bad_raw = bytearray(raws[2])
                bad_raw[12:16] = struct.pack(">I", len(bad_raw))
                run("u16 8k, corrupt chunk 2", raws[:2] + [bytes(bad_raw)] + raws[3:], want, e,
                    n * e, block, expect_rc=SBZ_ERR_CORRUPT, expect_replay=1)
                run("u16 8k, again after the error", raws, want, e, n * e, block)

    examples = os.environ.get("SLABITZ_EXAMPLES") or os.path.join(ROOT, "workspace", "examples")
    for name in ("silicon_lz4.h5", "ceo2_lz4.h5"):
        path = os.path.join(examples, name)
        if not os.path.exists(path):
            if os.environ.get("SLABITZ_REQUIRE_DATA") == "1":  # T02
                print("FAIL: data required but %s is missing (set SLABITZ_EXAMPLES)" % path)
                sys.exit(1)
            print("ok   %s: not readable, skipped (set SLABITZ_EXAMPLES)" % path)
            continue
        import h5py
        import hdf5plugin  # noqa: F401
        with h5py.File(path, "r") as f:
            ds = find_chunked(f)
            raws = [bytes(ds.id.read_direct_chunk((i, 0, 0))[1]) for i in range(16)]
            want = ds[:16].tobytes()
            e = ds.dtype.itemsize
            fb = ds.shape[1] * ds.shape[2] * e
        block = struct.unpack(">I", raws[0][8:12])[0]
        run(os.path.basename(path), raws, want, e, fb, block)

    lib.sbz_dctx_destroy(dc)
    ok = bad == 0 and compared > 0
    print("%s: %d B compared, %d failed" % ("PASS" if ok else "FAIL", compared, bad))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
