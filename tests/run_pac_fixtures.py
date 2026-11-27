#!/usr/bin/env python3
"""Decode the committed pac fixtures (T02) on the GPU and on the CPU.

For every case in data/pac/: the frames' chunks are decoded with
sbz_dctx_bshuf_decompress / sbz_dctx_bslz4_decompress (sync and async) and the
SHA-256 of each decoded frame is compared with the hash in <case>.json; the same
chunks are decoded on the CPU with hdf5plugin (bitshuffle) and compared with the
same hash, so a fixture cannot carry a wrong hash.  Compared byte counts are
printed and must be nonzero.

--stats additionally walks every zstd frame of every zstd case with
tools/edu_oracle.py and counts the sequences by the fast path the decoder takes:
off == 1 (a fill), off == P with len <= off (a plane copy) and off == P with
len > off (a merged match), where P is the block's plane byte count.  It prints
the counts per case and fails if a zstd case that has any sequence is missing
any of the three kinds (the RLE empty-frames case has no sequences by design).

Usage:  python3 tests/run_pac_fixtures.py [--stats]
"""
import ctypes as ct
import hashlib
import json
import os
import struct
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)
LIB = os.path.join(ROOT, "build", "libslabitz.so")
KDIR = os.environ.get("SLABITZ_KERNEL_DIR", os.path.join(ROOT, "build"))
DATA = os.path.join(ROOT, "data", "pac")


class Span(ct.Structure):
    _fields_ = [("offs", ct.c_uint64), ("size", ct.c_uint64)]


def load():
    lib = ct.CDLL(LIB)
    lib.sbz_dctx_create_cuda.restype = ct.c_void_p
    lib.sbz_dctx_create_cuda.argtypes = [ct.c_int, ct.c_char_p, ct.c_char_p, ct.c_size_t]
    lib.sbz_dctx_destroy.argtypes = [ct.c_void_p]
    lib.sbz_dctx_error.restype = ct.c_char_p
    lib.sbz_dctx_error.argtypes = [ct.c_void_p]
    for name in ("sbz_dctx_bshuf_decompress", "sbz_dctx_bslz4_decompress",
                 "sbz_dctx_bshuf_decompress_async", "sbz_dctx_bslz4_decompress_async"):
        getattr(lib, name).argtypes = [ct.c_void_p, ct.c_char_p, ct.c_size_t,
                                       ct.POINTER(Span), ct.c_uint32, ct.c_uint32]
    lib.sbz_dctx_read_buffer.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_void_p,
                                         ct.POINTER(ct.c_size_t)]
    lib.sbz_dctx_wait.argtypes = [ct.c_void_p]
    return lib


def new_ctx(lib):
    err = ct.create_string_buffer(512)
    dc = lib.sbz_dctx_create_cuda(0, KDIR.encode(), err, len(err))
    if not dc:
        raise SystemExit("FAIL: cannot create the decoder context: %s" % err.value.decode())
    return dc


def read_buf(lib, dc, name):
    n = ct.c_size_t(0)
    assert lib.sbz_dctx_read_buffer(dc, name.encode(), None, ct.byref(n)) == 0
    b = ct.create_string_buffer(max(n.value, 1))
    assert lib.sbz_dctx_read_buffer(dc, name.encode(), b, ct.byref(n)) == 0
    return b.raw[:n.value]


def frames_spans(index):
    spans = (Span * len(index))(*[Span(e["offset"], e["length"]) for e in index])
    return spans


def gpu_decode(lib, case, src, index, async_):
    """Decode all frames of `case` in one call (sync or async); return the decoded bytes."""
    e = case["elem_size"]
    fn = ("sbz_dctx_bslz4_decompress" if case["codec"] == "lz4" else "sbz_dctx_bshuf_decompress")
    if async_:
        fn += "_async"
    dc = new_ctx(lib)
    try:
        spans = frames_spans(index)
        rc = getattr(lib, fn)(dc, src, len(src), spans, len(index), e)
        if async_:
            rc = rc or lib.sbz_dctx_wait(dc)
        if rc != 0:
            raise SystemExit("FAIL %s %s: rc=%d: %s"
                             % (case["case"], "async" if async_ else "sync", rc,
                                lib.sbz_dctx_error(dc).decode()))
        return read_buf(lib, dc, "BshufOutput")
    finally:
        lib.sbz_dctx_destroy(dc)


def cpu_decode_frame(chunk, case, decoded_bytes):
    """Decode one frame's chunk with hdf5plugin (bitshuffle); return the flat bytes."""
    import h5py
    import hdf5plugin  # noqa: F401
    e = case["elem_size"]
    dt = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[e]
    nelems = case["block_bytes"] // e
    nel = decoded_bytes // e
    with h5py.File("cpu_decode.h5", "w", driver="core", backing_store=False) as f:
        ds = f.create_dataset("d", shape=(nel,), dtype=dt, chunks=(nel,),
                              **hdf5plugin.Bitshuffle(nelems=nelems, cname=case["codec"], clevel=2))
        ds.id.write_direct_chunk((0,), chunk)
        b = np.ascontiguousarray(ds[()]).tobytes()
    return b


# ------------------------------------------------------------------ decoding
def check_case(lib, case, chunk_bin, index, failures):
    ok = True

    def check_hash(out, label):
        off = 0
        for i, frm in enumerate(case["frames"]):
            nb = frm["decoded_bytes"]
            got = out[off:off + nb]
            if hashlib.sha256(got).hexdigest() != frm["sha256"]:
                failures.append("%s frame %d: %s hash differs" % (case["case"], i, label))
                return False
            off += nb
        return True

    ok = check_hash(gpu_decode(lib, case, chunk_bin, index, async_=False),
                    "GPU(sync)") and \
        check_hash(gpu_decode(lib, case, chunk_bin, index, async_=True),
                   "GPU(async)")

    cpu_compared = 0
    for i, frm in enumerate(case["frames"]):
        ch = chunk_bin[index[i]["offset"]:index[i]["offset"] + index[i]["length"]]
        cb = cpu_decode_frame(ch, case, frm["decoded_bytes"])
        if hashlib.sha256(cb).hexdigest() != frm["sha256"]:
            failures.append("%s frame %d: CPU hash differs" % (case["case"], i))
            ok = False
        cpu_compared += len(cb)

    total = sum(f["decoded_bytes"] for f in case["frames"])
    print("%-26s %s: GPU(sync) %d B, GPU(async) %d B, CPU %d B compared"
          % (case["case"], "ok  " if ok else "FAIL", total, total, cpu_compared))
    if total == 0 or cpu_compared == 0:
        failures.append("%s: compared zero bytes" % case["case"])
        return False
    return ok


# ---------------------------------------------------------------------- stats
def parse_chunk_blocks(chunk, e):
    """Split a filter-32008 chunk into its block frames and the raw tail."""
    total = int.from_bytes(chunk[0:8], "big")
    block_bytes = int.from_bytes(chunk[8:12], "big")
    S = total // e
    B = block_bytes // e
    nfull = S // B
    last = S - nfull * B
    last2 = last - last % 8
    nf = nfull + (1 if last2 else 0)
    pos = 12
    blocks = []
    for _ in range(nf):
        cs = int.from_bytes(chunk[pos:pos + 4], "big")
        blocks.append(chunk[pos + 4:pos + 4 + cs])
        pos += 4 + cs
    return blocks


def case_match_kinds(case, chunk_bin, index):
    """Count off==1, off==P-with-len<=off and off==P-with-len>off per zstd case."""
    from edu_oracle import run_oracle
    e = case["elem_size"]
    c1 = cP_le = cP_gt = 0
    nseq = 0
    for i in range(len(index)):
        ch = chunk_bin[index[i]["offset"]:index[i]["offset"] + index[i]["length"]]
        for blk in parse_chunk_blocks(ch, e):
            try:
                d = run_oracle(blk)
            except Exception:
                continue
            olen = len(d["_output"])
            if olen == 0:
                continue
            P = olen // (8 * e)
            triples = []
            for ent in d.get("SEQ", []):
                for k in range(0, len(ent), 3):
                    triples.append((ent[k], ent[k + 1], ent[k + 2]))
            offs = [o[0] for o in d.get("OFFS", [])]
            for (ll, ml, off), res in zip(triples, offs):
                nseq += 1
                if res == 1:
                    c1 += 1
                elif res == P and ml <= res:
                    cP_le += 1
                elif res == P and ml > res:
                    cP_gt += 1
    return c1, cP_le, cP_gt, nseq


def main():
    subprocess.run(["make", "-s", "-C", ROOT], check=True)
    lib = load()
    failures = []
    stats = "--stats" in sys.argv[1:]
    seq_cases = []

    names = sorted(f for f in os.listdir(DATA) if f.endswith(".json"))
    for name in names:
        case = json.load(open(os.path.join(DATA, name)))
        with open(os.path.join(DATA, case["case"] + ".chunk.bin"), "rb") as fh:
            chunk_bin = fh.read()
        index = json.load(open(os.path.join(DATA, case["case"] + ".index")))
        ok = check_case(lib, case, chunk_bin, index, failures)
        if stats and case["codec"] == "zstd":
            c1, cP_le, cP_gt, nseq = case_match_kinds(case, chunk_bin, index)
            seq_cases.append((case["case"], c1, cP_le, cP_gt, nseq))

    if stats:
        print("\n-- SEQ fast-path kinds per zstd case (P = plane bytes) --")
        for cs, c1, cP_le, cP_gt, nseq in seq_cases:
            print("%-26s off==1 %7d  off==P len<=off %7d  off==P len>off %7d  (sequences %d)"
                  % (cs, c1, cP_le, cP_gt, nseq))
        for cs, c1, cP_le, cP_gt, nseq in seq_cases:
            if nseq == 0:
                print("  %s: RLE-only (no sequences); the three fast paths do not apply"
                      % cs)
                continue
            if not (c1 > 0 and cP_le > 0 and cP_gt > 0):
                failures.append("%s: SEQ fast-path counts missing one of the three kinds "
                                "(off1=%d offP<=off=%d offP>off=%d)" % (cs, c1, cP_le, cP_gt))

    if failures:
        print("FAILURES:")
        for f in failures:
            print("  " + f)
        return 1
    print("all pac fixtures decoded and matched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
