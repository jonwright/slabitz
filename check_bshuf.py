#!/usr/bin/env python3
"""crunch / portable-cards check for sbz_dctx_bshuf_decompress.

No numpy, hdf5plugin, cupy, pytest or zstandard: it only needs the built
libslabitz.so, the per-kernel cubins, and the baked reference data written by
tools/gen_crunch_data.py. It decodes each reference chunk on the GPU and compares
BshufOutput byte-for-byte with the expected bytes, printing a per-case result and
a GB/s rate, then reports the device compute capability so you can confirm you
are on the target card.

Usage:
  python3 check_bshuf.py --lib build/libslabitz.so --kernels kernels --data data
  (the bundled layout is: slabitz_crunch/{libslabitz.so, kernels/, data/},
   so `python3 check_bshuf.py` with no arguments uses ./libslabitz.so etc.)
"""
import argparse
import ctypes as ct
import glob
import os
import sys
import time


def query_device():
    """Print the first CUDA device and its compute capability (via libcuda)."""
    try:
        cu = ct.CDLL(os.environ.get("CUDA_DRIVER", "libcuda.so.1"))
    except OSError as ex:
        print("  warning: cannot load libcuda: %s" % ex)
        return
    i = ct.c_int()
    if cu.cuInit(0) != 0:
        return
    dev = ct.c_int()
    if cu.cuDeviceGet(ct.byref(dev), 0) != 0:
        return
    maj, mnr = ct.c_int(), ct.c_int()
    cu.cuDeviceGetAttribute(ct.byref(maj), 75, dev)   # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
    cu.cuDeviceGetAttribute(ct.byref(mnr), 76, dev)   # ..._MINOR
    print("  device %d: compute capability %d.%d" % (0, maj.value, mnr.value))
    if maj.value < 3:
        print("  ERROR: compute capability %d.%d is below the sm_35 floor." % (maj.value, mnr.value))
        sys.exit(1)


class Span(ct.Structure):
    _fields_ = [("offs", ct.c_uint64), ("size", ct.c_uint64)]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lib", default=os.path.join(here, "libslabitz.so"))
    ap.add_argument("--kernels", default=os.path.join(here, "build"),
                    help="dir holding the per-kernel cubins (build/<sub>/<cubin>)")
    ap.add_argument("--data", default=os.path.join(here, "data"))
    ap.add_argument("--case", nargs="*", default=[], help="only these case names")
    ap.add_argument("--warmup", type=int, default=1, help="warm-up decode calls")
    a = ap.parse_args()

    query_device()

    for so in (a.lib,):
        if not os.path.exists(so):
            print("ERROR: %s not found" % so)
            return 1
    lib = ct.CDLL(a.lib)
    lib.sbz_dctx_create_cuda.restype = ct.c_void_p
    lib.sbz_dctx_create_cuda.argtypes = [ct.c_int, ct.c_char_p, ct.c_char_p, ct.c_size_t]
    lib.sbz_dctx_destroy.argtypes = [ct.c_void_p]
    lib.sbz_dctx_error.restype = ct.c_char_p
    lib.sbz_dctx_error.argtypes = [ct.c_void_p]
    lib.sbz_dctx_bshuf_decompress.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_size_t,
                                             ct.POINTER(Span), ct.c_uint32, ct.c_uint32]
    lib.sbz_dctx_read_buffer.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_void_p,
                                        ct.POINTER(ct.c_size_t)]

    names = sorted(os.path.basename(p)[: -len(".chunk.bin")]
                   for p in glob.glob(os.path.join(a.data, "*.chunk.bin")))
    if a.case:
        names = [n for n in names if n in a.case]
    if not names:
        print("ERROR: no cases (run tools/gen_crunch_data.py --data %s)" % a.data)
        return 1

    err = ct.create_string_buffer(512)

    def readbuf(dc, name):
        n = ct.c_size_t(0)
        lib.sbz_dctx_read_buffer(dc, name.encode(), None, ct.byref(n))
        buf = ct.create_string_buffer(max(n.value, 1))
        lib.sbz_dctx_read_buffer(dc, name.encode(), buf, ct.byref(n))
        return buf.raw[:n.value]

    bad = 0
    for name in names:
        chunk = open(os.path.join(a.data, name + ".chunk.bin"), "rb").read()
        want = open(os.path.join(a.data, name + ".expected.bin"), "rb").read()
        elem = int(open(os.path.join(a.data, name + ".elem")).read())
        span = (Span * 1)(Span(0, len(chunk)))
        # one fresh context per case: the device chain is not safe to reuse across
        # independent sbz_dctx_bshuf_decompress calls (only one call per context).
        dc = lib.sbz_dctx_create_cuda(0, a.kernels.encode(), err, len(err))
        if not dc:
            print("ERROR: sbz_dctx_create_cuda failed: %s" % err.value.decode())
            return 1

        def call(dc=dc):
            return lib.sbz_dctx_bshuf_decompress(dc, chunk, len(chunk), span, 1, elem)

        rc = call()
        if rc != 0:
            print("FAIL %-16s rc=%d: %s" % (name, rc, lib.sbz_dctx_error(dc).decode()))
            bad += 1
            lib.sbz_dctx_destroy(dc)
            continue
        got = readbuf(dc, "BshufOutput")
        if len(got) < len(want) or got[:len(want)] != want:
            i = next((k for k, (x, y) in enumerate(zip(got, want)) if x != y), len(want))
            print("FAIL %-16s mismatch at byte %d (%d vs %d bytes)" % (name, i, len(got), len(want)))
            bad += 1
            lib.sbz_dctx_destroy(dc)
            continue
        for _ in range(max(0, a.warmup - 1)):
            call()
        t0 = time.monotonic()
        rc = call()
        dt = time.monotonic() - t0
        if rc != 0:
            print("FAIL %-16s timed rc=%d" % (name, rc))
            bad += 1
        else:
            print("ok   %-16s e=%d decoded %d B in %.4f ms (%.1f GB/s)"
                  % (name, elem, len(want), dt * 1e3, len(want) / max(dt, 1e-9) / 1e9))
        lib.sbz_dctx_destroy(dc)

    print("%s: %d case(s) failed" % ("FAIL" if bad else "PASS", bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
