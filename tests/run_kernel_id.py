#!/usr/bin/env python3
"""The library refuses a kernel directory built from other kernel sources.

tools/gen_kernel_table.py puts a build ID (a hash of the kernel sources and the
launch table) into src/sbz_kernels_gen.h as SBZ_KERNELS_ID and into
build/sbz_kernels.id; build_kubins.py and build_fatbins.py copy the file into
their output. sbz_dctx_create_cuda must succeed for a directory with the same ID
and fail, with a message, for a directory with another ID or none (a stale
kernel set once loaded silently and crashed on a changed argument list).
"""
import ctypes as ct
import os
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = os.path.join(ROOT, "build", "libslabitz.so")
BUILD = os.path.join(ROOT, "build")


def create(lib, kdir):
    err = ct.create_string_buffer(512)
    dc = lib.sbz_dctx_create_cuda(0, kdir.encode(), err, len(err))
    if dc:
        lib.sbz_dctx_destroy(dc)
    return bool(dc), err.value.decode()


def main():
    lib = ct.CDLL(LIB)
    lib.sbz_dctx_create_cuda.restype = ct.c_void_p
    lib.sbz_dctx_create_cuda.argtypes = [ct.c_int, ct.c_char_p, ct.c_char_p, ct.c_size_t]
    lib.sbz_dctx_destroy.argtypes = [ct.c_void_p]

    hdr = open(os.path.join(ROOT, "src", "sbz_kernels_gen.h")).read()
    m = re.search(r'#define SBZ_KERNELS_ID "([0-9a-f]{16})"', hdr)
    want = m.group(1) if m else None
    have = open(os.path.join(BUILD, "sbz_kernels.id")).read().strip()
    bad = 0
    checks = 0

    def check(name, ok, detail=""):
        nonlocal bad, checks
        checks += 1
        bad += not ok
        print("%s %s%s" % ("ok  " if ok else "FAIL", name, (": " + detail) if detail else ""))

    check("header id == build/sbz_kernels.id", want is not None and want == have,
          "%s vs %s" % (want, have))

    ok, msg = create(lib, BUILD)
    check("build/ (matching id) loads", ok, msg)

    for sub in ("sm_35", "kernels"):
        d = os.path.join(BUILD, sub)
        if not os.path.isdir(d):
            print("skip build/%s (not built)" % sub)
            continue
        dirid = open(os.path.join(d, "sbz_kernels.id")).read().strip() \
            if os.path.exists(os.path.join(d, "sbz_kernels.id")) else None
        ok, msg = create(lib, d)
        # loads exactly when its id matches (an older set must be refused)
        check("build/%s (id %s) %s" % (sub, dirid, "loads" if dirid == want else "refused"),
              ok == (dirid == want), msg)

    t = os.path.join(BUILD, "_kernel_id_test")
    shutil.rmtree(t, ignore_errors=True)
    os.makedirs(t)
    try:
        ok, msg = create(lib, t)
        check("dir without sbz_kernels.id refused", not ok and "sbz_kernels.id" in msg, msg)
        with open(os.path.join(t, "sbz_kernels.id"), "w") as f:
            f.write("0123456789abcdef\n")
        ok, msg = create(lib, t)
        check("dir with another id refused", not ok and "0123456789abcdef" in msg
              and want in msg, msg)
    finally:
        shutil.rmtree(t, ignore_errors=True)

    print("%s: %d checks, %d failed" % ("PASS" if checks and not bad else "FAIL", checks, bad))
    return 1 if bad or not checks else 0


if __name__ == "__main__":
    sys.exit(main())
