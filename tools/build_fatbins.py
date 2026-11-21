#!/usr/bin/env python3
"""One fatbin per kernel, each holding every supported architecture.

For each entry in tools/build_all.py's KERNEL_MAP (22 Slang kernels), bundle into a
single `.fatbin` (build/kernels/<sub>/<stem>.fatbin, one CUDA module per file):

  - SASS cubins for sm_50 .. sm_90, built by the default environment's nvcc 12.6
    over the `.cu` that slangc already wrote (build/<sub>/<ker>.cu);
  - PTX for compute_90, so a card newer than this list JITs it;
  - the sm_35 cubin from build/sm_35 (NVRTC 11.8 - the last toolkit that supports
    Kepler; run `pixi run -e kepler build-sm35` first, or pass --no-sm35).

The architectures are compiled in parallel (a process pool). Each fatbin stays one
kernel's CUDA module - never several kernels in one file (AGENTS.md: a merged module
compiles the include-guarded DS shader bodies once, under the first kernel's macros).

Usage:
  pixi run build-fatbins                                   # default archs into build/kernels
  python3 tools/build_fatbins.py --archs none --out build/kernels_ptx   # PTX only
"""
import argparse
import concurrent.futures
import os
import pathlib
import shutil
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from build_all import KERNEL_MAP  # noqa: E402

SLANGINC = str(ROOT / "third_party" / "slang" / "include")
SHIM = "slang_cuda_prevolta.h"   # pre-Volta wave semantics (no-op on sm_70+)
ARCHS = "50,52,60,61,70,75,80,86,89,90"


def compile_one(job):
    """Compile one (kernel, arch) image. job is a tuple of strings."""
    _ker, _sub, arch, cu, dst, nvcc = job
    if arch.startswith("compute_"):
        # PTX: forward-compatible JIT for cards newer than the SASS list.
        cmd = [nvcc, "-arch=" + arch, "-ptx", "-I", str(HERE), "-I", SLANGINC,
               "-include", SHIM, cu, "-o", dst]
    else:
        # SASS cubin, with exactly the flags tools/build_all.py uses (the shim is a
        # no-op from sm_70, so one .cu serves every architecture).
        cmd = [nvcc, "-arch=" + arch, "-cubin", "-I", str(HERE), "-I", SLANGINC,
               "-include", SHIM, cu, "-o", dst]
    subprocess.run(cmd, check=True)
    return dst


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(ROOT / "build" / "kernels"),
                    help="output directory (build/kernels/<sub>/<stem>.fatbin)")
    ap.add_argument("--build", default=str(ROOT / "build"),
                    help="dir holding build/<sub>/<ker>.cu (tools/build_all.py) and build/sm_35")
    ap.add_argument("--archs", default=ARCHS,
                    help="comma-separated SASS archs, or 'none' for a PTX-only set")
    ap.add_argument("--ptx", default="90",
                    help="compute capability of the forward-compat PTX image (default 90)")
    ap.add_argument("--nvcc", default=os.environ.get("NVCC", "nvcc"))
    ap.add_argument("--fatbinary", default=os.environ.get("FATBINARY", "fatbinary"))
    ap.add_argument("--jobs", default=os.cpu_count(), type=int)
    ap.add_argument("--no-sm35", action="store_true")
    a = ap.parse_args(argv)

    build = pathlib.Path(a.build)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # the kernel set's build ID (tools/gen_kernel_table.py); the sm_35 cubins
    # must come from the same kernel sources
    kid = (build / "sbz_kernels.id")
    if not kid.exists():
        raise SystemExit("missing %s (run tools/build_all.py first)" % kid)
    kid35 = build / "sm_35" / "sbz_kernels.id"
    if not a.no_sm35 and (not kid35.exists() or kid35.read_bytes() != kid.read_bytes()):
        raise SystemExit("%s is missing or differs from %s: the sm_35 cubins are from other "
                         "kernel sources; rerun `pixi run -e kepler build-sm35`" % (kid35, kid))
    if a.archs.lower() == "none":
        sass_archs = []
    else:
        sass_archs = [x.strip() for x in a.archs.split(",") if x.strip()]

    tmp = build / "_fatbin_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    jobs = []                      # tuple for compile_one
    per_kernel = []                # (stem, [ (sm, kind, file), ... ]) in image order

    for ker, sub, cubin in KERNEL_MAP:
        cu = build / sub / ("%s.cu" % ker)
        if not cu.exists():
            raise SystemExit("missing %s (run tools/build_all.py first)" % cu)
        stem = cubin[:-len(".cubin")]
        imgs = []
        # fatbinary's image3 `sm=` is the bare compute capability (35, 50, ...).
        sm35_cubin = build / "sm_35" / sub / cubin
        if sm35_cubin.exists():
            imgs.append(("35", "elf", sm35_cubin))
        elif not a.no_sm35:
            raise SystemExit(
                "missing %s\n  run `pixi run -e kepler build-sm35` to build the sm_35 "
                "cubins, or pass --no-sm35 to skip them" % sm35_cubin)
        for arch in sass_archs:
            s = "sm_%s" % arch
            img = tmp / ("%s_%s.sass" % (sub, s))
            jobs.append((ker, sub, s, str(cu), str(img), a.nvcc))
            imgs.append((arch, "elf", img))
        ptx_arch = "compute_%s" % a.ptx
        ptx = tmp / ("%s_compute_%s.ptx" % (sub, a.ptx))
        jobs.append((ker, sub, ptx_arch, str(cu), str(ptx), a.nvcc))
        imgs.append((a.ptx, "ptx", ptx))
        per_kernel.append((sub, stem, imgs))

    total_size = 0
    built = 0
    try:
        print("compiling %d images for %d kernels with %d worker(s)" %
              (len(jobs), len(KERNEL_MAP), a.jobs), flush=True)
        with concurrent.futures.ProcessPoolExecutor(max_workers=a.jobs) as ex:
            for _ in ex.map(compile_one, jobs):
                pass
        for sub, stem, imgs in per_kernel:
            d = out / sub
            d.mkdir(parents=True, exist_ok=True)
            fat = d / ("%s.fatbin" % stem)
            cmd = [a.fatbinary, "--create=" + str(fat), "-64"] + \
                  ["--image3=kind=%s,sm=%s,file=%s" % (kind, sm, f)
                   for sm, kind, f in imgs]
            subprocess.run(cmd, check=True)
            total_size += fat.stat().st_size
            built += 1
            print("* %s (%.1f KB, %d images)" % (fat, fat.stat().st_size / 1e3, len(imgs)),
                  flush=True)
        (out / "sbz_kernels.id").write_bytes(kid.read_bytes())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    wall = time.time() - t0
    print("done: %d fatbins in %.1f s, %.1f MB" % (built, wall, total_size / 1e6))
    return 0


if __name__ == "__main__":
    sys.exit(main())
