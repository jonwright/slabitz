"""Compile Slang zstd kernels to a CUDA cubin plus a kernel-name map.

Pipeline:  .slang  --slangc -target cu-->  .cu  --nvcc -cubin-->  .cubin

The Slang CUDA target emits ``extern "C" __global__`` functions, so each
kernel's CUDA symbol equals its Slang function name (no mangling); the name map
is therefore identity, but it mirrors bslz4_to_sparse's fatbin+maps convention.

Toolchain:
  * slangc  shader-slang v2026.22: --slangc, else $SLANGC, else the in-repo
            third_party/slang/bin/slangc (tools/fetch_slangc.sh), else PATH
  * nvcc    CUDA 12.x: --nvcc, else $NVCC, else PATH
"""

import argparse, json, os, pathlib, subprocess, sys


def run(cmd):
    print("  +", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("slang_files", nargs="+")
    vendored = pathlib.Path(__file__).resolve().parent.parent / "third_party/slang/bin/slangc"
    p.add_argument("--slangc", default=os.environ.get(
        "SLANGC", str(vendored) if vendored.exists() else "slangc"))
    p.add_argument("--nvcc", default=os.environ.get("NVCC", "nvcc"))
    p.add_argument("--arch", default=os.environ.get("CUDA_ARCH", "sm_89"))
    p.add_argument("--slang-profile", default="sm_6_0")
    p.add_argument("--entries", nargs="+", required=True)
    p.add_argument("--outdir", default="build")
    a = p.parse_args(argv)

    outdir = pathlib.Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)

    combined = outdir / "slabitz_kernels.slang"
    with open(combined, "w") as fo:
        for f in a.slang_files:
            fo.write("// -----  %s  -----\n" % f)
            fo.write(pathlib.Path(f).read_text())
            fo.write("\n")

    cu_file = outdir / "slabitz_kernels.cu"
    cmd = [a.slangc, str(combined), "-target", "cu", "-profile", a.slang_profile, "-o", str(cu_file)]
    for e in a.entries:
        cmd += ["-entry", e]
    run(cmd)

    cubin = outdir / "slabitz_kernels.cubin"
    run([a.nvcc, "-arch=" + a.arch, "-cubin", str(cu_file), "-o", str(cubin)])

    with open(outdir / "slabitz_kernels.json", "w") as fo:
        json.dump({e: e for e in a.entries}, fo, indent=2)

    print("Wrote", cubin, "+", outdir / "slabitz_kernels.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
