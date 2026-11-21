#!/usr/bin/env python3
"""One-command build for the slabitz Slang->CUDA kernels.

Reproduces every kernel artifact from source, with no dependency on /tmp:
  transform DS tree -> build/ds
  slangc each kernel -> build/<dir>/<K>.cu   (entry `main`, symbol `main_0`)
  nvcc -cubin  each -> build/<dir>/<dir>.cubin

Each kernel is compiled as its OWN Slang module. Do not try to merge them into
one module/fatbin: the DS headers (zstdgpu_shaders.h) are include-guarded, so a
combined module compiles the shader entry bodies once under the FIRST kernel's
macro state (e.g. IS_MULTI_WAVE), silently breaking every later kernel that sets
its own. Per-kernel cubins, each loaded as its own CUDA module, are the correct
artifact. (tools/build_cuda_fatbin.py remains only as an experimental note.)

Tooling (in order): $SLANGC/env/--slangc then third_party/slang/bin/slangc;
$NVCC/env/--nvcc then `nvcc`. Always run slangc vendored in-repo so results don't
depend on a /tmp checkout. See tools/fetch_slangc.sh to restore slangc.

Usage:
  python3 tools/build_all.py                                   # per-kernel cubins (sm_89)
  CUDA_ARCH=sm_70 python3 tools/build_all.py                   # pick an arch
"""

import argparse
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
KERNELS = HERE.parent / "kernels"
# kernel -> (build subdir, cubin name). All kernels are Slang. These match the paths the test harnesses
# already take on argv (build/exec/exec.cubin etc.), so nothing else has to move.
KERNEL_MAP = [
    ("MemsetMemcpy", "mmcpy", "mmcpy.cubin"),
    ("ComputeDestBlockOffsets", "bo", "bo.cubin"),
    ("InitResources", "ir", "ir.cubin"),
    ("PropagateFseIndex", "pfi", "pfi.cubin"),
    ("DecodeHuffmanWeights", "dhw2", "dhw2.cubin"),
    ("PrefixSum", "ps", "ps.cubin"),
    ("ComputePrefixSum", "cps", "cps.cubin"),
    ("PrefixSequenceOffsets", "pso", "pso.cubin"),
    ("ParseFrames", "frames", "frames.cubin"),
    ("ParseCompressedBlocks", "pcb", "pcb.cubin"),
    ("InitFseTable", "ifs", "ifs.cubin"),
    ("DecompressHuffmanWeights", "dhw", "dhw.cubin"),
    ("InitHuffmanTableAndDecompressLiterals", "lit", "lit.cubin"),
    ("DecompressSequences", "seq", "seq.cubin"),
    ("DecompressSequencesMultiStream", "seqms", "seqms.cubin"),
    ("FinaliseSequenceOffsets", "fso", "fso.cubin"),
    ("ExecuteSequences", "exec", "exec.cubin"),
    ("BitUnshuffleSlow", "unshuf", "unshuf.cubin"),
    ("BitUnshuffleWarp", "unshufw", "unshufw.cubin"),
    ("Lz4DecodeBlocks", "lz4", "lz4.cubin"),
    ("Lz4DecodeUnshuffle", "lz4fus", "lz4fus.cubin"),
    ("BshufWalkChunks", "bswalk", "bswalk.cubin"),
]


def run(cmd, **kw):
    print("  +", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def find_slangc(arg):
    for c in (arg, os.environ.get("SLANGC"), ROOT / "third_party/slang/bin/slangc"):
        if c and (pathlib.Path(c).exists() or c == "slangc"):
            return c
    raise SystemExit("no slangc: set SLANGC or run tools/fetch_slangc.sh")


def transform(out):
    print("* transform DS tree -> %s/ds" % out)
    run([sys.executable, HERE / "transform_ds_for_slang.py", "--out", str(out / "ds")])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=os.environ.get("BUILD_DIR", ROOT / "build"))
    ap.add_argument("--arch", default=os.environ.get("CUDA_ARCH", "sm_89"))
    ap.add_argument("--slang-profile", default="sm_6_0")
    ap.add_argument("--slangc", default=os.environ.get("SLANGC"))
    ap.add_argument("--nvcc", default=os.environ.get("NVCC", "nvcc"))
    a = ap.parse_args(argv)

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    slangc = find_slangc(a.slangc)
    print("slangc:", slangc)

    transform(out)
    for ker, sub, cubin_name in KERNEL_MAP:
        d = out / sub
        d.mkdir(parents=True, exist_ok=True)
        print("* %s" % ker)
        src = KERNELS / ("%s.slang" % ker)
        cu = d / ("%s.cu" % ker)
        run([slangc, str(src), "-target", "cu", "-profile", a.slang_profile,
             "-I", str(HERE), "-I", str(out / "ds"), "-entry", "main", "-o", str(cu),
             "-reflection-json", str(d / ("%s.json" % ker))])
        # tools/slang_cuda_prevolta.h: pre-Volta wave-mask semantics (no-op on sm_70+),
        # so the same build works for any CUDA_ARCH.
        run([a.nvcc, "-arch=" + a.arch, "-cubin", "-I", str(HERE),
             "-I", str(ROOT / "third_party" / "slang" / "include"),
             "-include", "slang_cuda_prevolta.h", str(cu), "-o", str(d / cubin_name)])

    # C launch table (param names/kinds/element sizes + numthreads) for src/sbz_chain.c
    run([sys.executable, HERE / "gen_kernel_table.py", "--build", str(out)])
    print("done: kernels in %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
