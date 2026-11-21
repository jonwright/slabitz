#!/usr/bin/env python3
"""Compile a Slang-generated .cu into one CUDA fatbin (driver-API friendly).

*** EXPERIMENTAL / DO NOT SHIP ***
A single fatbin requires all kernels in ONE CUDA module (one .cu per arch). To
build that .cu you must merge the Slang kernels, and the DS headers
(zstdgpu_shaders.h) are include-guarded: a merged module compiles the shader
entry bodies once, under the FIRST kernel's macro state, silently breaking every
later kernel that sets its own (e.g. IS_MULTI_WAVE in InitFseTable -> wrong C1
table). Use tools/build_all.py (per-kernel cubins, one CUDA module each) and load
each cubin as its own module in the host.

slabitz analogue of bslz4_to_sparse-dev/tools/build_cuda_fatbin.py, but the CUDA
source is already written by slangc (one .cu holding every entry), so there is no
per-symbol templating: name each Slang entry function.

  python3 tools/build_cuda_fatbin.py --cu build/combined.cu --out build/fatbin \
      --entries ExecuteSequences InitFseTable ... \
      --nvrtc <libnvrtc.so.12> --fatbinary <fatbinary> --arch 61,70,75,80,86,89

writes build/fatbin/slabitz.fatbin (a cubin per arch + PTX) and
build/fatbin/slabitz_kernels.json (front-end name -> lowered CUDA symbol). Each
NVRTC uses nvcc-like options; use --nvrtc-for ARCH=libnvrtc for archs the default
toolkit dropped (sm_35 from CUDA 11.8). NVRTC finds libnvrtc-builtins via
LD_LIBRARY_PATH. Slang emits `extern "C"` entry functions, so the CUDA symbol is
the entry name itself (no mangling): get_function("<entry>_<Kernel>").
"""

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import tempfile

OPTIONS = ["--std=c++17", "-ftz=true"]
PTX_DEFAULT = "90"
ARCH_DEFAULT = "50,52,60,61,70,75,80,86,89,90"


class Nvrtc(object):
    def __init__(self, path):
        self.lib = ctypes.CDLL(path)
        self.ptxas = os.path.join(os.path.dirname(os.path.abspath(path)), "..", "bin", "ptxas")
        ma, mi = ctypes.c_int(), ctypes.c_int()
        self.lib.nvrtcVersion(ctypes.byref(ma), ctypes.byref(mi))
        self.version = (ma.value, mi.value)
        self.options = OPTIONS if ma.value >= 11 else ["--std=c++11", "-ftz=true"]

    def check(self, rc, prog=None):
        if rc:
            self.lib.nvrtcGetErrorString.restype = ctypes.c_char_p
            msg = self.lib.nvrtcGetErrorString(rc).decode()
            if prog is not None:
                n = ctypes.c_size_t()
                self.lib.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
                log = ctypes.create_string_buffer(n.value)
                self.lib.nvrtcGetProgramLog(prog, log)
                msg += "\n" + log.value.decode()
            raise SystemExit("NVRTC: " + msg)

    def compile(self, src, names, arch):
        """(cubin bytes for sm_<arch> | PTX for compute_<arch>, {name: lowered})"""
        if arch.startswith("sm_") and not hasattr(self.lib, "nvrtcGetCUBIN"):
            ptx, lowered = self.compile(src, names, "compute_" + arch[3:])
            with tempfile.TemporaryDirectory() as tmp:
                p, c = os.path.join(tmp, "k.ptx"), os.path.join(tmp, "k.cubin")
                with open(p, "wb") as fh:
                    fh.write(ptx.rstrip(b"\0"))
                subprocess.check_call([self.ptxas, "-arch=" + arch, p, "-o", c])
                return open(c, "rb").read(), lowered
        L, prog = self.lib, ctypes.c_void_p()
        self.check(L.nvrtcCreateProgram(ctypes.byref(prog), src.encode(), b"slabitz.cu", 0, None, None))
        for n in names:
            self.check(L.nvrtcAddNameExpression(prog, n.encode()))
        opts = [o.encode() for o in self.options + ["-arch=%s" % arch]]
        self.check(L.nvrtcCompileProgram(prog, len(opts), (ctypes.c_char_p * len(opts))(*opts)), prog)
        lowered = {}
        for n in names:
            p = ctypes.c_char_p()
            self.check(L.nvrtcGetLoweredName(prog, n.encode(), ctypes.byref(p)))
            lowered[n] = p.value.decode()
        kind = "CUBIN" if arch.startswith("sm_") else "PTX"
        size = ctypes.c_size_t()
        self.check(getattr(L, "nvrtcGet%sSize" % kind)(prog, ctypes.byref(size)))
        buf = ctypes.create_string_buffer(size.value)
        self.check(getattr(L, "nvrtcGet%s" % kind)(prog, buf))
        L.nvrtcDestroyProgram(ctypes.byref(prog))
        return buf.raw, lowered


def symbols(cu):
    """Front-end symbols (entry names) from the .cu: every `extern "C" __global__ void F(`."""
    out = []
    for l in open(cu):
        if l.startswith('extern "C" __global__ void '):
            out.append(l.split("void ")[1].split("(")[0])
    return out


def build(nvrtc_path, fatbinary, cu, out, archs, ptx, nvrtc_for=None, names=None):
    src = open(cu, "rb").read().decode("utf-8", "replace")
    names = names or symbols(cu)
    expr = names
    nv = Nvrtc(nvrtc_path)
    other = dict(x.split("=", 1) for x in (nvrtc_for or []))
    old = sorted(other, key=int)
    nvs = {c: Nvrtc(other[c]) for c in old}
    al = archs.split(",")

    os.makedirs(out, exist_ok=True)
    # Keep scratch files (per-arch images, ptxas round trips) under the output dir, not /tmp.
    tempfile.tempdir = out
    fatbin = os.path.join(out, "slabitz.fatbin")
    with tempfile.TemporaryDirectory() as tmp:
        images, table = [], None
        for c in old + al + [None]:
            arch = "sm_%s" % c if c else "compute_%s" % ptx
            data, lowered = nvs.get(c, nv).compile(src, expr, arch)
            if table is not None and lowered != table:
                raise SystemExit("lowered names differ between architectures")
            table = lowered
            path = os.path.join(tmp, arch + (".cubin" if c else ".ptx"))
            with open(path, "wb") as fh:
                fh.write(data)
            images.append("--image3=kind=%s,sm=%s,file=%s" % ("elf" if c else "ptx", c or ptx, path))
        subprocess.check_call([fatbinary, "--create=" + fatbin, "-64"] + images)
    sym = {n: n for n in names}
    sym["_build"] = {"cu": os.path.basename(cu),
                     "source_sha256": hashlib.sha256(src.encode()).hexdigest(),
                     "nvrtc": {c: "%d.%d" % nvs.get(c, nv).version for c in old + al}}
    with open(os.path.join(out, "slabitz_kernels.json"), "w") as fh:
        json.dump(sym, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print("wrote %s (%.1f MB, %d kernels, sm_%s + compute_%s PTX)"
          % (fatbin, os.path.getsize(fatbin) / 1e6, len(names) - 1, ",".join(old + al), ptx))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cu", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--entries", nargs="*", default=[],
                    help="front-end kernel names (default: derive from the .cu)")
    ap.add_argument("--nvrtc", default=os.environ.get("NVRTC", "libnvrtc.so.12"))
    ap.add_argument("--fatbinary", default=os.environ.get("FATBINARY", "fatbinary"))
    ap.add_argument("--arch", default=os.environ.get("FATBIN_ARCH", ARCH_DEFAULT))
    ap.add_argument("--ptx", default=os.environ.get("FATBIN_PTX", PTX_DEFAULT))
    ap.add_argument("--nvrtc-for", action="append", default=[], metavar="ARCH=LIBNVRTC")
    a = ap.parse_args(argv)
    build(a.nvrtc, a.fatbinary, a.cu, a.out, a.arch, a.ptx, a.nvrtc_for,
          names=a.entries or None)


if __name__ == "__main__":
    sys.exit(main())
