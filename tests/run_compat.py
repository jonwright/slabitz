#!/usr/bin/env python3
"""T07: zstd version corpus runner (slabitz decode vs every zstd 1.4.7..1.6.0).

Decodes the T07 corpus (zstd files per version/mode plus bitshuffle-0.4.0
filter-32008 chunks) with the slabitz GPU decoder and checks each frame against
its manifest expectation: byte-exact (decode) or a documented reject code.
A stored manifest (tests/data/compat/manifest.json for the committed subset,
data/compat/manifest.json for the full corpus) gives per file the expectation
and the SHA-256 of the decoded output.

Every version's files are decoded in its OWN subprocess (a GPU fault poisons
the shared primary context for the rest of the process): a crashed subprocess
is a FAIL for its files and the run goes on.

There are two modes:

* default (no args): the committed subset, tests/data/compat/, added to
  `sbz test`.
* --full DIR: the whole corpus in DIR (data/compat), decoded through
  sbz_dctx_decompress_frames, and cross-checked against each version's own
  libzstd and against the 1.6.0 build.

The version x mode table: each cell shows pass / expected-reject / FAIL / n/a.
A mode a version's CLI lacks is n/a (recorded in the manifest, not a failure).

Usage:  python3 tests/run_compat.py [--full DIR] [--manifest PATH] [--table-only]
"""
import argparse
import ctypes as ct
import hashlib
import json
import os
import struct
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

LIB = os.path.join(ROOT, "build", "libslabitz.so")
KDIR = os.environ.get("SLABITZ_KERNEL_DIR", os.path.join(ROOT, "build"))
BSHUF_LIB = os.environ.get("SLABITZ_BSHUF040",
                           os.path.join(os.environ.get("SLABITZ_DEV_ROOT", ""),
                                        "toolchains", "bitshuffle-0.4.0",
                                        "libbshuf040.so"))
DEFAULT_MANIFEST = os.path.join(ROOT, "tests", "data", "compat", "manifest.json")

ZSTD_MAGIC = 0xFD2FB528
SKIP_BASE = 0x184D2A50
SBZ_ERR_BACKEND, SBZ_ERR_CORRUPT, SBZ_ERR_LIMIT = -1, -5, -6
EXPECT_CODES = {"SBZ_ERR_CORRUPT": SBZ_ERR_CORRUPT, "SBZ_ERR_LIMIT": SBZ_ERR_LIMIT,
                "-5": SBZ_ERR_CORRUPT, "-6": SBZ_ERR_LIMIT}

# order the table in
VERSIONS_ORDER = ["v1.4.7", "18d02cbf", "v1.4.9", "v1.5.0", "v1.5.2",
                  "v1.5.5", "v1.5.6", "v1.5.7", "v1.6.0"]
MODES_ORDER = ["level1", "level3", "level9", "level19", "ultra22",
               "long27", "long30", "long31", "T0", "streaming", "nocontent",
               "check", "nocheck", "B65536", "patchfrom", "dict"]
BSHUF_VERSION = "bshuf040"
ROWS = VERSIONS_ORDER + [BSHUF_VERSION]
COLS = MODES_ORDER + [BSHUF_VERSION]

CURRENT = ["<none>"]


class GpuFault(RuntimeError):
    pass


# --------------------------------------------------------------------------
# decoder plumbing
# --------------------------------------------------------------------------
class FrameSpan(ct.Structure):
    _fields_ = [("offs", ct.c_uint64), ("size", ct.c_uint64)]


def load():
    lib = ct.CDLL(LIB)
    lib.sbz_dctx_create_cuda.restype = ct.c_void_p
    lib.sbz_dctx_create_cuda.argtypes = [ct.c_int, ct.c_char_p, ct.c_char_p,
                                         ct.c_size_t]
    lib.sbz_dctx_destroy.argtypes = [ct.c_void_p]
    lib.sbz_dctx_error.restype = ct.c_char_p
    lib.sbz_dctx_error.argtypes = [ct.c_void_p]
    lib.sbz_dctx_decompress.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_size_t]
    lib.sbz_dctx_decompress_frames.argtypes = [ct.c_void_p, ct.c_char_p,
                                               ct.c_size_t, ct.POINTER(FrameSpan),
                                               ct.c_uint32]
    lib.sbz_dctx_bshuf_decompress.argtypes = [ct.c_void_p, ct.c_char_p,
                                              ct.c_size_t, ct.POINTER(FrameSpan),
                                              ct.c_uint32, ct.c_uint32]
    lib.sbz_dctx_read_buffer.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_void_p,
                                         ct.POINTER(ct.c_size_t)]
    return lib


def read_buf(lib, dc, name):
    n = ct.c_size_t(0)
    assert lib.sbz_dctx_read_buffer(dc, name.encode(), None, ct.byref(n)) == 0
    b = ct.create_string_buffer(max(n.value, 1))
    assert lib.sbz_dctx_read_buffer(dc, name.encode(), b, ct.byref(n)) == 0, \
        lib.sbz_dctx_error(dc)
    return b.raw[:n.value]


def make_dctx(lib):
    err = ct.create_string_buffer(512)
    dc = lib.sbz_dctx_create_cuda(0, KDIR.encode(), err, len(err))
    if not dc:
        raise GpuFault("sbz_dctx_create_cuda: %s" % err.value.decode())
    return dc


def decompress_frames(lib, dc, data, spans):
    arr = (FrameSpan * len(spans))(*[FrameSpan(o, s) for o, s in spans])
    rc = lib.sbz_dctx_decompress_frames(dc, data, len(data), arr, len(spans))
    if rc == SBZ_ERR_BACKEND:
        raise GpuFault("sbz_dctx_decompress_frames: %s"
                       % lib.sbz_dctx_error(dc).decode())
    if rc == 0:
        return rc, read_buf(lib, dc, "UnCompressedFramesData")
    return rc, b""


def decompress(lib, dc, data):
    rc = lib.sbz_dctx_decompress(dc, data, len(data))
    if rc == SBZ_ERR_BACKEND:
        raise GpuFault("sbz_dctx_decompress: %s" % lib.sbz_dctx_error(dc).decode())
    if rc == 0:
        return rc, read_buf(lib, dc, "UnCompressedFramesData")
    return rc, b""


def decompress_bshuf(lib, dc, data, elem):
    arr = (FrameSpan * 1)(FrameSpan(0, len(data)))
    rc = lib.sbz_dctx_bshuf_decompress(dc, data, len(data), arr, 1, elem)
    if rc == SBZ_ERR_BACKEND:
        raise GpuFault("sbz_dctx_bshuf_decompress: %s"
                       % lib.sbz_dctx_error(dc).decode())
    if rc == 0:
        return rc, read_buf(lib, dc, "BshufOutput")
    return rc, b""


# --------------------------------------------------------------------------
# frame span parsing (no oracle, no python-zstandard)
# --------------------------------------------------------------------------
def frame_spans(buf):
    """Yield (offset, size) of each zstd frame in a stream (skipping skippable)."""
    n = len(buf)
    pos = 0
    spans = []
    while pos < n:
        magic = struct.unpack_from("<I", buf, pos)[0]
        if SKIP_BASE <= magic <= SKIP_BASE + 0xF:
            fsz = struct.unpack_from("<I", buf, pos + 4)[0]
            pos += 8 + fsz
            continue
        if magic != ZSTD_MAGIC:
            raise ValueError("bad magic at %d" % pos)
        orig = pos
        desc = buf[pos + 4]
        single = bool(desc & 0x20)
        cab = 5
        if not single:
            cab += 1
        dict_flag = desc & 3
        cab += {0: 0, 1: 1, 2: 2, 3: 4}[dict_flag]
        fcs_flag = (desc >> 6) & 3
        if single or fcs_flag:
            cab += {0: 1, 1: 2, 2: 4, 3: 8}[fcs_flag]
        pos = orig + cab
        while True:
            bh = int.from_bytes(buf[pos:pos + 3], "little")
            last = bh & 1
            btype = (bh >> 1) & 3
            bsize = (bh >> 3) & 0x3FFFFF
            pos += 3
            if btype == 0:
                pos += bsize
            elif btype == 1:
                pos += 1
            elif btype == 2:
                pos += bsize
            else:
                raise ValueError("reserved block type")
            if last:
                break
        if desc & 0x04:
            pos += 4
        spans.append((orig, pos - orig))
    return spans


# --------------------------------------------------------------------------
# reference decoders (each version's own libzstd / the 1.6.0 build)
# --------------------------------------------------------------------------
def _cli_of(devroot, version):
    return os.path.join(devroot, "toolchains", "zstd-" + version,
                        "programs", "zstd")


def cli_reference(manifest, entry, full_dir, devroot, version):
    """Decode a zstd entry's file with `version`'s CLI, return sha256 + size.
    Raises if the CLI fails to decode a frame the corpus marks as a zstd stream."""
    filepath = os.path.join(full_dir, entry["file"])
    cli = _cli_of(devroot, version)
    cmd = [cli, "-d", "-c"]
    if entry["mode"] == "patchfrom":
        base = os.path.join(devroot, "toolchains", "zstd-" + entry["version"],
                            "_patch_base.bin")
        cmd = [cli, "-d", "-c", "--patch-from=" + base]
    elif entry["mode"] == "dict":
        dfile = os.path.join(devroot, "toolchains", "zstd-" + entry["version"],
                             "_dict.bin")
        cmd = [cli, "-d", "-c", "-D", dfile]
    p = subprocess.run(cmd + [filepath], capture_output=True)
    if p.returncode != 0:
        raise RuntimeError("%s CLI -d failed (%s): %s" % (
            version, p.returncode, p.stderr.decode("latin1")[:300]))
    out = p.stdout
    return hashlib.sha256(out).hexdigest(), len(out)


def bshuf_reference(entry, full_dir):
    """Decode a bitshuffle chunk with bitshuffle-0.4.0's bshuf_decompress_zstd."""
    import ctypes
    filepath = os.path.join(full_dir, entry["file"])
    data = open(filepath, "rb").read()
    elem = entry["bshuf"]["elem"]
    block = entry["bshuf"]["block"] or 0
    # bitshuffle 0.4.0 default block size (elements): 8192 bytes / elem,
    # rounded down to a multiple of 8, min 128.
    if not block:
        block = max((8192 // elem) // 8 * 8, 128)
    total = int.from_bytes(data[:8], "big")
    n = total // elem
    lib = ctypes.CDLL(BSHUF_LIB)
    lib.bshuf_decompress_zstd.restype = ctypes.c_int64
    lib.bshuf_decompress_zstd.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t,
                                          ct.c_size_t, ct.c_size_t]
    # the 12-byte chunk header is NOT part of bshuf_decompress_zstd's input
    inb = ct.create_string_buffer(data, len(data))
    outb = ct.create_string_buffer(entry["decoded_size"] or total)
    m = lib.bshuf_decompress_zstd(ct.byref(inb, 12), outb, n, elem, block)
    if m < 0:
        raise RuntimeError("bshuf_decompress_zstd failed: %d" % m)
    out = outb.raw[:total]
    return hashlib.sha256(out).hexdigest(), len(out)


# --------------------------------------------------------------------------
# worker: decode one version's entries in one subprocess
# --------------------------------------------------------------------------
def worker(args):
    lib = load()
    version = getattr(args, "worker", None) or getattr(args, "version", None)
    namef = getattr(args, "name", None)
    manifest = json.load(open(args.manifest))
    entries = [e for e in manifest["entries"] if e["version"] == version
               and (namef is None or e["name"] == namef)]
    results = []
    for entry in entries:
        CURRENT[0] = "%s/%s" % (entry["version"], entry["mode"])
        filepath = os.path.join(args.dir, entry["file"])
        data = open(filepath, "rb").read()
        r = {"key": "%s|%s" % (entry["version"], entry["name"]),
             "name": entry["name"], "mode": entry["mode"],
             "version": entry["version"], "expectation": entry["expectation"],
             "status": "ok", "reason": ""}
        dc = None
        try:
            dc = make_dctx(lib)
            if entry["kind"] == "bshuf":
                rc, got = decompress_bshuf(lib, dc, data, entry["bshuf"]["elem"])
            else:
                spans = frame_spans(data)
                rc, got = decompress_frames(lib, dc, data, spans)
            r["rc"] = rc
            if entry["expectation"] == "decode":
                want = entry["decoded_sha256"]
                if rc != 0:
                    r["status"] = "FAIL"
                    r["reason"] = "decode rc=%d (%s)" % (rc, err_str(lib, dc))
                else:
                    got = got[:entry["decoded_size"]]
                    if hashlib.sha256(got).hexdigest() != want:
                        r["status"] = "FAIL"
                        r["reason"] = ("output sha256 mismatch "
                                       "(got %d bytes)" % len(got))
                    else:
                        r["compared"] = len(got)
            else:
                code = EXPECT_CODES.get(entry["expectation"].split(":")[1])
                if rc != code:
                    r["status"] = "FAIL"
                    r["reason"] = ("expected reject:%d got rc=%d (%s)"
                                   % (code, rc, err_str(lib, dc)))
                else:
                    r["expected_reject"] = True
        except GpuFault as e:
            r["status"] = "FAIL"
            r["reason"] = "GPU fault: %s" % e
        finally:
            if dc is not None:
                lib.sbz_dctx_destroy(dc)
        results.append(r)
    print(json.dumps(results))
    return 0


def err_str(lib, dc):
    try:
        return lib.sbz_dctx_error(dc).decode()
    except Exception:
        return "?"


# --------------------------------------------------------------------------
# reference cross-check (--full)
# --------------------------------------------------------------------------
def reference_check_entry(entry, full_dir):
    """Compare the reference (own libzstd) and the 1.6.0 build against the
    manifest sha for a decode entry.  Returns (ok, detail)."""
    devroot = os.environ.get("SLABITZ_DEV_ROOT", "")
    if entry["kind"] == "bshuf":
        if not os.path.exists(BSHUF_LIB):
            return True, "no bshuf lib (skipped)"
        try:
            sha, size = bshuf_reference(entry, full_dir)
        except Exception as e:
            return False, "bshuf reference failed: %s" % e
        ok = sha == entry["decoded_sha256"]
        return ok, ("bshuf040 %s/%s ref matches" % ("bshuf040", entry["name"])
                    if ok else "bshuf040 ref MISMATCH")
    # zstd: own version + 1.6.0
    errs = []
    good = True
    for tag, vv in (("own", entry["version"]), ("v1.6.0", "v1.6.0")):
        try:
            sha, size = cli_reference(None, entry, full_dir, devroot, vv)
            if sha != entry["decoded_sha256"]:
                good = False
                errs.append("%s CLI sha mismatch" % vv)
        except Exception as e:
            good = False
            errs.append("%s CLI ref failed: %s" % (vv, e))
    return good, ("ref matches" if good else "; ".join(errs))


# --------------------------------------------------------------------------
# table
# --------------------------------------------------------------------------
def build_table(results_by_name):
    """results_by_name: {manifest-name -> worker result}.  Returns table cells."""
    cells = {}
    for name, r in results_by_name.items():
        version = r["version"]
        mode = r["mode"]
        key = (version, mode)
        if r["expectation"] == "n/a":
            cells[key] = "n/a"
            continue
        if r["status"] == "FAIL":
            cells[key] = "FAIL"
            continue
        piece = "reject" if r.get("expected_reject") else "pass"
        cur = cells.get(key, "")
        if cur == "":
            cells[key] = piece
        elif piece not in cur.split("+"):
            cells[key] = cur + "+" + piece
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", dest="full_dir", default=None,
                    help="decode the whole corpus in this directory")
    ap.add_argument("--manifest", dest="manifest", default=None,
                    help="use this manifest (default: the committed subset)")
    ap.add_argument("--dir", dest="dir", default=None,
                    help="data directory holding the manifest's files "
                         "(default: the manifest's directory)")
    args = ap.parse_args()

    if args.full_dir:
        manifest_path = os.path.join(args.full_dir, "manifest.json")
    elif args.manifest:
        manifest_path = args.manifest
    else:
        manifest_path = DEFAULT_MANIFEST
    direc = args.dir or args.full_dir or os.path.dirname(manifest_path)

    manifest = json.load(open(manifest_path))
    entries = manifest["entries"]
    by_version = {}
    for e in entries:
        by_version.setdefault(e["version"], []).append(e)

    # decode: spawn one subprocess per ENTRY (a GPU fault poisons the CUDA
    # primary context for the whole process - see T06 - so decoding all of a
    # version's files in one process would cascade a single corrupted frame
    # into every later file of that version; isolating per entry reports the
    # fault on the exact entry that caused it, which is the point of T07).
    results_by_name = {}
    failures = []
    for version in sorted(by_version, key=lambda v: ROWS.index(v) if v in ROWS else 99):
        for e in by_version[version]:
            cmd = [sys.executable, os.path.abspath(__file__), "--worker", version,
                   "--name", e["name"],
                   "--manifest", manifest_path,
                   "--dir", os.path.abspath(direc),
                   "--kd", KDIR]
            p = subprocess.run(cmd, capture_output=True, text=True)
            key = "%s|%s" % (version, e["name"])
            if p.returncode != 0:
                results_by_name[key] = {
                    "name": e["name"], "version": version, "mode": e["mode"],
                    "expectation": e["expectation"], "status": "FAIL",
                    "reason": "subprocess rc=%d" % p.returncode}
                failures.append("%s: worker crashed (%s)"
                                % (e["name"], "gpu fault or crash"))
                continue
            try:
                res = json.loads(p.stdout)
            except Exception:
                res = []
            for r in res:
                results_by_name[r["key"]] = r
                if r["status"] == "FAIL":
                    failures.append("%s: %s" % (r["name"], r["reason"]))

    # cross-version reference check (only --full)
    ref_failures = []
    total_compared = 0
    if args.full_dir:
        for e in entries:
            if e["expectation"] == "decode":
                ok, detail = reference_check_entry(e, direc)
                if not ok:
                    ref_failures.append("%s: %s" % (e["name"], detail))
        for e in entries:
            r = results_by_name.get("%s|%s" % (e["version"], e["name"]))
            if r and r["status"] == "ok" and r.get("compared"):
                total_compared += r["compared"]

    print_table(results_by_name, manifest)
    print()
    print("entries=%d  compared_bytes=%d"
          % (len(entries), total_compared or _sum_compared(results_by_name)))
    fail_list = failures + ref_failures
    print("FAILs=%d  reference_failures=%d" % (len(failures), len(ref_failures)))
    for f in fail_list:
        print("  " + f)
    if fail_list:
        print("FAIL")
        return 1
    print("PASS")
    return 0


def _sum_compared(results_by_name):
    s = 0
    for r in results_by_name.values():
        if r.get("compared"):
            s += r["compared"]
    return s


def print_table(results_by_name, manifest):
    cells = build_table(results_by_name)
    print("zstd version x mode table (pass / reject / FAIL / n/a):")
    rows = [v for v in ROWS if any(e["version"] == v for e in manifest["entries"])]
    cols = [c for c in COLS if any(e["mode"] == c for e in manifest["entries"])]
    header = "  %-10s " % "ver\\mode" + "  ".join("%-11s" % c for c in cols)
    print(header)
    for v in rows:
        row = []
        for c in cols:
            cell = cells.get((v, c), ".")
            row.append("%-11s" % cell)
        print("  %-10s " % v + "  ".join(row))
    print("  (.) = absent from the corpus")


def run_worker(args):
    return worker(args)


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--worker" in argv:
        ap = argparse.ArgumentParser()
        ap.add_argument("--worker")
        ap.add_argument("--name")
        ap.add_argument("--manifest")
        ap.add_argument("--dir")
        ap.add_argument("--kd")
        a = ap.parse_args(argv)
        KDIR = a.kd
        sys.exit(worker(a))
    else:
        sys.exit(main())
