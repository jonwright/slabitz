#!/usr/bin/env python3
"""T07: zstd version corpus generator (bitshuffle 0.4.0's zstd to 1.6.0).

Builds the zstd CLI + lib for each version and the bitshuffle 0.4.0 shared
library, then generates a corpus of zstd-compressed files (and bitshuffle-zstd
chunks) that the slabitz GPU decoder must either decode byte-exact or reject
with a documented code.

Two corpora are produced:

* FULL:    $SLABITZ_DEV_ROOT/data/compat/  (shared root, this task owns it;
           one file per (version, mode), the window cases, patch/dict, and the
           bitshuffle 0.4.0 chunks).  Its manifest is data/compat/manifest.json.
* SUBSET:  tests/data/compat/   (committed, at most 5 MB): one file per version
           for levels 3 and 19, the no-FCS and window-30 cases (expected reject
           until T09/T10), and the bitshuffle 0.4.0 chunks.

The manifest records, per file: version, mode, input kind (zstd | bshuf),
expectation (decode | reject:<code> | n/a), the SHA-256 and size of the decoded
output (== the original uncompressed input: zstd + bitshuffle are lossless),
and the CLI detail needed to reproduce it.  Modes an old CLI lacks are recorded
as n/a, not as a failure.

Usage:
    python3 tools/gen_compat_corpus.py --list         [print versions + modes]
    python3 tools/gen_compat_corpus.py --build        [build all CLIs + bshuf lib]
    python3 tools/gen_compat_corpus.py --generate     [write FULL + SUBSET corpora]
    python3 tools/gen_compat_corpus.py --generate --subset-only
    python3 tools/gen_compat_corpus.py --generate --full-only

Never downloads anything: all sources are in toolchains/src/ (read-only).
Builds out of a copy in toolchains/zstd-<version>/.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent       # worktree root
DEV = Path(os.environ.get("SLABITZ_DEV_ROOT")) if os.environ.get("SLABITZ_DEV_ROOT") else None
TL_SRC = DEV / "toolchains" / "src"
TL_BUILD = DEV / "toolchains"
FULL_DIR = DEV / "data" / "compat"
SUBSET_DIR = ROOT / "tests" / "data" / "compat"
TMP = Path(os.environ.get("TMPDIR", str(ROOT / "scratch_work")))

# The 9 zstd versions, in corpus order.  Directory names under toolchains/src/.
VERSIONS = ["v1.4.7", "18d02cbf", "v1.4.9", "v1.5.0", "v1.5.2",
            "v1.5.5", "v1.5.6", "v1.5.7", "v1.6.0"]

BSHUF_SRC = TL_SRC / "bitshuffle-0.4.0"
BSHUF_LIB = TL_BUILD / "bitshuffle-0.4.0" / "libbshuf040.so"

# zstd of the bitshuffle-0.4.0 submodule (the floor encoder).
BSHUF_ZSTD = "18d02cbf"

REJECT = "reject:SBZ_ERR_UNSUPPORTED"   # valid zstd this decoder does not handle (T09)
REJECT_CORRUPT = "reject:SBZ_ERR_CORRUPT"  # --patch-from: indistinguishable from corrupt (T09)

# ---- zstd CLI modes ---------------------------------------------------------
# (mode, list-of-CLI-args, input-key, needs-special-arg)
MODES = [
    ("level1",     ["-1"],            "text",   False),
    ("level3",     ["-3"],            "text",   False),
    ("level9",     ["-9"],            "text",   False),
    ("level19",    ["-19"],           "text",   False),
    ("ultra22",    ["--ultra", "-22"], "text",  False),
    ("long27",     ["--long=27"],     "long27", False),
    ("long30",     ["--long=30"],     "long30", False),
    ("long31",     ["--long=31"],     "long31", False),
    ("T0",         ["-T0"],           "text",   False),
    ("streaming",  ["-c"],            "text",   False),   # stdin -> no FCS
    ("nocontent",  ["--no-content-size"], "text", False),  # no FCS
    ("check",      ["--check"],       "text",   False),
    ("nocheck",    ["--no-check"],    "text",   False),
    ("B65536",     ["-B65536"],       "text",   False),
    ("patchfrom",  [],                "text",   True),    # --patch-from=<base>
    ("dict",       [],                "text",   True),    # -D <dict>
]

# Modes whose frame format the decoder today is expected to reject.
EXPECT_REJECT_MODES = {"streaming", "nocontent", "long30", "long31",
                       "patchfrom", "dict"}
# long27 (window 2^27 < 2^29) decodes.

# Extra per-input tasks, in addition to the MODES above: the same mode (and CLI
# args) run on a *different* input.  Detector crops (raw little-endian frames,
# no bitshuffle), a 1 MB incompressible random input (raw blocks), and a >= 24 MB
# multi-frame detector input for -T0 so the multithreaded encoder really splits
# the work into several jobs.
EXTRA_TASKS = [
    ("level1",   ["-1"],             "det_u16"),
    ("level1",   ["-1"],             "det_u32"),
    ("level3",   ["-3"],             "det_u16"),
    ("level3",   ["-3"],             "det_u32"),
    ("level3",   ["-3"],             "random"),
    ("level19",  ["-19"],            "det_u16"),
    ("level19",  ["-19"],            "det_u32"),
    ("level19",  ["-19"],            "random"),
    ("ultra22",  ["--ultra", "-22"], "det_u16"),
    ("ultra22",  ["--ultra", "-22"], "det_u32"),
    ("check",    ["--check"],        "det_u16"),
    ("check",    ["--check"],        "det_u32"),
    ("B65536",   ["-B65536"],        "det_u16"),
    ("B65536",   ["-B65536"],        "det_u32"),
    ("T0",       ["-T0"],            "det_multi"),
]

DICT_SAMPLES = 12
DICT_SIZE = 4096

# window source sizes in MiB: large enough that zstd --long writes a
# multi-segment frame whose window descriptor is >= the requested log.
# long27 -> 2^27 (128 MiB); long30 -> 2^30 (1 GiB); long31 -> 2^31 (2 GiB).
LONG_SIZES = {"long27": 150, "long30": 1200, "long31": 2200}

# detector crop width (columns) and the number of frames for the -T0 input.
DET_NCOLS = 512
T0_FRAMES = 6
RANDOM_SIZE = 1024 * 1024   # 1 MB incompressible


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def sha256_zeros(n):
    """sha256 of n zero bytes, computed streaming (never materialise)."""
    h = hashlib.sha256()
    chunk = bytes(1 << 20)
    while n:
        take = min(len(chunk), n)
        h.update(chunk[:take])
        n -= take
    return h.hexdigest()


def zstd_dir(version):
    return TL_BUILD / ("zstd-" + version)


def zstd_cli(version):
    return zstd_dir(version) / "programs" / "zstd"


def build_ok(version):
    return zstd_cli(version).exists()


def build_zstd(version):
    """Build one zstd version's lib + CLI out of a copy.  Returns (ok, reason)."""
    src = TL_SRC / ("zstd-" + version)
    dest = zstd_dir(version)
    if build_ok(version):
        return True, "already built"
    if not src.is_dir():
        return False, "missing source tree toolchains/src/zstd-%s" % version
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    for subdir, tgt in (("lib", "libzstd"), ("programs", "zstd")):
        p = subprocess.run(["make", "-C", subdir, tgt], cwd=dest,
                           capture_output=True, text=True)
        if p.returncode != 0:
            return False, "make %s %s failed:\n%s" % (subdir, tgt, p.stderr[-2000:])
    return build_ok(version), ""


def build_bshuf():
    """Build libbshuf040.so (bitshuffle 0.4.0, -DZSTD_SUPPORT) against
    zstd-<BSHUF_ZSTD>'s lib and its own lz4.  Returns (ok, reason)."""
    if BSHUF_LIB.exists():
        return True, "already built"
    if not BSHUF_SRC.is_dir():
        return False, "missing toolchains/src/bitshuffle-0.4.0"
    zs = TL_SRC / ("zstd-" + BSHUF_ZSTD)
    if not (zs / "lib").is_dir():
        return False, "missing zstd source for bitshuffle (%s)" % BSHUF_ZSTD
    bsrc = BSHUF_SRC / "src"
    files = [str(bsrc / f) for f in ("bitshuffle.c", "bitshuffle_core.c",
                                     "iochain.c")]
    files += [str(BSHUF_SRC / "lz4" / "lz4.c")]
    for sub in ("common", "compress", "decompress"):
        files += sorted(str(p) for p in (zs / "lib" / sub).glob("*.c"))
    cmd = ["cc", "-O2", "-std=gnu99", "-DZSTD_SUPPORT", "-DZSTD_DISABLE_ASM=1",
           "-DZSTD_LEGACY_SUPPORT=0", "-fPIC", "-shared", "-Wl,--no-undefined",
           "-I", str(bsrc), "-I", str(BSHUF_SRC / "lz4"),
           "-I", str(zs / "lib"), "-I", str(zs / "lib" / "common")] + files
    dest = BSHUF_LIB.parent
    dest.mkdir(parents=True, exist_ok=True)
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(dest))
    if p.returncode != 0:
        return False, "bshuf build failed:\n%s" % p.stderr[-2000:]
    return BSHUF_LIB.exists(), ""


def _ret_cmd(cmd, cwd=None, stdin_data=None, timeout=3600):
    try:
        p = subprocess.run(cmd, capture_output=True, cwd=cwd, input=stdin_data,
                           timeout=timeout)
        return p.returncode, p.stdout, p.stderr.decode("latin1")
    except subprocess.TimeoutExpired as e:
        return 124, b"", "timeout"


# ---- inputs ----------------------------------------------------------------
def make_text():
    lines = []
    base = ("The quick brown fox jumps over the lazy dog. "
            "Pack my box with five dozen liquor jugs. "
            "Sphinx of black quartz, judge my vow. ")
    n = 0
    while sum(len(x) for x in lines) < 120000:
        lines.append(base * ((n % 7) + 1))
        lines.append(str(n) + ": " + ("0" if n % 11 else "1") * (n % 40))
        n += 1
    return "".join(lines).encode("ascii")


def make_random(n=120000):
    import numpy as np
    rng = np.random.default_rng(20261127)
    return rng.integers(0, 256, n, dtype=np.uint8).tobytes()


def crop_detector(h5name, dtype, ncols):
    import h5py
    import hdf5plugin  # noqa: F401  -- registers the filter-32008 decoder
    import numpy as np
    path = os.path.join(DEV, "data", "real", h5name)
    with h5py.File(path, "r") as h:
        d = h["entry_0000/ESRF-ID11/eiger/data"]
        a = d[0]
    return np.asarray(a[:, :ncols]).astype(np.dtype(dtype), copy=False).tobytes()


def prepare_long_files():
    """Write the window-encoding zero sources to TMP once; return {mode: path},
    plus the decoded (reference) size and sha256 per mode."""
    TMP.mkdir(parents=True, exist_ok=True)
    info = {}
    for mode, mb in LONG_SIZES.items():
        path = TMP / ("_long_%s.bin" % mode)
        if not path.exists():
            msg = "generating %d MiB zero source for %s" % (mb, mode)
            print("  " + msg)
            with open(path, "wb") as f:
                total = mb * 1024 * 1024
                chunk = bytes(1 << 20)
                while total:
                    take = min(len(chunk), total)
                    f.write(chunk[:take])
                    total -= take
        n = mb * 1024 * 1024
        info[mode] = {"path": path, "size": n, "sha256": sha256_zeros(n)}
    return info


def generate_inputs():
    import numpy as np
    det_u16 = crop_detector("silicon_u16_zstd2_256k.h5", "uint16", DET_NCOLS)
    det_u32 = crop_detector("ceo2_u32_zstd2_256k.h5", "uint32", DET_NCOLS)
    # det_multi: T0_FRAMES detector frames, each a small per-frame change so the
    # encoder cannot collapse them into one long match -> -T0 must split jobs.
    arr = np.frombuffer(det_u32, dtype=np.uint32).copy()
    frames = []
    for i in range(T0_FRAMES):
        f = arr.copy()
        f[::4096] = (f[::4096] + (i + 1)).astype(np.uint32)
        frames.append(f.tobytes())
    return {
        "text": make_text(),
        "random": make_random(RANDOM_SIZE),
        "det_u16": det_u16,
        "det_u32": det_u32,
        "det_multi": b"".join(frames),
    }


# ---- build the per-version special files (patch base, dict) ---------------
def perver_dir(version):
    d = zstd_dir(version)
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_patch_base(version, text):
    d = perver_dir(version)
    base = d / "_patch_base.bin"
    if not base.exists():
        base.write_bytes(text[:20000])
    return str(base)


def make_dict(version, text):
    d = perver_dir(version)
    dfile = d / "_dict.bin"
    if not dfile.exists():
        ddir = d / "_samples"
        ddir.mkdir(exist_ok=True)
        for i in range(DICT_SAMPLES):
            s = text[20000 + i * 1000:21000 + i * 1000]
            if len(s) < 256:
                s = s + text[:256 - len(s)]
            (ddir / ("s%d" % i)).write_bytes(s)
        rc, _, err = _ret_cmd([str(zstd_cli(version)), "--train",
                               "-o", str(dfile), "--maxdict=%d" % DICT_SIZE]
                              + sorted(str(p) for p in ddir.iterdir()))
        if rc != 0 or not dfile.exists():
            return None
    if not dfile.exists():
        return None
    return str(dfile)


# ---- CLI run ---------------------------------------------------------------
def run_zstd(cli, args, infile=None, stdin_data=None):
    """Run the CLI, writing the compressed stream to STDOUT.  stdin_data feeds
    stdin (streaming, no FCS) when infile is None."""
    cmd = [str(cli)] + list(args)
    if "-c" not in cmd and "--stdout" not in cmd:
        cmd.append("-c")
    if infile is not None:
        cmd += ["-f", str(infile)]
    return _ret_cmd(cmd, stdin_data=stdin_data)


def frame_has_fcs(frame_bytes):
    if len(frame_bytes) < 5:
        return False
    desc = frame_bytes[4]
    single = bool(desc & 0x20)
    fcs_flag = (desc >> 6) & 3
    return bool(single or fcs_flag)


# ---- corpus generation -----------------------------------------------------
def gen_full(log):
    inputs = generate_inputs()
    longinfo = prepare_long_files()
    full_files = FULL_DIR / "zstd"
    full_files.mkdir(parents=True, exist_ok=True)
    entries = []

    # MODES (text-based / window / patch / dict) plus every EXTRA_TASKS entry.
    tasks = list(MODES) + [(m, a, i, False) for m, a, i in EXTRA_TASKS]
    for version in VERSIONS:
        cli = zstd_cli(version)
        for mode, args_in, inkey, special in tasks:
            args = list(args_in)
            data = inputs.get(inkey) if inkey not in longinfo else None

            if mode == "patchfrom":
                base = make_patch_base(version, inputs["text"])
                args = ["--patch-from=" + base]
                data = inputs["text"]
            elif mode == "dict":
                dfile = make_dict(version, inputs["text"])
                if dfile is None:
                    log("  %s/%s: dict train failed -> n/a" % (version, mode))
                    entries.append(_entry(version, mode, inkey, None, "n/a"))
                    continue
                args = ["-D", dfile]
                data = inputs["text"]

            name = "%s_%s.zst" % (mode, inkey)
            relfile = "zstd/%s/%s" % (version, name)
            out = FULL_DIR / relfile

            if mode == "streaming":
                rc, cbytes, err = run_zstd(cli, ["-c"], stdin_data=data)
            elif inkey in longinfo:
                li = longinfo[inkey]
                rc, cbytes, err = run_zstd(cli, args, infile=li["path"])
                data = None
            else:
                tmpin = full_files / ("_in_%s_%s" % (version, inkey))
                tmpin.write_bytes(data)
                rc, cbytes, err = run_zstd(cli, args, infile=str(tmpin))
                tmpin.unlink(missing_ok=True)

            if rc != 0 or not cbytes:
                first = (err.strip().splitlines() or [""])[0]
                log("  %s/%s: CLI error rc=%d (%s) -> n/a"
                    % (version, mode, rc, first[:100]))
                entries.append(_entry(version, mode, inkey, None, "n/a"))
                continue

            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(cbytes)
            if inkey in longinfo:
                li = longinfo[inkey]
                decoded = None
                dsize = li["size"]
                dsha = li["sha256"]
            else:
                decoded = data
                dsize = len(data)
                dsha = sha256(data)

            exp = (REJECT_CORRUPT if mode == "patchfrom" else
                   REJECT if mode in EXPECT_REJECT_MODES else "decode")
            if mode in ("streaming", "nocontent") and frame_has_fcs(cbytes):
                exp = "decode"     # this CLI wrote an FCS after all
            entries.append(_entry(version, mode, inkey, cbytes, exp,
                                  decoded=decoded, dsize=dsize, dsha=dsha,
                                  name=name, relfile=relfile))
            log("  %s/%s: %d bytes -> %s" % (version, mode, len(cbytes), relfile))

    entries += gen_bshuf(log)
    manifest = {"schema": "slabitz-compat-corpus-v1", "versions": VERSIONS,
                "modes": [m[0] for m in MODES], "entries": entries}
    (FULL_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log("full corpus: %d entries -> %s" % (len(entries), FULL_DIR))
    return entries


def _entry(version, mode, inkey, cbytes, expectation, decoded=None, dsize=None,
           dsha=None, name=None, relfile=None, kind="zstd", fcs=None, bshuf=None):
    e = {"version": version, "mode": mode, "input": inkey, "kind": kind,
         "expectation": expectation}
    if name is not None:
        e["name"] = name
    if relfile is not None:
        e["file"] = relfile
    if cbytes is not None:
        e["compressed_size"] = len(cbytes)
    if dsha is not None:
        e["decoded_sha256"] = dsha
    if dsize is not None:
        e["decoded_size"] = dsize
    if decoded is not None:
        e["decoded_sha256"] = sha256(decoded)
        e["decoded_size"] = len(decoded)
    if fcs is not None:
        e["fcs"] = fcs
    if bshuf is not None:
        e["bshuf"] = bshuf
    return e


def gen_bshuf(log):
    """Generate the bitshuffle-0.4.0 zstd chunks (filter 32008).  Returns entries."""
    import ctypes as ct
    import struct
    import numpy as np
    lib = ct.CDLL(str(BSHUF_LIB))
    lib.bshuf_compress_zstd_bound.restype = ct.c_size_t
    lib.bshuf_compress_zstd_bound.argtypes = [ct.c_size_t, ct.c_size_t, ct.c_size_t]
    lib.bshuf_compress_zstd.restype = ct.c_int64
    lib.bshuf_compress_zstd.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t,
                                         ct.c_size_t, ct.c_size_t, ct.c_int]

    bdir = FULL_DIR / "bshuf040"
    bdir.mkdir(parents=True, exist_ok=True)
    out = []

    def default_block_elems(elem):
        # bitshuffle 0.4.0 bshuf_default_block_size: 8192 bytes / elem,
        # rounded down to a multiple of 8, min 128.
        return max((8192 // elem) // 8 * 8, 128)

    def chunk_zu(data, elem, block, level):
        # block: element count (0 -> bshuf default).  The chunk header records
        # block*elem bytes (the resolved default when block == 0), exactly as
        # bshuf_h5filter.c writes it.
        be = block if block else default_block_elems(elem)
        n = len(data) // elem
        bound = lib.bshuf_compress_zstd_bound(n, elem, be)
        inb = ct.create_string_buffer(data, len(data))
        outb = ct.create_string_buffer(bound)
        m = lib.bshuf_compress_zstd(inb, outb, n, elem, be, level)
        if m < 0:
            raise RuntimeError("bshuf_compress_zstd failed: %d" % m)
        hdr = struct.pack(">Q", len(data)) + struct.pack(">I", be * elem)
        return hdr + outb.raw[:m]

    try:
        det_u16 = crop_detector("silicon_u16_zstd2_256k.h5", "uint16", 512)
        det_u32 = crop_detector("ceo2_u32_zstd2_256k.h5", "uint32", 512)
    except Exception as e:
        log("  bshuf040: detector crop failed: %s (subset still generated)" % e)
        det_u16 = b"A" * (512 * 2162 * 2)
        det_u32 = b"B" * (512 * 2162 * 4)

    specs = [
        ("bshuf040_silicon_u16_default", det_u16, 2, 0, 2),
        ("bshuf040_silicon_u16_256k",    det_u16, 2, 131072, 2),
        ("bshuf040_ceo2_u32_default",    det_u32, 4, 0, 2),
        ("bshuf040_ceo2_u32_64k",        det_u32, 4, 65536, 2),
    ]
    for name, data, elem, block, level in specs:
        n = len(data) // elem
        chunk = chunk_zu(data, elem, block, level)
        cd = [0, 4, elem, block, 3, level]
        rel = "bshuf040/%s.chunk" % name
        (FULL_DIR / rel).parent.mkdir(parents=True, exist_ok=True)
        (FULL_DIR / rel).write_bytes(chunk)
        out.append({
            "version": "bshuf040", "mode": "bshuf040", "input": name,
            "kind": "bshuf", "expectation": "decode", "name": name,
            "compressed_size": len(chunk),
            "decoded_sha256": sha256(data), "decoded_size": len(data),
            "bshuf": {"elem": elem, "block": block, "cd_values": cd,
                      "zstd_version": BSHUF_ZSTD},
            "file": rel,
        })
        log("  bshuf040 %s: %d bytes (block %d)" % (name, len(chunk), block))
    return out


# ---- subset ----------------------------------------------------------------
def gen_subset(log, full_entries):
    subset_dir = SUBSET_DIR
    subset_dir.mkdir(parents=True, exist_ok=True)
    relmap = {}
    for e in full_entries:
        relmap[(e["version"], e["name"])] = e

    chosen = {}
    for version in VERSIONS:
        # text-based levels + a detector-input file per level
        for name in ("level3_text.zst", "level19_text.zst",
                     "level3_det_u16.zst", "level19_det_u16.zst"):
            e = relmap.get((version, name))
            if e and e["expectation"] == "decode":
                chosen["%s/%s" % (version, name)] = e
        # no-FCS and window-30 expected-reject cases
        for name in ("streaming_text.zst", "long30_long30.zst"):
            e = relmap.get((version, name))
            if e and e["expectation"].startswith("reject:"):
                chosen["%s/%s" % (version, name)] = e

    for e in full_entries:
        if e["kind"] == "bshuf":
            chosen["bshuf/%s" % e["name"]] = e

    sub_entries = []
    total = 0
    for key, e in chosen.items():
        src = FULL_DIR / e["file"]
        if not src.exists():
            log("  subset: MISSING %s" % src)
            continue
        data = src.read_bytes()
        total += len(data)
        rel = e.get("file", os.path.basename(src))
        base = os.path.basename(rel)
        dest = subset_dir / ("%s--%s" % (e["version"], base))
        dest.write_bytes(data)
        ne = dict(e)
        ne["file"] = dest.name
        sub_entries.append(ne)

    if total > 5 * 1024 * 1024:
        log("  SUBSET TOO BIG: %d bytes (> 5 MB)" % total)
    manifest = {"schema": "slabitz-compat-corpus-v1", "versions": VERSIONS,
                "modes": [m[0] for m in MODES],
                "subset_of": str(FULL_DIR), "entries": sub_entries}
    (subset_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log("subset: %d entries, %d bytes -> %s" % (len(sub_entries), total, subset_dir))
    return sub_entries


# ---- CLI -------------------------------------------------------------------
def cmd_list(log):
    print("zstd corpus versions (bitshuffle 0.4.0 -> 1.6.0):")
    for v in VERSIONS:
        mark = "  (bitshuffle 0.4.0's bundled snapshot)" if v == BSHUF_ZSTD else ""
        print("  %s%s" % (v, mark))
    print("modes (on the text input; see below for extra inputs):")
    for mode, args, inkey, special in MODES:
        what = " ".join(args)
        if special:
            what = "--patch-from=<base>" if mode == "patchfrom" else "-D <dict>"
        print("  %-11s %s  (input: %s)" % (mode, what, inkey))
    print("extra inputs:")
    for mode, args, inkey in EXTRA_TASKS:
        print("  %-11s %s  (input: %s)" % (mode, " ".join(args), inkey))
    print("  bitshuffle  filter-32008 chunks (zstd %s)" % BSHUF_ZSTD)


def cmd_build(log):
    t0 = time.time()
    failed = []
    for v in VERSIONS:
        t = time.time()
        ok, reason = build_zstd(v)
        log("  %-9s %s (%5.1fs) %s" % (v, "OK" if ok else "FAILED",
                                       time.time() - t, reason))
        if not ok:
            failed.append((v, reason))
    t = time.time()
    ok, reason = build_bshuf()
    log("  %-9s %s (%5.1fs) %s" % ("bshuf040", "OK" if ok else "FAILED",
                                   time.time() - t, reason))
    if not ok:
        failed.append(("bshuf040", reason))
    log("build time: %.1fs total" % (time.time() - t0))
    if failed:
        log("BUILD FAILURES:")
        for v, r in failed:
            log("  %s: %s" % (v, r))
        return 1
    log("all built")
    return 0


def cmd_generate(log, only):
    if only in ("full", None):
        gene_full(log)
    if only in ("subset", None):
        # subset needs the full corpus on disk; (re)read its manifest
        full = json.loads((FULL_DIR / "manifest.json").read_text())
        gen_subset(log, full["entries"])
    return 0


def gene_full(log):
    full = gen_full(log)
    (FULL_DIR / "manifest.json").write_text(
        json.dumps({"schema": "slabitz-compat-corpus-v1",
                    "versions": VERSIONS, "modes": [m[0] for m in MODES],
                    "entries": full}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--subset-only", action="store_true")
    ap.add_argument("--full-only", action="store_true")
    args = ap.parse_args()

    def log(msg):
        print(msg)

    if not (args.list or args.build or args.generate):
        ap.print_help()
        return 1
    if args.list:
        cmd_list(log)
        return 0
    if args.build:
        return cmd_build(log)
    if args.generate:
        if not all(build_ok(v) for v in VERSIONS):
            log("ERROR: some zstd CLI missing; run --build first")
            return 1
        if not BSHUF_LIB.exists():
            log("ERROR: %s not built; run --build first" % BSHUF_LIB)
            return 1
        only = "subset" if args.subset_only else ("full" if args.full_only else None)
        return cmd_generate(log, only)
    return 1


if __name__ == "__main__":
    sys.exit(main())
