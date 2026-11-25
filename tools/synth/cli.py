"""T00 synthetic-data CLI: fit, generate, compare, worst, subset.

Usage:  python3 -m tools.synth <command> [args]

Commands
  fit        <key>            measure P1 + gaps from a real file -> tools/synth/params/<key>.json
  generate   <key> [--frames N] [--seed S]  build the twin from the params
  compare    [--all] [--key K] [--frames N]  P1/P3/P4: real vs twin, tolerances
  worst      [--dir DIR]      build the named worst cases
  subset     [--dir DIR]      build tests/data/synth (small committed subset)
"""

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import sys

import numpy as np

from . import encode, fit as fit_mod, geometry, metrics, models

ROOT = os.path.dirname(os.path.abspath(__file__))  # tools/synth
_TOOLS = os.path.dirname(ROOT)                      # tools
_WT = os.path.dirname(_TOOLS)                       # worktree = .../wt/t00-synthetic-data
_DEV_ROOT = os.path.dirname(os.path.dirname(_WT))   # .../slabitz
EXAMPLES = os.environ.get("SLABITZ_EXAMPLES", os.path.join(_DEV_ROOT, "data", "real"))
SYNTH_ROOT = os.environ.get("SLABITZ_SYNTH", os.path.join(_DEV_ROOT, "data", "synth", "current"))
PAC_LIB = os.environ.get("PAC_LIB", os.path.join(_DEV_ROOT, "ext", "pac-zstd", "build", "libpac.so"))
PARAMS_DIR = os.path.join(ROOT, "params")

_DTI = {"u8": 1, "u16": 2, "u32": 3, "u64": 4}

# key -> primary real file (compare reference), dtype, model, geometry, gap_kind
KEYS = {
    "silicon": {"real": "silicon_lz4.h5", "ds": "entry_0000/ESRF-ID11/eiger/data",
                "dtype": "u2", "model": "spots", "geom": "eiger4m", "gap_kind": "zero",
                "encodings": [("lz4_8k", "zero"), ("lz4_8k", "max"), ("zstd2_256k", "zero")],
                "params": {"n_spots": 2600, "spot_sigma": 2.2, "intensity": 45.0,
                           "max_intensity": 9000.0, "alpha": 2.6}},
    "ceo2": {"real": "ceo2_lz4.h5", "ds": "entry_0000/ESRF-ID11/eiger/data",
             "dtype": "u4", "model": "rings", "geom": "eiger4m", "gap_kind": "zero",
             "encodings": [("lz4_8k", "zero"), ("zstd2_256k", "zero")],
             "params": {"n_rings": 8, "radii": [180, 320, 520, 700, 900, 1100, 1300, 1520],
                        "widths": [22, 30, 26, 34, 30, 38, 30, 32],
                        "heights": [90, 60, 220, 90, 140, 70, 100, 60], "az": 0.0}},
    "thaumatin": {"real": "thaumatin_9_1_000001.h5", "ds": "data", "dtype": "u2",
                  "model": "mx", "geom": "eiger2_16m", "gap_kind": "zero",
                  "encodings": [("lz4_8k", "zero")],
                  "params": {"halo_peak": 4000.0, "halo_radius": 1400.0, "halo_width": 300.0,
                             "n_spots": 4000, "spot_sigma": 2.5, "intensity": 800.0,
                             "max_intensity": 60000.0, "concentrate": True}},
    "eiger4m_u32": {"real": "eiger4m_u32.h5", "ds": "entry_0000/ESRF-ID11/eiger/data",
                    "dtype": "u4", "model": "mx", "geom": "eiger4m", "gap_kind": "zero",
                    "encodings": [("lz4_8k", "zero")],
                    "params": {"halo_peak": 3000.0, "halo_radius": 900.0, "halo_width": 260.0,
                               "n_spots": 3000, "spot_sigma": 3.0, "intensity": 1200.0,
                               "max_intensity": 70000.0, "concentrate": True}},
    "big_eiger": {"real": "big_eiger_0000.h5", "ds": "entry_0000/ESRF-ID11/eiger/data",
                  "dtype": "u2", "model": "mx", "geom": "eiger4m", "gap_kind": "zero",
                  "encodings": [("lz4_8k", "zero")],
                  "params": {"halo_peak": 3000.0, "halo_radius": 900.0, "halo_width": 260.0,
                             "n_spots": 3000, "spot_sigma": 3.0, "intensity": 1200.0,
                             "max_intensity": 70000.0, "concentrate": True}},
    "kevlar": {"real": "kevlar.h5", "ds": "entry/data/data", "dtype": "u2", "model": "fibre",
               "geom": "eiger1_4m", "gap_kind": "zero", "encodings": [("lz4_8k", "zero")],
               "params": {"arc_radius": 700.0, "arc_step": 250.0, "arc_width": 60.0,
                          "arc_intensity": 1500.0, "arcs": 4, "n_spots": 2000,
                          "intensity": 800.0, "concentrate": True}},
    "hplc": {"real": "hplc.h5", "ds": "entry_0000/instrument/Pilatus/data", "dtype": "i4",
             "model": "rings", "geom": "pilatus", "gap_kind": "negative", "neg_mask": 0.002,
             "encodings": [("lz4_8k", "zero")],
             "params": {"n_rings": 5, "radii": [120, 300, 520, 760, 980],
                        "widths": [18, 26, 30, 26, 30], "heights": [60, 90, 140, 80, 60],
                        "az": 0.0}},
    "al2o3": {"real": "Calibration_Al2O3.h5", "ds": "data", "dtype": "u4", "model": "rings",
              "geom": "eiger1_4m", "gap_kind": "zero", "encodings": [("lz4_8k", "zero")],
              "params": {"n_rings": 6, "radii": [150, 330, 520, 720, 900, 1100],
                         "widths": [20, 24, 28, 30, 26, 28],
                         "heights": [40, 60, 120, 70, 50, 40], "az": 0.0}},
}


def _dtype(d):
    return np.dtype(d)


def real_path(key):
    return os.path.join(EXAMPLES, KEYS[key]["real"])


# ------------------------------------------------------------------ generate
def _intensity_mean(model, params, shape, seed):
    """Mean of the intensity image averaged over several draws (the spot field
    is stochastic, so a single draw is noisy; the average is what the Poisson
    mean of a frame tends to)."""
    acc = 0.0
    for k in range(8):
        rng = np.random.default_rng(seed + k * 101 + 5)
        img = models.MODELS[model](shape, dict(params), rng)
        acc += float(img.mean())
    return acc / 8.0, img


def _make_frames(params, n, seed):
    """Generate `n` raw frames calibrated so the mean matches the target."""
    shape = tuple(params["shape"])
    dtype = params["dtype"]
    model = params["model"]
    p = dict(params.get("parameters", {}))
    target_mean = params["targets"]["mean"]
    # calibrate intensity scale: E[Poisson(c*I)] = c*E[I]
    m, _ = _intensity_mean(model, p, shape, seed)
    c = (target_mean / m) if m > 0 else 1.0
    params = dict(params)
    # concentrate the signal into the small nonzero fraction only for the
    # sparse-BRIGHT keys (thaumatin/eiger/kevlar); sparse-low and dense keys
    # use the ordinary Poisson path.
    params["nz_target"] = params["targets"].get("nonzero_frac", 1.0) if params.get("concentrate") else 1.0
    return _apply_scale(model, p, shape, dtype, c, params, n, seed)


def _apply_scale(model, p, shape, dtype, c, params, n, seed):
    nz_target = float(params.get("nz_target", 1.0))
    target_mean = float(params.get("targets", {}).get("mean", None) or 0.0)
    frames = []
    for i in range(n):
        rng = np.random.default_rng(seed + i * 7919 + 1)
        img = models.MODELS[model](shape, dict(p), rng)
        if target_mean and nz_target < 1.0:
            # concentrate the signal into the top nz_target fraction, then
            # scale so the whole-frame mean hits the target.  This reproduces
            # the sparse-bright real frames (mostly exactly 0, signal only in a
            # small fraction).
            t = np.quantile(img, 1.0 - nz_target)
            img = np.where(img >= t, img, 0.0)
            s = float(img.sum())
            if s > 0:
                img = img * (target_mean * img.size / s)
        else:
            img = img * c
        dt = np.dtype(dtype)
        info = np.iinfo(dt)
        counts = rng.poisson(np.clip(img, 0, None))
        counts = np.clip(counts, info.min, info.max)
        frame = counts.astype(dt)
        gap_value = geometry.gap_value(params.get("gap_kind", "zero"), dtype)
        if params.get("gap_rows"):
            frame[list(params["gap_rows"]), :] = gap_value
        if params.get("gap_cols"):
            frame[:, list(params["gap_cols"])] = gap_value
        if params.get("neg_mask"):
            nf = int(shape[0] * shape[1] * params["neg_mask"])
            ys = rng.integers(0, shape[0], nf)
            xs = rng.integers(0, shape[1], nf)
            frame[ys, xs] = np.where(rng.random(nf) < 0.5, -1, -2).astype(dt)
        frames.append(frame)
    return np.stack(frames)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cmd_generate(args):
    key = args.key
    kd = KEYS[key]
    params_path = os.path.join(PARAMS_DIR, key + ".json")
    if not os.path.exists(params_path):
        print("no params for %s (run fit first)" % key, file=sys.stderr)
        return 1
    params = json.load(open(params_path))
    params["concentrate"] = kd.get("concentrate", kd.get("params", {}).get("concentrate", False))
    n = args.frames
    seed = args.seed
    frames = _make_frames(params, n, seed)
    outdir = os.path.join(SYNTH_ROOT, "twins", key)
    os.makedirs(outdir, exist_ok=True)
    rawname = os.path.join(outdir, "raw.npy")
    np.save(rawname, frames)
    man = {"seed": seed, "frames": n, "version": 1,
           "params": params, "files": {rawname: sha256_file(rawname)}}
    for enc, gk in kd["encodings"]:
        name = "%s.%s.h5" % (enc, gk)
        outp = os.path.join(outdir, name)
        encode.write_h5(frames, params["dtype"], outp, enc, gap_kind=gk,
                        gap_cols=params.get("gap_cols", ()),
                        gap_rows=params.get("gap_rows", ()),
                        gap_value=None if gk != "max" else geometry.gap_value("max", params["dtype"]))
        man["files"][outp] = sha256_file(outp)
    with open(os.path.join(outdir, "manifest.json"), "w") as fh:
        json.dump(man, fh, indent=2, sort_keys=True)
    print("wrote", outdir)
    return 0


# ------------------------------------------------------------------- compare
def _real_frames(key, n):
    kd = KEYS[key]
    frames = fit_mod.read_frames(real_path(key), kd["ds"], n)
    return np.stack(frames), kd


def _twin_frames(key, n):
    p = os.path.join(SYNTH_ROOT, "twins", key, "raw.npy")
    arr = np.load(p)
    return arr[:min(n, arr.shape[0])]


def cmd_compare(args):
    keys = list(KEYS) if args.all else [args.key]
    passed, failed = [], []
    for key in keys:
        n = args.frames
        real, kd = _real_frames(key, n)
        try:
            twin = _twin_frames(key, n)
        except OSError:
            print("%-10s not run (no twin; generate first)" % key)
            failed.append(key)
            continue
        print("\n=== %s ===" % key)
        r = compare_key(key, real, twin, kd, args)
        (passed if r is None else failed).append(key)
    print("\n== SUMMARY ==")
    print("  pass:", ", ".join(passed) if passed else "(none)")
    print("  fail:", ", ".join(failed) if failed else "(none)")
    return 1 if failed else 0


def compare_key(key, real, twin, kd, args):
    dtype = real.dtype
    gk = kd["gap_kind"]
    gv = geometry.gap_value("zero" if gk != "negative" else "zero", dtype)
    p_real = metrics.compute_P1(real, dtype, masked_value=geometry.gap_value("zero", dtype))
    p_twin = metrics.compute_P1(twin, dtype, masked_value=geometry.gap_value("zero", dtype))
    print("  P1 real: mean %.4g var %.4g nz %.4f | twin: mean %.4g var %.4g nz %.4f"
          % (p_real["mean"], p_real["var"], p_real["nonzero_frac"],
             p_twin["mean"], p_twin["var"], p_twin["nonzero_frac"]))
    js = metrics.js_distance(p_real["hist"], p_twin["hist"])
    # tolerance checks
    ok = True
    def chk(label, got, want, rel_tol=None, abs_tol=None):
        nonlocal ok
        if rel_tol is not None:
            denom = max(abs(want), 1e-9)
            err = abs(got - want) / denom
            passed = err <= rel_tol
        else:
            err = abs(got - want)
            passed = err <= abs_tol
        ok = ok and passed
        print("    %-26s got %.5g want %.5g %s (%s)" % (label, got, want,
              "PASS" if passed else "FAIL", "%.1f%%" % (err * 100) if rel_tol is not None else "%.3f" % err))
        return passed
    chk("P1 nonzero_frac", p_twin["nonzero_frac"], p_real["nonzero_frac"], rel_tol=0.10)
    chk("P1 masked_frac", p_twin["masked_frac"], p_real["masked_frac"], rel_tol=0.10)
    chk("P1 mean", p_twin["mean"], p_real["mean"], rel_tol=0.05)
    print("    %-26s got %.5f want <0.05  %s" % ("P1 JSD", js, "PASS" if js <= 0.05 else "FAIL"))
    ok = ok and js <= 0.05
    # P4 ratio per encoder
    encs = args.enc if args.enc else ["lz4_8k", "zstd2_256k", "zstd3_256k", "pac_lz4", "pac_zstd"]
    for name in encs:
        enc = metrics.ENCODERS[name]
        try:
            rr = metrics.compression_ratio(real.astype(dtype), dtype, enc, PAC_LIB)
            rt = metrics.compression_ratio(twin.astype(dtype), dtype, enc, PAC_LIB)
            chk("P4 ratio %-10s" % name, rt, rr, rel_tol=0.10)
        except Exception as e:
            print("  P4 %-10s not run: %s" % (name, e))
    return None if ok else key


# --------------------------------------------------------------------- worst
_worst = {}


def _zstd_compress(data, level=3):
    import subprocess, tempfile
    td = tempfile.mkdtemp(prefix="sbz_z_")
    inp = os.path.join(td, "i"); out = os.path.join(td, "o")
    with open(inp, "wb") as f:
        f.write(data)
    subprocess.run(["/bin/zstd", "-%d" % level, "-q", "-f", inp, "-o", out], check=True)
    with open(out, "rb") as f:
        return f.read()


class _WorstBuilder:
    def __init__(self, root):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def put(self, case, raw, zst=None, h5=None, **meta):
        d = os.path.join(self.root, case)
        os.makedirs(d, exist_ok=True)
        rp = os.path.join(d, "raw.bin")
        with open(rp, "wb") as f:
            f.write(raw)
        files = {"raw.bin": sha256_file(rp)}
        if zst:
            zp = os.path.join(d, "data.zst")
            with open(zp, "wb") as f:
                f.write(zst)
            files["data.zst"] = sha256_file(zp)
        if h5:
            files["data.h5"] = sha256_file(h5)
        man = {"case": case, "files": files}
        man.update(meta)
        with open(os.path.join(d, "manifest.json"), "w") as fh:
            json.dump(man, fh, indent=2, sort_keys=True)


def _worst_case(name, b, raw, meta):
    z = _zstd_compress(raw)
    mix = metrics.zstd_block_mix(z)
    meta["zstd_block_mix"] = mix
    meta["zstd_raw_ratio"] = len(raw) / len(z) if z else float("nan")
    b.put(name, raw, zst=z, **meta)


def cmd_worst(args):
    b = _WorstBuilder(args.dir)
    print("worst cases ->", args.dir)
    # Incompressible: all raw blocks (verified by the block-type walker).
    for dt in ["u8", "u16", "u32", "u64"]:
        raw = np.random.default_rng(7 + _DTI[dt]).bytes(1 << 16)
        _worst_case("random_%s" % dt, b, raw, {"claim": "all raw blocks (incompressible)",
                                               "property": "zstd_block_mix.raw == 1"})
    z = bytes(1 << 16)
    _worst_case("zeros", b, z, {"claim": "compresses to a tiny frame (RLE/compressed)",
                                "property": "len(zst) << 64KiB"})
    _worst_case("all_max", b, bytes([255]) * (1 << 16),
                {"claim": "masked-detector extreme (every byte 2^n-1)",
                 "property": "zstd_raw_ratio < 1e-3"})
    hot = np.zeros((1 << 16,), dtype=np.uint8); hot[0] = 255
    _worst_case("single_hot_pixel", b, hot.tobytes(),
                {"claim": "one nonzero byte per frame", "property": "zstd_raw_ratio < 1e-3"})
    # Tiny / many frames (decoder tail handling).
    _worst_case("tiny_frames", b, np.arange(7, dtype=np.uint8).tobytes(),
                {"claim": "frames of 7 elements (tail only)", "property": "len(raw) == 7"})
    _worst_case("many_frames", b, np.arange(8 * 1000, dtype=np.uint8).tobytes(),
                {"claim": "1000 frames of 8 elements in one call",
                 "property": "len(raw) == 8000"})
    # A large Eiger2 16M u32 high dynamic range frame.
    if os.environ.get("SBZ_WORST_LARGE", "1") != "0":
        shape = (4362, 4148)
        rng = np.random.default_rng(11)
        img = rng.poisson(rng.random(shape) * 40000.0).astype(np.uint32)
        raw = img.tobytes()
        _worst_case("large_frame_16m_u32", b, raw,
                    {"claim": "one high-dynamic-range u32 16M frame (large)",
                     "property": "len(raw) == %d" % (shape[0] * shape[1] * 4)})
    print("not built (need libzstd/DS-specific construction): min_matches, short_offsets_1_31,\n"
          "      far_offsets, huffman_max_depth, literals_128k, fse_max_accuracy,\n"
          "      alternating_planes, exact_128k_block, block_tail_mix, zero_seq_blocks")
    return 0


# -------------------------------------------------------------------- subset
def cmd_subset(args):
    out = os.path.join(_WT, "tests", "data", "synth")
    os.makedirs(out, exist_ok=True)
    # one cropped frame per model, encoded LZ4 8k and zstd 256k
    models_keys = {"spots": "silicon", "rings": "ceo2", "mx": "thaumatin",
                   "fibre": "kevlar", "flat": "ceo2"}
    crop = args.crop
    for model, key in models_keys.items():
        src = os.path.join(SYNTH_ROOT, "twins", key, "raw.npy")
        if not os.path.exists(src):
            print("skip %s (no twin %s)" % (model, key))
            continue
        frames = np.load(src)
        dtype = frames.dtype
        fr = frames[0][:crop, :crop]
        fr = np.ascontiguousarray(fr)
        arr = fr[None, :, :]
        for enc in ["lz4_8k", "zstd2_256k"]:
            outp = os.path.join(out, "%s.%s.h5" % (model, enc.split("_")[0]))
            encode.write_h5(arr, dtype, outp, enc, gap_kind="zero")
    # small worst cases (the .zst frames) straight over, LZ4+zstd re-encodes of the raw
    worst_dir = os.path.join(SYNTH_ROOT, "worst")
    for case in os.listdir(worst_dir):
        mp = os.path.join(worst_dir, case, "manifest.json")
        if not os.path.exists(mp):
            continue
        man = json.load(open(mp))
        rp = os.path.join(worst_dir, case, "raw.bin")
        if not os.path.exists(rp):
            continue
        raw = open(rp, "rb").read()
        if len(raw) > (1 << 17):  # skip the large frame
            continue
        z = _zstd_compress(raw)
        with open(os.path.join(out, "worst_%s.zst" % case), "wb") as f:
            f.write(z)
    total = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out))
    print("subset ->", out, "(%d files, %.3f MB)" % (len(os.listdir(out)), total / 1e6))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tools.synth")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fit"); p.add_argument("key"); p.set_defaults(fn=cmd_fit)
    p = sub.add_parser("generate"); p.add_argument("key"); p.add_argument("--frames", type=int, default=8)
    p.add_argument("--seed", type=int, default=4213); p.set_defaults(fn=cmd_generate)
    p = sub.add_parser("compare"); p.add_argument("--all", action="store_true"); p.add_argument("--key")
    p.add_argument("--frames", type=int, default=8); p.add_argument("--enc", action="append"); p.set_defaults(fn=cmd_compare)
    p = sub.add_parser("worst"); p.add_argument("--dir", default=os.path.join(SYNTH_ROOT, "worst")); p.set_defaults(fn=cmd_worst)
    p = sub.add_parser("subset"); p.add_argument("--crop", type=int, default=256); p.set_defaults(fn=cmd_subset)
    a = ap.parse_args(argv)
    return a.fn(a)


def cmd_fit(args):
    kd = dict(KEYS[args.key])
    kd["real_path"] = real_path(args.key)
    kd["dsname"] = kd["ds"]
    out = os.path.join(PARAMS_DIR, args.key + ".json")
    p = fit_mod.fit(kd, out, n=8)
    print("fit %s -> %s  mean %.4g nz %.4f" % (args.key, out, p["targets"]["mean"], p["targets"]["nonzero_frac"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
