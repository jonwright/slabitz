"""T00 synthetic data: python3 -m tools.synth <command>

  fit [KEY ...]            measure real files -> tools/synth/params/<key>.{json,npz}
  generate [KEY ...]       build twins into $SLABITZ_SYNTH/twins/<key>/
  compare [--key K | --all]  P1 / P3 / P4, real against twin, with tolerances
  ext                      extension set -> $SLABITZ_SYNTH/ext/
  worst                    named worst cases -> $SLABITZ_SYNTH/worst/
  check-worst              the claimed property of each worst case, measured
  subset                   the committed subset -> tests/data/synth/
  determinism              generate twice, compare sha256 of every file
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

from . import encode, fit as fit_mod, keys as K, metrics, twin

TWIN_FRAMES = 16
SEED = 4213


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def synth_root(args):
    return getattr(args, "out", None) or K.SYNTH_ROOT


# ------------------------------------------------------------------- fit
def cmd_fit(args):
    for key in args.keys or list(K.KEYS):
        fit_mod.fit(key)
    return 0


# -------------------------------------------------------------- generate
def write_twin(key, outdir, nframes=TWIN_FRAMES, seed=SEED):
    params, arrays = twin.load_params(key, K.PARAMS_DIR)
    fr = twin.frames(params, arrays, nframes, seed, mask_kind="native")
    os.makedirs(outdir, exist_ok=True)
    files = {}
    raw = os.path.join(outdir, "raw.npy")
    np.save(raw, fr)
    files["raw.npy"] = sha256_file(raw)
    kinds = [("native", "")]
    if params["mask_values"] == [0]:
        # these detectors' gaps are 0 in the plain files and 2**n-1 in the mask
        # variants (silicon_lz4mask, ceo2_lz4mask): write both
        kinds = [("native", ".mask0"), ("max", ".maskmax")]
    for enc in K.KEYS[key]["enc"]:
        for kind, suffix in kinds:
            f2 = fr if kind == "native" else twin.frames(params, arrays, nframes, seed, mask_kind=kind)
            name = "%s%s.h5" % (enc, suffix)
            path = os.path.join(outdir, name)
            encode.write_h5(f2, path, enc)
            files[name] = sha256_file(path)
    man = {"key": key, "generator_version": twin.VERSION, "seed": seed, "frames": nframes,
           "numpy": np.__version__, "params": params, "files": files}
    with open(os.path.join(outdir, "manifest.json"), "w") as fh:
        json.dump(man, fh, indent=1, sort_keys=True)
        fh.write("\n")
    return files


def cmd_generate(args):
    root = synth_root(args)
    for key in args.keys or list(K.KEYS):
        files = write_twin(key, os.path.join(root, "twins", key), args.frames, args.seed)
        print("twin %-12s %d files" % (key, len(files)))
    return 0


# --------------------------------------------------------------- compare
TOL_CLASS_REL = 0.10
TOL_CLASS_ABS = 2e-4          # classes this rare: absolute tolerance
TOL_MEAN = 0.05
TOL_JSD = 0.05
TOL_P3 = 0.05
TOL_P4_RATIO = 0.10
TOL_P4_FRAC = 0.05


class Report:
    def __init__(self):
        self.rows = []

    def check(self, label, got, want, rel=None, abs_=None):
        err = abs(got - want)
        ok = True
        if rel is not None and abs_ is not None:
            ok = err <= max(rel * abs(want), abs_)
        elif rel is not None:
            ok = err <= rel * abs(want)
        else:
            ok = err <= abs_
        self.rows.append((label, got, want, ok))
        print("  %-30s twin %-12.5g real %-12.5g %s" % (label, got, want, "PASS" if ok else "FAIL"))
        return ok

    def ok(self):
        return all(r[3] for r in self.rows)


def real_and_twin(key, root, perturb=False):
    params, arrays = twin.load_params(key, K.PARAMS_DIR)
    fr, _ = fit_mod.read_frames(K.real_path(key), K.KEYS[key]["ds"], params["frames_used"])
    if perturb:
        # reviewer check: twice the pattern density (every expected count and the hot pixels)
        p2 = dict(params, hot_per_frame=2 * params["hot_per_frame"])
        tw = twin.frames(p2, arrays, len(fr), SEED, lam_scale=2.0)
    else:
        tw = np.load(os.path.join(root, "twins", key, "raw.npy"), mmap_mode="r")[:len(fr)]
    return params, arrays, np.stack(fr), np.asarray(tw)


def compare_key(key, root, enc=None, quick=False, perturb=False):
    params, arrays, real, tw = real_and_twin(key, root, perturb)
    idx = arrays["mask_idx"]
    mvals = params["mask_values"]
    print("\n=== %s: %d frames %s %s ===" % (key, len(real), real.shape[1:], real.dtype))
    rep = Report()
    pr = metrics.p1(real, idx, mvals, params["clip"])
    pt = metrics.p1(tw, idx, mvals, params["clip"])
    for name, _, _ in metrics.CLASSES:
        rep.check("P1 class %s" % name, pt["class_frac"][name], pr["class_frac"][name],
                  rel=TOL_CLASS_REL, abs_=TOL_CLASS_ABS)
    rep.check("P1 mean up to clip %g" % params["clip"], pt["mean_cold"], pr["mean_cold"], rel=TOL_MEAN,
              abs_=1e-4)
    rep.check("P1 hot fraction", pt["hot_frac"], pr["hot_frac"], rel=0.5, abs_=2e-5)
    rep.check("P1 masked fraction", pt["masked_frac"], pr["masked_frac"], rel=TOL_CLASS_REL)
    js = metrics.js_distance(pr["hist"], pt["hist"])
    rep.check("P1 JSD", js, 0.0, abs_=TOL_JSD)
    print("  (live mean incl. hot: twin %.4g real %.4g; not a check: a few pixels dominate it)"
          % (pt["live_mean"], pr["live_mean"]))
    if quick:
        return rep
    for bb in (8192, 262144):
        if real.dtype.itemsize * 8 > bb:
            continue
        a = metrics.p3(real, bb)
        b = metrics.p3(tw, bb)
        rep.check("P3 zero planes %dk" % (bb // 1024), b["zero_frac"], a["zero_frac"], abs_=TOL_P3)
        rep.check("P3 eq-prev planes %dk" % (bb // 1024), b["eq_prev_frac"], a["eq_prev_frac"],
                  abs_=TOL_P3)
    for name in (enc or ["lz4_8k", "zstd2_256k", "zstd3_256k", "pac_lz4", "pac_zstd"]):
        try:
            a = metrics.p4(real, name, K.PAC_LIB)
            b = metrics.p4(tw, name, K.PAC_LIB)
        except Exception as e:                 # a check that cannot run is "not run"
            print("  P4 %-12s not run: %s" % (name, e))
            continue
        rep.check("P4 ratio %s" % name, b["ratio"], a["ratio"], rel=TOL_P4_RATIO)
        if "seq" in a:
            for f in ("literal", "off1", "offP"):
                rep.check("P4 %s frac %s" % (f, name), b["seq"][f], a["seq"][f], abs_=TOL_P4_FRAC)
            print("  P4 block mix %s: twin %s real %s" % (name, b["block_mix"], a["block_mix"]))
    return rep


def cmd_compare(args):
    root = synth_root(args)
    keys = list(K.KEYS) if args.all else [args.key]
    results = {}
    for key in keys:
        results[key] = compare_key(key, root, args.enc, args.quick, args.perturb).ok()
    print("\n== SUMMARY ==")
    for k, v in results.items():
        print("  %-12s %s" % (k, "pass" if v else "FAIL"))
    return 0 if all(results.values()) else 1


# ------------------------------------------------------------- ext, worst
def cmd_ext(args):
    from . import ext
    root = os.path.join(synth_root(args), "ext")
    n = 0
    for spec in ext.specs():
        if args.only and args.only not in spec["name"]:
            continue
        files = ext.write(spec, root)
        n += 1
        print("ext %-34s %d files" % (spec["name"], len(files)))
    return 0


def _cases(with_large=True):
    from . import worst
    cases = worst.build_cases()
    if with_large:
        cases.append(worst.large_case())
    return worst, cases


def print_table(rows):
    w = max(len(r[0]) for r in rows)
    print("%-*s  %-4s  %s" % (w, "case", "pass", "claimed property: measured"))
    for name, claim, meas, ok in rows:
        print("%-*s  %-4s  %s: %s" % (w, name, "PASS" if ok else "FAIL", claim, meas))
    bad = [r[0] for r in rows if not r[3]]
    print("\n%d cases, %d failed%s" % (len(rows), len(bad), (": " + ", ".join(bad)) if bad else ""))
    return not bad


def cmd_worst(args):
    worst, cases = _cases(not args.no_large)
    root = os.path.join(synth_root(args), "worst")
    names = worst.write(cases, root)
    print("wrote %d cases to %s" % (len(names), root))
    return 0 if print_table(worst.check(cases)) else 1


def cmd_check_worst(args):
    worst, cases = _cases(not args.no_large)
    return 0 if print_table(worst.check(cases)) else 1


# -------------------------------------------------------------------- subset
SUBSET_MODELS = [("spots", "u16"), ("rings", "u32"), ("mx", "u16"), ("fibre", "u16"), ("flat", "u16"),
                 ("pilatus", "i32")]
CROP = 256


def _crop_window(idx, size=CROP, want=0.12):
    """The size x size window whose masked fraction is closest to `want`."""
    best = None
    for r in range(0, idx.shape[0] - size, size // 2):
        for c in range(0, idx.shape[1] - size, size // 2):
            f = float((idx[r:r + size, c:c + size] != 0).mean())
            sc = abs(f - want)
            if best is None or sc < best[0]:
                best = (sc, r, c)
    return best[1], best[2]


def cmd_subset(args):
    from . import ext, worst
    out = args.dir or os.path.join(K.WT, "tests", "data", "synth")
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(out):
        os.remove(os.path.join(out, f))
    man = {}
    for model, dt in SUBSET_MODELS:
        if model == "pilatus":
            params, arrays = twin.load_params("hplc", K.PARAMS_DIR)
            fr = twin.frames(params, arrays, 1, SEED, mask_kind="native")
        else:
            spec = dict(model=model, dtype=dt, det="4M", mask="native", scale=1.0, frames=1)
            fr, params = ext.frames_for(spec)
        if model == "flat":
            idx = np.zeros(fr.shape[1:], dtype=np.uint8)
        else:
            idx = twin.load_params(ext.MODEL_KEY.get(model, "hplc"), K.PARAMS_DIR)[1]["mask_idx"]
            if idx.shape != fr.shape[1:]:
                idx = twin.resize_labels(idx, fr.shape[1:])
        r, c = _crop_window(idx)
        crop = np.ascontiguousarray(fr[:, r:r + CROP, c:c + CROP])
        for enc in ("lz4_8k", "zstd2_256k"):
            name = "%s_%s.%s.h5" % (model, dt, enc.split("_")[0] if enc.startswith("lz4") else "zstd")
            encode.write_h5(crop, os.path.join(out, name), enc)
            man[name] = {"raw_sha256": hashlib.sha256(crop.tobytes()).hexdigest(), "dtype": str(crop.dtype),
                         "shape": list(crop.shape), "crop": [r, c], "bytes": int(crop.nbytes)}
    worst_ = worst.build_cases()
    for c in worst_:
        if c.zst is None or len(c.zst) > 450000:
            continue
        name = "worst_%s.zst" % c.name
        open(os.path.join(out, name), "wb").write(c.zst)
        raw = bytes(c.raw)
        man[name] = {"raw_sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                     "expected_reject": bool(c.meta.get("expected_reject"))}
    for c in worst_:                               # the small HDF5 shape cases
        if c.meta.get("prop") in ("tiny_frames", "many_frames"):
            for enc, arrays in c.h5[:1]:
                name = "worst_%s.%s.h5" % (c.name, enc.split("_")[0] if enc.startswith("lz4") else "zstd")
                encode.write_h5(arrays, os.path.join(out, name), enc)
                man[name] = {"datasets": {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
                                          for k, v in arrays.items()}, "bytes": sum(v.nbytes for v in arrays.values())}
    with open(os.path.join(out, "MANIFEST.json"), "w") as fh:
        json.dump(man, fh, indent=1, sort_keys=True)
        fh.write("\n")
    total = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out))
    print("subset: %d files, %.3f MB in %s" % (len(os.listdir(out)), total / 1e6, out))
    return 0 if total <= 5e6 else 1


# --------------------------------------------------------------- determinism
def _tree_sha(root):
    out = {}
    for dp, _, fns in os.walk(root):
        for fn in sorted(fns):
            p = os.path.join(dp, fn)
            out[os.path.relpath(p, root)] = sha256_file(p)
    return out


def cmd_determinism(args):
    import shutil
    import tempfile
    from . import ext
    td = tempfile.mkdtemp(prefix="sbz_det_", dir=os.environ.get("SLABITZ_TMP"))
    sums = []
    try:
        for run in ("a", "b"):
            root = os.path.join(td, run)
            for key in K.KEYS:
                write_twin(key, os.path.join(root, "twins", key), 2, SEED)
            for spec in ext.specs():
                if spec["det"] == "16M" or spec["dtype"] == "u64":
                    continue                         # large: the same code path at other sizes
                ext.write(dict(spec, frames=1), os.path.join(root, "ext"))
            worst, cases = _cases(with_large=False)
            worst.write(cases, os.path.join(root, "worst"))
            sums.append(_tree_sha(root))
    finally:
        shutil.rmtree(td, ignore_errors=True)
    a, b = sums
    bad = [k for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
    print("compared %d files (twins, ext, worst) generated twice with the same seed" % len(a))
    if bad:
        print("DIFFER:", *bad[:20], sep="\n  ")
        return 1
    print("all sha256 identical")
    return 0 if a else 1


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tools.synth")
    ap.add_argument("--out", help="output root (default $SLABITZ_SYNTH)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fit"); p.add_argument("keys", nargs="*"); p.set_defaults(fn=cmd_fit)
    p = sub.add_parser("generate"); p.add_argument("keys", nargs="*")
    p.add_argument("--frames", type=int, default=TWIN_FRAMES); p.add_argument("--seed", type=int, default=SEED)
    p.set_defaults(fn=cmd_generate)
    p = sub.add_parser("compare"); p.add_argument("--all", action="store_true"); p.add_argument("--key")
    p.add_argument("--enc", action="append"); p.add_argument("--quick", action="store_true")
    p.add_argument("--perturb", action="store_true", help="reviewer check: double the density, must FAIL")
    p.set_defaults(fn=cmd_compare)
    p = sub.add_parser("ext"); p.add_argument("--only"); p.set_defaults(fn=cmd_ext)
    p = sub.add_parser("worst"); p.add_argument("--no-large", action="store_true"); p.set_defaults(fn=cmd_worst)
    p = sub.add_parser("check-worst"); p.add_argument("--no-large", action="store_true"); p.set_defaults(fn=cmd_check_worst)
    p = sub.add_parser("subset"); p.add_argument("--dir"); p.set_defaults(fn=cmd_subset)
    p = sub.add_parser("determinism"); p.set_defaults(fn=cmd_determinism)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
