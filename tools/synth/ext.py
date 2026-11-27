"""The extension set: the twins' patterns in the encodings and sizes the real
files do not cover.

Every entry is a function of (model, dtype, detector size, mask kind, lam
scale, seed): the fitted pattern of one twin key, resampled to another detector
size, written as another element type, with the masked pixels as 0 or 2^n-1, at
another brightness, in another block size.  Models:

    spots   silicon (sparse Bragg spots)       rings   ceo2 (dense powder rings)
    mx      thaumatin (halo and spots)         fibre   kevlar (fibre arcs)
    flat    constant Poisson mean 70, no mask
"""

import json
import os

import numpy as np

from . import encode, keys as K, twin

MODEL_KEY = {"spots": "silicon", "rings": "ceo2", "mx": "thaumatin", "fibre": "kevlar"}
DT = {"u8": "u1", "u16": "<u2", "u32": "<u4", "u64": "<u8", "i32": "<i4"}
SHAPES = {"1M": (1024, 1024), "4M": (2162, 2068), "16M": (4362, 4148)}
BASE = ["lz4_8k", "zstd2_256k"]
BLOCKS = ["lz4_8k", "lz4_56k", "zstd_248k", "zstd2_256k"]
SEED = 5150


def specs():
    out = []

    def add(model, dt, det="4M", mask="native", scale=1.0, encs=BASE, frames=2, tag=""):
        name = "%s_%s_%s%s%s" % (model, dt, det, "" if mask == "native" else "_mask" + mask,
                                 "_x%g" % scale if scale != 1.0 else "") + tag
        out.append(dict(name=name, model=model, dtype=dt, det=det, mask=mask, scale=scale, encs=encs, frames=frames))

    for model in ("spots", "rings", "mx", "fibre", "flat"):
        for dt in ("u8", "u16", "u32", "u64"):
            sc = 0.25 if (dt == "u8" and model in ("rings", "flat")) else 1.0
            add(model, dt, scale=sc)
    for model, dt in (("spots", "u16"), ("rings", "u32"), ("mx", "u16"), ("fibre", "u16")):
        add(model, dt, mask="zero", encs=["lz4_8k"])
        add(model, dt, mask="max", encs=["lz4_8k"])
    add("spots", "u16", encs=BLOCKS, tag="_blocks")
    add("rings", "u32", encs=BLOCKS, tag="_blocks")
    for det in ("1M", "16M"):
        add("spots", "u16", det=det)
        add("rings", "u32", det=det)
    add("rings", "u32", scale=1000.0)                       # high dynamic range
    add("rings", "i32", mask="zero", tag="_signed")          # signed element, no negative values
    return out


def frames_for(spec, seed=SEED):
    if spec["model"] == "flat":
        params, arrays = twin.flat_params(shape=SHAPES[spec["det"]])
    else:
        params, arrays = twin.load_params(MODEL_KEY[spec["model"]], K.PARAMS_DIR)
    shape = SHAPES[spec["det"]] if spec["model"] != "flat" else None
    fr = twin.frames(params, arrays, spec["frames"], seed, dtype=DT[spec["dtype"]], shape=shape,
                     mask_kind=spec["mask"], lam_scale=spec["scale"])
    return fr, params


def write(spec, outdir, seed=SEED):
    from .cli import sha256_file
    fr, params = frames_for(spec, seed)
    d = os.path.join(outdir, spec["name"])
    os.makedirs(d, exist_ok=True)
    files = {}
    p = os.path.join(d, "raw.npy")
    np.save(p, fr)
    files["raw.npy"] = sha256_file(p)
    for enc in spec["encs"]:
        p = os.path.join(d, enc + ".h5")
        encode.write_h5(fr, p, enc)
        files[enc + ".h5"] = sha256_file(p)
    man = dict(spec, seed=seed, generator_version=twin.VERSION, numpy=np.__version__, files=files)
    with open(os.path.join(d, "manifest.json"), "w") as fh:
        json.dump(man, fh, indent=1, sort_keys=True)
        fh.write("\n")
    return files
