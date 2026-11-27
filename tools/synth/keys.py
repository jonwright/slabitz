"""The real data sets a twin is made for, and where they live.

Each key names one real file (read for ``fit`` and ``compare``), the dataset
inside it, the way its masked pixels are found, and the encodings the twin is
written in.  ``mask_values`` lists the values a pixel always holds in every
sampled frame when it is masked (gap, dead or bad pixel).  ``mask_file`` names
a second file of the same detector in which the mask is written as 2**n-1
(silicon and ceo2: their plain files write the gaps as 0, which cannot be told
from an empty pixel in a sparse file).
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))     # tools/synth
WT = os.path.dirname(os.path.dirname(HERE))           # the checkout
DEV_ROOT = os.environ.get("SLABITZ_DEV_ROOT") or os.path.dirname(os.path.dirname(WT))
EXAMPLES = os.environ.get("SLABITZ_EXAMPLES", os.path.join(DEV_ROOT, "data", "real"))
SYNTH_ROOT = os.environ.get("SLABITZ_SYNTH", os.path.join(DEV_ROOT, "data", "synth", "current"))
PAC_LIB = os.environ.get("PAC_LIB", os.path.join(DEV_ROOT, "ext", "pac-zstd", "build", "libpac.so"))
PARAMS_DIR = os.path.join(HERE, "params")

EIG = "entry_0000/ESRF-ID11/eiger/data"

# model: the family the key stands for in the extension set (see ext.py).
KEYS = {
    "silicon": dict(real="silicon_lz4.h5", ds=EIG, model="spots", mask_file="silicon_lz4mask.h5",
                    mask_values=[], enc=["lz4_8k", "zstd2_256k"]),
    "ceo2": dict(real="ceo2_lz4.h5", ds=EIG, model="rings", mask_values=[0],
                 mask_file="ceo2_lz4mask.h5", enc=["lz4_8k", "zstd2_256k"]),
    "thaumatin": dict(real="thaumatin_9_1_000001.h5", ds="data", model="mx",
                      mask_values=[65535], enc=["lz4_8k"]),
    "eiger4m_u32": dict(real="eiger4m_u32.h5", ds=EIG, model="scan", mask_values=[65535],
                        enc=["lz4_8k"]),
    "big_eiger": dict(real="big_eiger_0000.h5", ds=EIG, model="scan", mask_values=[65535],
                      enc=["lz4_8k"]),
    "kevlar": dict(real="kevlar.h5", ds="entry/data/data", model="fibre", mask_values=[65535],
                   enc=["lz4_8k"]),
    "hplc": dict(real="hplc.h5", ds="entry_0000/instrument/Pilatus/data", model="pilatus",
                 mask_values=[-1, 0], enc=["lz4_8k"]),
    "al2o3": dict(real="Calibration_Al2O3.h5", ds="data", model="calibrant", mask_values=[0],
                  enc=["lz4_8k"]),
}


def real_path(key, name=None):
    return os.path.join(EXAMPLES, name or KEYS[key]["real"])
