"""Detector geometries, gaps and masks.

Each detector is described by a pixel shape and a "gap mask": the columns and
rows that hold no sensitive pixels (module/chip gaps, beamstop shadow). Gap
pixels are written as the detector's masked value (0 for Eiger, 2**n-1 for the
Eiger masked-detector extreme, -1/-2 for Pilatus int32).

The synthetic generator needs the geometry so a twin has the same shape, the
same zeroed gap lines and the same masked fraction as its real file.  The
standard detector shapes (from data/real/INVENTORY.md):

    eiger4m   2162 x 2068  (ID11 Eiger 4M detectors used for silicon/ceo2/eiger4m_u32)
    eiger1_4m 2167 x 2070  (Eiger1 4M, kevlar)
    eiger2_16m 4362 x 4148 (DLS i04 Eiger2 16M, thaumatin)
    pilatus   1679 x 1475  (hplc, Pilatus 6M)

The exact gap lines are detector-specific and, in the real files, sometimes
depend on geometry/trim settings.  fit.py therefore *measures* the gap lines
(columns/rows that are always masked across the sampled frames) and stores them
in the parameter file; geometry only provides the shape and defaults.
"""

import numpy as np

# shape (rows, cols); module grid (rows x cols) of sensitive tiles.
DETECTORS = {
    # ID11 Eiger 2x5-module 4M detector (frames are 2162 x 2068)
    "eiger4m": {"shape": (2162, 2068), "modules": (2, 5)},
    "eiger1_4m": {"shape": (2167, 2070), "modules": (2, 5)},
    "eiger2_16m": {"shape": (4362, 4148), "modules": (2, 8)},
    "pilatus": {"shape": (1679, 1475), "modules": (2, 3)},
}

# Masked (gap) value written for each dtype/detector family.
#  "zero": gaps are 0 (Eiger default).
#  "max": gaps are 2**n-1 (the masked-detector extreme; a "max-1 mask").
#  "negative": Pilatus int32 gaps are -1; dead/bad pixels -2.
def gap_value(kind, dtype):
    dt = np.dtype(dtype)
    if kind == "zero":
        return 0
    if kind == "max":
        return int(dt.type(np.array([1], dtype=dt).max()))
    if kind == "negative":
        return -1
    raise ValueError("unknown gap kind %r" % (kind,))
