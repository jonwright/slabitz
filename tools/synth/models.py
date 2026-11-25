"""Generator models for detector frames.

A "model" maps a parameter dict plus an RNG to a frame of expected counts
(a float array the same shape as the detector).  The frame pipeline in
``models.sample`` then (1) adds the detector gaps/mask, (2) samples counts
(Poisson, or discrete counts for the sparse case), (3) clips to the dtype.

Each model returns only the *intensity* image (per-pixel expected count); the
Poisson draw is common and lives in ``sample`` so every model gets identical
noise handling.  Models are:

    spots    2D Gaussians on a smooth background        (silicon: Bragg spots)
    rings    powder diffraction rings + smooth azimuth  (ceo2, al2o3)
    mx       diffuse halo + sharp spots                 (thaumatin, eiger scans)
    fibre    fibre arcs over a smooth background        (kevlar)
    flat     constant background (optional azimuthal texture) (plain Poisson)

All models are pure numpy and deterministic for a given (params, seed).
"""

import numpy as np

_BIG = 1 << 60


def _smooth_field(shape, coef):
    """A smooth, slowly varying positive field built from low-frequency
    sinusoids; used as an intensity multiplier so the mean varies smoothly
    over the detector.  Returns a float array on the detector grid."""
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float64)
    fx = (xx - shape[1] / 2.0) / max(shape[1], 1)
    fy = (yy - shape[0] / 2.0) / max(shape[0], 1)
    ph = np.arctan2(fy, fx)
    r = np.sqrt(fx * fx + fy * fy)
    w = (
        1.0
        + coef[0] * np.sin(r * 4.5 * np.pi + 0.7)
        + coef[1] * np.sin(ph * 2.0 + 1.9)
        + coef[2] * np.cos(r * 7.0 * np.pi + 2.5)
    )
    return np.clip(w, 0.05, None)



def _field_coef(p, default=0.1):
    """Return a length-3 list of smooth-field coefficients from model params."""
    v = p.get("field", default)
    if isinstance(v, (list, tuple)):
        return [float(x) for x in v][:3]
    return [float(v)] * 3

def _poisson(rng, lam):
    """Poisson sample of a float intensity array, as an integer array."""
    return rng.poisson(lam).astype(np.float64)


def _pareto(rng, n, xmin, alpha):
    """Pareto power-law tail: p(x) ~ x^{-(alpha+1)} for x >= xmin."""
    u = rng.random(n)
    return xmin * np.power(1.0 - u, -1.0 / alpha)


def spots(shape, p, rng):
    """Sparse Bragg spots on a smooth, small background.

    p keys: mean (background mean), field (smooth multiplier amplitude),
    n_spots, spot_sigma (mean blob radius), alpha (intensity power-law),
    max_intensity, relative spot intensity scale.
    """
    mean = float(p.get("mean", 0.004))
    field = np.array(p.get("field", [0.05, 0.05, 0.05]))
    back = mean * _smooth_field(shape, field)
    img = back.copy()
    n = int(p.get("n_spots", 2200))
    sigma_mean = float(p.get("spot_sigma", 4.0))
    alpha = float(p.get("alpha", 2.2))
    scale = float(p.get("intensity", 9.0))
    if n:
        ys = rng.integers(0, shape[0], n).astype(int)
        xs = rng.integers(0, shape[1], n).astype(int)
        sig = sigma_mean * (1.0 + rng.normal(0, 0.4, n))
        sig = np.clip(sig, 1.0, 40.0)
        amp = _pareto(rng, n, 1.0, alpha) * scale
        amp = np.clip(amp, 0, float(p.get("max_intensity", 30.0)))
        # accumulate the Gaussians on the image grid by scattering into a hot buffer
        delta = np.zeros(shape)
        for j in range(n):
            if amp[j] < 0.05:
                continue
            s = max(1.0, float(sig[j]))
            r = max(1.0, int(np.ceil(3.0 * s)))
            y0, y1 = max(0, int(ys[j]) - r), min(shape[0], int(ys[j]) + r + 1)
            x0, x1 = max(0, int(xs[j]) - r), min(shape[1], int(xs[j]) + r + 1)
            if y1 <= y0 or x1 <= x0:
                continue
            yb = np.arange(y0, y1) - int(ys[j])
            xb = np.arange(x0, x1) - int(xs[j])
            g = np.exp(-0.5 * (yb[:, None] ** 2 + xb[None, :] ** 2) / (s * s))
            delta[y0:y1, x0:x1] += float(amp[j]) * g
        img = img + delta
    return img


def rings(shape, p, rng):
    """Powder diffraction rings: a radial profile with sharp peaks plus a
    smooth azimuthal texture and a flat background.

    p keys: mid (background mean), n_rings, radii (absolute, in pixels),
    widths, heights, az (azimuthal modulation amplitude), center offset.
    """
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float64)
    cx = shape[1] / 2.0 + float(p.get("cx", 0.0))
    cy = shape[0] / 2.0 + float(p.get("cy", 0.0))
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    ph = np.arctan2(yy - cy, xx - cx)
    img = np.full(shape, float(p.get("mid", 2.0)))
    img = img * _smooth_field(shape, _field_coef(p))
    radii = p.get("radii", [])
    widths = p.get("widths", [])
    heights = p.get("heights", [])
    az = float(p.get("az", 0.0))
    for k, rad in enumerate(radii):
        w = widths[k] if k < len(widths) else 20.0
        h = heights[k] if k < len(heights) else 200.0
        profile = h * np.exp(-0.5 * ((r - rad) / w) ** 2)
        if az:
            profile = profile * (1.0 + az * np.cos(2.0 * ph + 0.6) * np.cos(3.0 * ph + 1.1))
        img = img + profile
    return img


def mx(shape, p, rng):
    """Macromolecular crystallography: a broad diffuse halo, a low smooth
    background, and a modest number of sharp spots.  p keys: mid, halo_peak,
    halo_radius, halo_width, n_spots, spot_sigma, intensity/min/max."""
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float64)
    cx = shape[1] / 2.0 + float(p.get("cx", 0.0))
    cy = shape[0] / 2.0 + float(p.get("cy", 0.0))
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    img = np.full(shape, float(p.get("mid", 1.0)))
    img = img + float(p.get("halo_peak", 0.0)) * np.exp(
        -0.5 * ((r - float(p.get("halo_radius", 900.0))) / float(p.get("halo_width", 200.0))) ** 2
    )
    img = img * _smooth_field(shape, _field_coef(p))
    n = int(p.get("n_spots", 4000))
    sigma_mean = float(p.get("spot_sigma", 2.5))
    alpha = float(p.get("alpha", 2.0))
    scale = float(p.get("intensity", 300.0))
    if n:
        ys = rng.integers(0, shape[0], n).astype(int)
        xs = rng.integers(0, shape[1], n).astype(int)
        sig = np.clip(sigma_mean * (1.0 + rng.normal(0, 0.4, n)), 1.0, 30.0)
        amp = _pareto(rng, n, 1.0, alpha) * scale
        amp = np.clip(amp, 0, float(p.get("max_intensity", 40000.0)))
        delta = np.zeros(shape)
        for j in range(n):
            if amp[j] < 0.1:
                continue
            s = max(1.0, float(sig[j]))
            r_ = max(1.0, int(np.ceil(2.0 * s)))
            y0, y1 = max(0, int(ys[j]) - r_), min(shape[0], int(ys[j]) + r_ + 1)
            x0, x1 = max(0, int(xs[j]) - r_), min(shape[1], int(xs[j]) + r_ + 1)
            if y1 <= y0 or x1 <= x0:
                continue
            yb = np.arange(y0, y1) - int(ys[j])
            xb = np.arange(x0, x1) - int(xs[j])
            g = np.exp(-0.5 * (yb[:, None] ** 2 + xb[None, :] ** 2) / (s * s))
            delta[y0:y1, x0:x1] += float(amp[j]) * g
        img = img + delta
    return img


def fibre(shape, p, rng):
    """Fibre diffraction: concentric arcs strong along a principal axis.
    p keys: mid, arc_intensity, arc_radius, arc_width, arcs (number of
    period arcs), n_spots, spot_sigma."""
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float64)
    cx = shape[1] / 2.0 + float(p.get("cx", 0.0))
    cy = shape[0] / 2.0 + float(p.get("cy", 0.0))
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    ph = np.arctan2(yy - cy, xx - cx)
    img = np.full(shape, float(p.get("mid", 5.0)))
    # Sharpen the ring in the horizontal (phi ~ 0 or pi) direction -> fibre arcs
    anisotropy = np.exp(-(np.sin(ph) ** 2) / (2.0 * float(p.get("tx", 0.2)) ** 2))
    n_arcs = int(p.get("arcs", 3))
    base = float(p.get("arc_radius", 700.0))
    step = float(p.get("arc_step", 250.0))
    w = float(p.get("arc_width", 60.0))
    peak = float(p.get("arc_intensity", 900.0))
    for k in range(n_arcs):
        rad = base + k * step
        img = img + peak * np.exp(-0.5 * ((r - rad) / w) ** 2) * anisotropy
    img = img + _smooth_field(shape, _field_coef(p, default=0.15)) * float(
        p.get("mid", 5.0)
    )
    n = int(p.get("n_spots", 2000))
    if n:
        ys = rng.integers(0, shape[0], n).astype(int)
        xs = rng.integers(0, shape[1], n).astype(int)
        sig = np.clip(float(p.get("spot_sigma", 2.0)) * (1.0 + rng.normal(0, 0.4, n)), 1.0, 25.0)
        amp = _pareto(rng, n, 1.0, float(p.get("alpha", 2.0))) * float(p.get("intensity", 500.0))
        amp = np.clip(amp, 0, float(p.get("max_intensity", 40000.0)))
        delta = np.zeros(shape)
        for j in range(n):
            if amp[j] < 0.1:
                continue
            s = max(1.0, float(sig[j]))
            r_ = max(1.0, int(np.ceil(2.0 * s)))
            y0, y1 = max(0, int(ys[j]) - r_), min(shape[0], int(ys[j]) + r_ + 1)
            x0, x1 = max(0, int(xs[j]) - r_), min(shape[1], int(xs[j]) + r_ + 1)
            if y1 <= y0 or x1 <= x0:
                continue
            yb = np.arange(y0, y1) - int(ys[j])
            xb = np.arange(x0, x1) - int(xs[j])
            g = np.exp(-0.5 * (yb[:, None] ** 2 + xb[None, :] ** 2) / (s * s))
            delta[y0:y1, x0:x1] += float(amp[j]) * g
        img = img + delta
    return img


def flat(shape, p, rng):
    """Flat background with optional smooth azimuthal texture (plain Poisson)."""
    img = np.full(shape, float(p.get("mid", 70.0)))
    img = img * _smooth_field(shape, _field_coef(p))
    return img


MODELS = {"spots": spots, "rings": rings, "mx": mx, "fibre": fibre, "flat": flat}


def sample(model, params, shape, dtype, rng, gap_kind="zero", gap_cols=(), gap_rows=(),
           mask_value=None, neg_mask=0):
    """Generate one frame.

    Returns the uint/int array.  The intensity image is computed by `model`,
    then counts are drawn as Poisson and clipped to dtype, then the gap lines
    are overwritten with the detector's masked value.

    `gap_cols`/`gap_rows` are lists of pixel line indices to mask.  With
    `neg_mask`, a number of *fraction* of pixels are set to -1 / -2 (Pilatus).
    """
    intensity = MODELS[model](shape, params, rng)
    counts = _poisson(rng, np.clip(intensity, 0, None))
    dt = np.dtype(dtype)
    info = np.iinfo(dt)
    counts = np.clip(counts, info.min, info.max)
    frame = counts.astype(dt)

    # gaps / masked lines -> the masked (gap) value
    if gap_cols or gap_rows:
        if mask_value is None:
            mask_value = gap_value(gap_kind, dtype)
        if gap_rows:
            frame[list(gap_rows), :] = mask_value
        if gap_cols:
            frame[:, list(gap_cols)] = mask_value

    # Pilatus-style negative masked/bad pixels
    if neg_mask:
        n = int(shape[0] * shape[1] * neg_mask)
        ys = rng.integers(0, shape[0], n).astype(int)
        xs = rng.integers(0, shape[1], n).astype(int)
        vals = np.where(rng.random(n) < 0.5, -1, -2)
        frame[ys, xs] = vals.astype(dt)
    return frame
