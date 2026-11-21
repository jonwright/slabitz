"""Dataset generation for the slabitz benchmark.

Each configuration is (dataset, chunk_bytes, level). A "frame" is one independent zstd frame
whose decompressed payload is exactly `chunk_bytes` (one contiguous chunk). We generate
deterministically, cache nothing here (cache is handled by the callers in ``bench/cache``).

Datasets
--------
text            word-soup text (like tests/run_exec_multiblock.py). High compression ratio.
detector        synthetic uint16 X-ray detector stack with Poisson(mean=0.1) noise -> sparse.
detector_dense  same but Poisson(mean=5) -> denser, lower compression ratio.

Generation contract
-------------------
``generate_blob(dataset, n_bytes, seed)`` returns exactly ``n_bytes`` of bytes, deterministic
for a given (dataset, n_bytes, seed). The caller slices this blob into ``n_bytes/chunk_bytes``
frames; because a fixed seed yields one fixed blob, the frames are distinct and reproducible.
"""
import numpy as np
import zstandard as zstd

WORDS = "the quick brown fox jumps over lazy dog run fast stop slow near far".split()


def _text_blob(n_bytes, seed):
    import random
    rng = random.Random(seed)
    parts = []
    size = 0
    while size < n_bytes:
        w = rng.choice(WORDS)
        s = (b" " if parts else b"") + w.encode()
        parts.append(s)
        size += len(s)
    return b"".join(parts)[:n_bytes]


def _detector_blob(n_bytes, seed, mean):
    n = (n_bytes + 1) // 2
    rng = np.random.default_rng(seed)
    vals = rng.poisson(mean, size=n).astype(np.uint16)
    return np.ascontiguousarray(vals).tobytes()[:n_bytes]


def generate_blob(dataset, n_bytes, seed):
    """Return exactly ``n_bytes`` of deterministic bytes for ``dataset``."""
    if dataset == "text":
        return _text_blob(n_bytes, seed)
    if dataset == "detector":
        return _detector_blob(n_bytes, seed, 0.1)
    if dataset == "detector_dense":
        return _detector_blob(n_bytes, seed, 5.0)
    raise ValueError("unknown dataset %r" % (dataset,))


def split_frames(blob, chunk_bytes):
    """Split ``blob`` into contiguous ``chunk_bytes`` frames (the last is truncated)."""
    return [blob[i:i + chunk_bytes] for i in range(0, len(blob), chunk_bytes)]


def make_frames(dataset, n_frames, chunk_bytes, seed):
    """Return a list of ``n_frames`` distinct payloads (one frame per chunk)."""
    blob = generate_blob(dataset, n_frames * chunk_bytes, seed)
    return split_frames(blob, chunk_bytes)


def compress(payloads, level):
    """Compress each payload into one independent zstd frame (content size written)."""
    c = zstd.ZstdCompressor(level=level)
    return [c.compress(p) for p in payloads]
