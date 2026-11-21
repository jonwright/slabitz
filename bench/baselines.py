"""Baselines: nvCOMP batched zstd decode (GPU) and CPU libzstd multi-thread decode.

Both are measured on the SAME compressed frames used for the ExecuteSequences benchmark, so the
methods are comparable on identical input. Host<->device copies and compression are excluded from
all timings (the compressed frames are uploaded once; the timed part is purely the decompression).

nvCOMP
------
``nvidia.nvcomp`` with ``algorithm='Zstd'`` and ``bitstream_kind=RVB``... no: ``RAW`` (zstd whole
frames must be fed with ``BitstreamKind.RAW``). Batched decode via ``nvcomp.BatchArray`` built with
``nvcomp.as_arrays(contiguous_device_buffer, offsets, sizes)`` (a list of arrays is rejected), then
``codec.decode(batch, data_type='|u1')``. The frames' content size is written by ``zstandard`` so
nvCOMP knows each output size. The decode is async, so we time it on a CUDA stream plus a
``stream.synchronize()``.

CPU
---
``zstandard.ZstdDecompressor().multi_decompress_to_buffer(comps, threads=threads)`` over a list of
compressed frames, with ``threads=-1`` (all cores) and ``threads=1`` (single core).
"""
import os
import sys
import time

import numpy as np
import zstandard as zstd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)


def _cupy():
    """Lazily import cupy + nvCOMP. Importing cupy first initialises the CUDA runtime primary
    context; if the ExecuteSequences loader (which creates its OWN driver context) was already
    used, creating another explicit context on a busy/shared GPU can fail with out-of-memory.
    Importing here (after the loader context is created and destroyed) avoids the clash and keeps
    the loader's cuCtxCreate free to succeed."""
    import cupy as _cp
    import nvidia.nvcomp as _nv
    return _cp, _nv


def _expected(comps):
    d = zstd.ZstdDecompressor()
    return [d.decompress(c) for c in comps]


def probe_nvcomp(comps):
    """One batched nvCOMP decode used to (a) read back the uncompressed bytes and (b) verify the
    decode is byte-exact on CPU-computed ground truth. Returns (uncompressed_bytes, verified)."""
    cp, nvcomp = _cupy()
    codec = nvcomp.Codec(algorithm="Zstd", bitstream_kind=nvcomp.BitstreamKind.RAW)
    gpu, offs, sizes = _pack(comps)
    batch = nvcomp.as_arrays(gpu, offs, sizes)
    out = codec.decode(batch, data_type="|u1")
    cp.cuda.Stream.null.synchronize()
    views = [cp.asarray(out[i]).get().tobytes() for i in range(len(out))]
    uncompressed = sum(len(v) for v in views)
    verified = all(v == o for v, o in zip(views, _expected(comps)))
    return uncompressed, verified


def _pack(comps):
    cp, _nv = _cupy()
    buf = b"".join(comps)
    offs, sizes = [], []
    pos = 0
    for c in comps:
        offs.append(pos)
        sizes.append(len(c))
        pos += len(c)
    gpu = cp.asarray(np.frombuffer(buf, np.uint8))
    return gpu, offs, sizes


def nvcomp_decode_timed(comps, codec=None, gpu=None, offs=None, sizes=None, warmup=2, reps=5):
    """Time the batched nvCOMP decode (async decode + stream synchronize). Returns seconds_median."""
    cp, nvcomp = _cupy()
    if codec is None:
        codec = nvcomp.Codec(algorithm="Zstd", bitstream_kind=nvcomp.BitstreamKind.RAW)
    if gpu is None:
        gpu, offs, sizes = _pack(comps)
    batch = nvcomp.as_arrays(gpu, offs, sizes)
    for _ in range(warmup):
        s = cp.cuda.Stream()
        with s:
            codec.decode(batch, data_type="|u1")
        s.synchronize()
    ts = []
    for _ in range(reps):
        s = cp.cuda.Stream()
        t0 = time.perf_counter()
        with s:
            codec.decode(batch, data_type="|u1")
        s.synchronize()
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2]


def cpu_decode_timed(comps, threads, warmup=2, reps=5):
    """Time CPU libzstd multi_decompress_to_buffer over ``comps``. Returns seconds_median."""
    def one_run():
        t0 = time.perf_counter()
        zstd.ZstdDecompressor().multi_decompress_to_buffer(comps, threads=threads)
        return time.perf_counter() - t0
    for _ in range(warmup):
        one_run()
    ts = [one_run() for _ in range(reps)]
    return sorted(ts)[len(ts) // 2]


def run_baselines(comps, n_cpu=None):
    """Run both baselines on the same compressed frames; returns a dict of results."""
    import os as _os
    n_cpu = n_cpu or (os.cpu_count() if hasattr(os, "cpu_count") else 1)

    uncompressed, verified_nv = probe_nvcomp(comps)
    _cp, nvcomp = _cupy()
    codec = nvcomp.Codec(algorithm="Zstd", bitstream_kind=nvcomp.BitstreamKind.RAW)
    gpu, offs, sizes = _pack(comps)
    t_nv = nvcomp_decode_timed(comps, codec=codec, gpu=gpu, offs=offs, sizes=sizes)

    t_cpu_mt = cpu_decode_timed(comps, threads=-1)
    t_cpu_st = cpu_decode_timed(comps, threads=1)

    # The GPU is shared; release cupy's cached blocks so the next config's allocations don't
    # compound with another user's memory and cause an out-of-memory.
    try:
        cp, _nv = _cupy()
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass

    compressed = sum(len(c) for c in comps)
    return {
        "uncompressed_bytes": uncompressed,
        "compressed_bytes": compressed,
        "verified_nvcomp": verified_nv,
        "seconds_nvcomp": t_nv,
        "gbps_nvcomp": uncompressed / t_nv / 1e9,
        "seconds_cpu_mt": t_cpu_mt,
        "gbps_cpu_mt": uncompressed / t_cpu_mt / 1e9,
        "seconds_cpu_st": t_cpu_st,
        "gbps_cpu_st": uncompressed / t_cpu_st / 1e9,
        "n_cpu": n_cpu,
    }
