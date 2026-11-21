"""nvCOMP batched zstd decode vs batch size: host enqueue, wall time, and legacy-stream CUDA events.

Run: .venv/bin/python bench/nvcomp_scaling.py
Outputs are preallocated (out=), so per-call output allocation is excluded."""
import sys, time; sys.path.insert(0,".")
from bench import gen, baselines
import cupy as cp
cpm, nv = baselines._cupy()
frames = gen.make_frames("text", 16, 65536, 1)
base = gen.compress(frames, 3)
codec = nv.Codec(algorithm="Zstd", bitstream_kind=nv.BitstreamKind.RAW)
null = cp.cuda.Stream.null
for mult in (64, 256, 1024):
    comps = base * mult
    gpu, offs, sizes = baselines._pack(comps)
    batch = nv.as_arrays(gpu, offs, sizes)
    out = codec.decode(batch, data_type="|u1"); cp.cuda.Device().synchronize()
    tot = 65536 * len(comps)
    res = []
    for _ in range(7):
        cp.cuda.Device().synchronize()
        e0, e1 = cp.cuda.Event(), cp.cuda.Event()
        t0 = time.perf_counter(); e0.record(null)
        codec.decode(batch, data_type="|u1", out=out)
        t1 = time.perf_counter(); e1.record(null); e1.synchronize(); t2 = time.perf_counter()
        res.append((t1 - t0, t2 - t0, cp.cuda.get_elapsed_time(e0, e1) / 1e3))
    res.sort(key=lambda r: r[1]); enq, wall, ev = res[3]
    print("frames=%5d  enqueue=%.1fms  wall=%.1fms (%.1f GB/s)  null-stream-events=%.1fms (%.1f GB/s)"
          % (len(comps), enq*1e3, wall*1e3, tot/wall/1e9, ev*1e3, tot/ev/1e9))
