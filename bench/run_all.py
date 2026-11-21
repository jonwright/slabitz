"""Benchmark orchestrator: run ExecuteSequences (GPU, last stage only) vs nvCOMP and CPU libzstd.

Run:  .venv/bin/python bench/run_all.py           # full sweep
      .venv/bin/python bench/run_all.py text 65536 3   # one (dataset, chunk, level) config

For each configuration:
  * generate K distinct frames of ``chunk`` bytes, compress each with ``level`` into one zstd frame
  * ExecuteSequences: run the CPU reference chain to prebuild the kernel inputs, tile the K frames N
    times so the total output is >= 256 MiB, verify the whole batched output byte-for-byte against
    the original payloads, and time the kernel (only the launch is timed).
  * nvCOMP and CPU libzstd decode the SAME N-tiled frame copies (identical frame population).

The GPU ExecuteSequences is launched with one 32-thread workgroup per frame (grid.x = nFrames),
as in the DirectStorage design.

Every GPU launch is wrapped in a timeout by the caller environment; the orchestrator keeps the batch
small enough that a single config completes well within a minute.
"""
import os
import sys
import csv
import time
import json
import math

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "tests"))

from bench import gen, baselines, execute_bench

RESULTS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.csv")
MIN_BATCH_BYTES = 256 * 1024 * 1024

DATASETS = ["text", "detector", "detector_dense"]
CHUNKS = [65536, 262144, 1048576]
LEVELS = {65536: [1, 3, 9, 19], 262144: [1, 3, 9, 19], 1048576: [1, 3, 9]}
K = 16  # distinct frames prepared for ExecuteSequences (CPU reference path is slow)


def data_header():
    return ["dataset", "chunk_bytes", "level", "method", "n_frames", "uncompressed_bytes",
            "compressed_bytes", "ratio", "seconds_median", "gbps", "verified"]


def collect_system_info():
    """Return a dict of setup info (GPU, driver, nvCOMP version, CPU cores)."""
    import subprocess
    import zstandard as zstd
    info = {"python": sys.version.split()[0], "libzstd": zstd.__version__}
    try:
        import nvidia.nvcomp as nvcomp
        info["nvcomp_version"] = getattr(nvcomp, "__version__", "?")
    except Exception as e:
        info["nvcomp_version"] = "?" + str(e)
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
                              "--format=csv"], capture_output=True, text=True).stdout.strip()
        info["nvidia_smi"] = out.replace("\n", " | ")
    except Exception as e:
        info["nvidia_smi"] = "? " + str(e)
    info["os_cpu_count"] = os.cpu_count()
    try:
        info["nproc"] = subprocess.run(["nproc"], capture_output=True, text=True).stdout.strip()
    except Exception as e:
        info["nproc"] = "? " + str(e)
    import shutil
    info["nvcc"] = shutil.which("nvcc") or "not found (cannot rebuild kernels)"
    return info



def write_row(row, path=RESULTS_CSV):
    newfile = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if newfile:
            w.writerow(data_header())
        w.writerow(row)


def n_for_chunk(chunk, k=K, min_bytes=MIN_BATCH_BYTES):
    """Pick N so K*N*chunk >= min_bytes (tile the K distinct frames N times)."""
    per_tile = k * chunk
    return max(1, math.ceil(min_bytes / per_tile))


def prep_gpu_frames(dataset, chunk, level, k=K):
    """Compress K distinct frames and build the ExecuteSequences inputs for each."""
    payloads = gen.make_frames(dataset, k, chunk, seed=20261118)
    comps = gen.compress(payloads, level)
    frames = []
    for c, p in zip(comps, payloads):
        t0 = time.time()
        fr = execute_bench.prepare_frame(c, p)
        if fr["offset_mismatches"] not in (None, 0):
            print(f"    ! offset mismatches vs oracle: {fr['offset_mismatches']} "
                  f"({dataset}/{chunk}@{level})", flush=True)
        frames.append(fr)
        if chunk >= 1048576 and (len(frames)) % 4 == 0:
            print(f"    ... prepped {len(frames)}/{k} frames "
                  f"({time.time() - t0:.2f}s for last)", flush=True)
    return frames, comps


def run_config(dataset, chunk, level):
    print(f"=== {dataset}  chunk={chunk}  level={level} ===", flush=True)
    t0 = time.time()
    frames, comps = prep_gpu_frames(dataset, chunk, level)
    print(f"  prepped K={len(frames)} frames in {time.time() - t0:.1f}s; "
          f"compressed {sum(len(c) for c in comps)} B", flush=True)

    N = n_for_chunk(chunk)
    nFrames = K * N
    batch_comps = comps * N  # [K distinct] repeated N times, identical to the GPU tile
    comp_bytes = sum(len(c) for c in batch_comps)

    # ---- GPU ExecuteSequences (last stage only, single-workgroup serial) ----
    t0 = time.time()
    gpu = execute_bench.run_gpu_bench(frames, N, warmup=1, reps=2, verify=True)
    print(f"  GPU exec: verified={gpu['verified']} nFrames={gpu['nFrames']} "
          f"out={gpu['out_bytes']/2**20:.1f}MiB kernel {gpu['seconds_median']*1e3:.1f}ms "
          f"-> {gpu['gbps_total']:.3f} GB/s ({time.time()-t0:.1f}s)", flush=True)
    # persist the GPU row immediately so a later baseline failure does not lose it
    write_row([dataset, chunk, level, "zstd_gpu_exec_last_stage", gpu["nFrames"],
               gpu["out_bytes"], comp_bytes, gpu["out_bytes"] / comp_bytes,
               gpu["seconds_median"], gpu["gbps_total"], gpu["verified"]])

    # ---- baselines on the SAME tiled frame copies ----
    bas = baselines.run_baselines(batch_comps)

    rows = []
    rows.append([dataset, chunk, level, "nvcomp_batched", nFrames,
                 bas["uncompressed_bytes"], bas["compressed_bytes"],
                 bas["uncompressed_bytes"] / bas["compressed_bytes"],
                 bas["seconds_nvcomp"], bas["gbps_nvcomp"], bas["verified_nvcomp"]])
    rows.append([dataset, chunk, level, "cpu_zstd_mt", nFrames,
                 bas["uncompressed_bytes"], bas["compressed_bytes"],
                 bas["uncompressed_bytes"] / bas["compressed_bytes"],
                 bas["seconds_cpu_mt"], bas["gbps_cpu_mt"], True])
    rows.append([dataset, chunk, level, "cpu_zstd_st", nFrames,
                 bas["uncompressed_bytes"], bas["compressed_bytes"],
                 bas["uncompressed_bytes"] / bas["compressed_bytes"],
                 bas["seconds_cpu_st"], bas["gbps_cpu_st"], True])
    for r in rows:
        write_row(r)
    return rows


def main():
    import subprocess
    args = sys.argv[1:]
    # Single-config worker mode: one fresh process, GPU-exec (no cupy yet) then baselines (cupy),
    # then exit. Used both directly (dev) and as the child of the sweep below.
    if len(args) >= 3:
        run_config(args[0], int(args[1]), int(args[2]))
        return

    setup = collect_system_info()
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "setup.json"), "w") as f:
        json.dump(setup, f, indent=2)
    configs = []
    for ds in DATASETS:
        for ch in CHUNKS:
            for lv in LEVELS[ch]:
                configs.append((ds, ch, lv))

    # Each configuration runs in a fresh subprocess. The ExecuteSequences loader creates its OWN
    # CUDA driver context; once the CUDA runtime context (cupy/nvCOMP) exists in a process, creating
    # a further explicit driver context on this busy, shared GPU fails with out-of-memory. Isolating
    # per config (GPU-exec before any cupy import, then baselines, then process exit) avoids that.
    this = os.path.abspath(__file__)
    for (ds, ch, lv) in configs:
        print(f"### config {ds}/{ch}/{lv} (subprocess)...", flush=True)
        r = subprocess.run([sys.executable, this, str(ds), str(ch), str(lv)],
                           cwd=_REPO, capture_output=True, text=True)
        sys.stdout.write(r.stdout)
        sys.stderr.write(r.stderr)
        if r.returncode != 0:
            print(f"  subprocess {ds}/{ch}/{lv} exited {r.returncode}", flush=True)
            write_row([ds, ch, lv, "ERROR", "", "", "", "", "", "", "subprocess rc=%d" % r.returncode])
    return None


if __name__ == "__main__":
    main()
