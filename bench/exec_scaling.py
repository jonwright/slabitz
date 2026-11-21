"""ExecuteSequences (one workgroup per frame) throughput vs batch size, byte-exact verified.

Run: .venv/bin/python bench/exec_scaling.py   (text, 64 KiB chunks, level 3)"""
import sys; sys.path.insert(0,"."); sys.path.insert(0,"tests")
from bench import run_all, execute_bench
frames, _ = run_all.prep_gpu_frames("text", 65536, 3)
for N in (64, 256, 1024):
    r = execute_bench.run_gpu_bench(frames, N, warmup=2, reps=5, verify=True)
    print("frames=%5d verified=%s %.1fms %.1f GB/s" % (r["nFrames"], r["verified"], r["seconds_median"]*1e3, r["out_bytes"]/r["seconds_median"]/1e9), flush=True)
