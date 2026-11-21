"""Benchmark harness for slabitz ExecuteSequences vs nvCOMP and CPU libzstd.

Submodules:
  gen         dataset generation, chunking into frames, compression, caching
  execute_bench  GPU ExecuteSequences prep+tile+verify+timing (last stage only)
  baselines   nvCOMP batched decode and CPU libzstd decode timing
"""
