# T20: ExecuteSequences parallelism

Status: todo
Executor: Claude
Depends on: T03 (gate), T04 (the answers to the open performance questions)
Branch: `t20-exec-parallel`

## Goal

ExecuteSequences is the largest device cost of the zstd decode, and it runs
one 32-thread workgroup per frame. Give it more parallelism, so that a
batch of frames uses the whole GPU, with output unchanged.

## Background (measured 2026-11-20, L40S, 16 frames/call, device us/frame)

- **silicon zstd:**
  - ExecuteSequences 87
  - DecompressSequencesMultiStream 18
  - unshuffle 34
  - about 13 small kernels, about 15 (synced)
- **CeO2 zstd:**
  - upload 193 (PCIe-bound, 21 GB/s)
  - ExecuteSequences 112
  - literals 71
  - sequences 45
  - unshuffle 63
- **One silicon frame per call:** 1.55 ms wall, of which ExecuteSequences
  is 1.0 ms. With overlapped uploads (P4), zstd is device-bound on both
  files. ExecuteSequences parallelism is the main lever (former AGENTS.md
  P6).
- **Why one workgroup per frame.** A frame's blocks are serial: a match
  can reference bytes written by earlier blocks of the same frame, within
  the window. Within a block, sequences are serial in the same way. DS
  parallelises inside a sequence (lanes copy bytes) and across frames.
- **Fusion check.** Fusing the unshuffle into ExecuteSequences fails the
  fusion rule: one workgroup per frame, 256 kB blocks, and it would run on
  16 SMs instead of the whole GPU. Revisit only after this task.
- **pac frames.** For pac frames (one frame per 248 kB bitshuffle block),
  most bytes come from `off=1` fills and `off=P` copies (P is at least
  256 bytes), so a block's dependencies reach back at most one plane.

## Candidate approaches (measure first, then choose; record the choice)

1. **Several workgroups per frame, by block, with dependency tracking.**
   Each compressed block gets its own workgroup. The block's destination
   offset is already known (ComputeDestBlockOffsets). A block that only
   reads within itself, plus bytes already final, can start at once.
   - Gather per-block "minimum source offset" statistics in the sequence
     stage: the minimum of (dest - offset) over its sequences.
   - Run blocks in waves, or with a device-side flag per block (spin-wait
     on predecessors). Spin-waits need a forward-progress guarantee
     across workgroups that pre-Volta cards and Vulkan do not give, so
     prefer waves: one launch per dependency level.
2. **Wider workgroups:** 64 or 128 threads per frame, with sequence-level
   parallelism (the prefix sums of literal and match lengths are known).
   Literal copies of different sequences can run in parallel. Match copies
   must respect dependencies; resolve the chains with a "source is final"
   test per sequence.
3. **Split by sequence ranges inside a block**, using the sequence
   prefix sums, with the same dependency test as 2.

The first commit is a measurement-only report in
`docs/dev/reports/T20.md`. Using the T06/T07 corpora, the pac fixtures and
the gate files, it gives:
- the distribution of dependency distance per block and per sequence;
- what fraction of blocks and sequences could run independently under
  each approach.

Choose from that, and record the choice in the report.

## Files that may change

`tools/transform_ds_for_slang.py`, `kernels/ExecuteSequences.slang`,
`kernels/lib/*`, a possible stats kernel or debug entry, `src/sbz_chain.c`
(launch shape), `src/sbz_kernels_gen.h`, `bench/*`, `bench/baselines/*.json`,
`docs/dev/reports/T20.md`.

## Acceptance

1. A measurement report exists (the first commit) with the dependency
   statistics, and the chosen approach is justified by them.
2. `bench/gate.py --compare` on the L40S shows ExecuteSequences device
   time per frame at least 1.5x faster on silicon zstd (stock and pac) and
   on CeO2 zstd, at 16 frames per call. No other cell regresses beyond the
   gate thresholds. The baseline is refreshed.
3. **Byte-exact everywhere:**
   - `sbz test`, `sbz test-kernels`, `run_pac_fixtures.py`,
     `run_features.py` and `run_compat.py` (subset and `--full`);
   - `run_roundtrip.py --long` once;
   - `run_gpu_corrupt.py` with no new hangs.
4. **Old cards.** The T05 suite passes on the TITAN Black and the 1050 Ti
   with rebuilt `build/kernels`, or the report states which were not
   available. No spin-wait across workgroups unless this passes on
   pre-Volta.
5. The kernels still compile for SPIR-V and Metal, or the report lists
   what newly fails.

## Reviewer checks

- **Dependency removal.** If waves are used, launch all blocks in one wave
  (temporary): the tests must fail on data with cross-block matches, to
  prove the corpus contains them. If they pass, the corpus is missing a
  case: add one.
- Rerun the gate compare yourself on an idle GPU.
