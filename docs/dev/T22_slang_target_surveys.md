# T22: Slang target surveys (CPU, Metal, SPIR-V), compile only

Status: todo
Executor: worker (DeepSeek), reviewed by Claude
Depends on: - (rerun after T17-T19 add kernels)
Branch: `t22-slang-surveys`

## Goal

Know which kernels would port to the other Slang targets, without any
runtime work: compile every entry for Slang's CPU target, Metal and SPIR-V,
and record exactly what fails.

## Background

- **The last survey** was run at 21 kernels with slangc 2026.22:
  - SPIR-V and HLSL: 19/19.
  - Metal: 13/19.
    - `WavePrefixCountBits` has no Metal lowering (PrefixSum,
      ComputePrefixSum, ParseCompressedBlocks, InitFseTable).
    - Metal lacks `GetDimensions` (both unshuffle kernels; pass byte counts
      as scalars instead).
  - The two LZ4 kernels build for SPIR-V and Metal.
  - `ExecuteSequences` and the literals kernel were re-checked after
    tuning.
- **Not yet surveyed:** `BshufWalkChunks`, which was added later.
- **Slang's CPU target** is experimental (C++ kernel and host, exe or lib)
  and compute-only. It has never been tried.

## Settled design

- **New `tools/survey_targets.py`.** It compiles every entry under
  `kernels/*.slang` with the pinned `third_party/slang/bin/slangc` for
  `-target spirv`, `-target metal`, `-target hlsl` and `-target cpp` (or
  `-target shader-sharedlib`, whichever the pinned slangc supports for
  compute). It records per entry and target: ok, or the first error line
  with its code. It writes `docs/dev/reports/T22_targets.md` as a table.
- **CPU target summary.** For the CPU target, list which wave, groupshared
  and barrier intrinsics fail. End with a recommendation (worth a CPU
  backend or not) and its reasons. Do not decide; the user decides.
- **Rules.** No kernel source changes in this task.

## Files that may change

`tools/survey_targets.py`, `docs/dev/reports/T22.md`,
`docs/dev/reports/T22_targets.md`, `AGENTS.md` (the "Other Slang targets"
Status row).

## Acceptance

1. `python3 tools/survey_targets.py` exits 0 and writes the table, with
   every entry from `src/sbz_kernels_gen.h` (22 at the time of writing,
   more after T17-T19) x 4 targets.
2. Spot check: the counts for SPIR-V and Metal agree with the previous
   survey for the 21 kernels that existed then, or every difference is
   explained.
3. The `AGENTS.md` Status row is updated with the date and counts.

## Reviewer checks

Run `slangc` by hand for one failing Metal entry and one CPU entry, and
confirm that the recorded error matches.
