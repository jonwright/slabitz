# T04a: ExecuteSequences regression on stock zstd since b840caa

Status: review (no fix, accepted by the user; `docs/dev/reports/T04a.md`)
Executor: Claude (kernel, through the DS transform)
Depends on: T04 (the measurements and scripts)
Branch: `t04a-execseq-stock`

## Goal

b840caa made ExecuteSequences 8-19% slower on stock zstd frames with many
plane repeats (wall +5% to +7%), while the same commits made pac-zstd
ExecuteSequences 11-37% faster. Find the part of b840caa's ExecuteSequences
rewrite that costs the stock time. Fix it if that can be done without
losing the pac gains, otherwise record why not.

## Background (`docs/dev/reports/T04.md`, question 2)

- **Cells** (L40S, 16 frames, sync), ExecuteSequences us, 439e2f2 -> main:
  - zstd3/silicon_u16_maskmax-1: 1679.5 -> 1893.5 (+12.7%)
  - zstd3/thaumatin_u16_maskmax-1: 3192.5 -> 3683.5 (+15.4%)
  - twin/silicon_zstd2_256k.maskmax: 1652.5 -> 1847.5 (+11.8%)
  - worst/short_offsets_1_31_zst: 1206.5 -> 1304.0 (+8.1%)

  No other kernel moved. The pac counterweight is
  pac-zstd/silicon_u32_maskmax-1: 1320.0 (439e2f2) -> 854.5 (main).
- **Already excluded**, as variants of the current kernel (one edit each
  to `tools/transform_ds_for_slang.py`, in `MATCH_COPY_NEW`):
  - a: `false &&` in front of the aligned `uint32` copy condition;
  - b: `false &&` in front of the aligned `uint4` copy condition;
  - c: `isOff1 = false` (no offset-1 vector fill);
  - d: a + b + c, which recovers only 48 of the 206 us;
  - e: `[noinline]` on `zstdgpu_MatchCopy`, which is slower still.

  Registers (50 at 439e2f2 and main, no local memory) and code size (the
  non-inlined variant is slower) are also excluded.
- **What is left** in b840caa's rewrite:
  1. the two extra aliasing views `inoutUnCompressedFramesData32/16`,
     passed through every MatchCopy and FusedCopy call site (the
     `EXEC_VEC_PATCHES` list in the transform);
  2. the short-offset path's restructuring: `copyLen` computed before the
     branch chain, and offset 1 moved out of the swizzle loop.
- **Where the bytes go** (`bench/t04_seqstats.py --codec zstd3 --case
  silicon_u16_maskmax-1`, 4 frames): 16.3 MB of the 36 MB of match bytes
  are in 644 long off=P plane repeats (mean 25 kB), and 17.4 MB in 128,456
  matches on the byte loop.

## Read first

`AGENTS.md` (the rules for editing DS only through the transform);
`docs/dev/reports/T04.md` (method, question 2); `tools/transform_ds_for_slang.py`
(`MATCH_COPY_OLD`, `MATCH_COPY_NEW`, `EXEC_VEC_PATCHES`,
`patch_match_copy_swizzle`, `patch_execute_sequences_srt`); `git show
b840caa -- tools/transform_ds_for_slang.py`; `bench/t04_runs.sbatch` (mode
`bisect`, `BISECT_LIBS`).

## Settled design

- **Bisect first, one commit with only measurements.** Two new variants
  of the 439e2f2 kernel (made in `wt/t04-at-439e2f2`, or recreated with
  `dev-root new` at 439e2f2 and a `third_party/slang` link to
  `shared/slang`):
  - f: 439e2f2's MatchCopy with only the two views added (passed but not
    used);
  - g: 439e2f2's MatchCopy with only the short-offset restructuring.

  Time them with `BISECT_LIBS="439e2f2 main var-f var-g"
  sbatch bench/t04_runs.sbatch bisect 3`, the whole job on one node. Add
  `var-f` and `var-g` paths to the script's `lib()` if needed.
- **Then the fix**, in the transform only. Whatever the bisect points at,
  keep the pac paths (the `uint4` and `uint32` copies, the off=1 fills and
  the fused short fills). If the views are the cause, try one view with
  casts at the use sites, or the views only on the vector branches.
- **No fix is also a result.** If no change recovers the stock time
  without losing pac time, the report says which part costs it and the
  task closes with the regression documented in `AGENTS.md`. The T20
  rewrite of ExecuteSequences then starts from that knowledge.

## Files that may change

`tools/transform_ds_for_slang.py`, `src/sbz_kernels_gen.h` (regenerated),
`bench/t04_runs.sbatch` (variant paths), `bench/baselines/*.json` (if the
fix lands), `AGENTS.md` (the T04 Status row), `docs/dev/reports/T04a.md`.

## Acceptance

1. The report has the bisect table (439e2f2, main, f, g) for the four
   cells above plus pac-zstd/silicon_u32_maskmax-1, 3 rounds, with
   spreads, and names the part that costs the time.
2. With a fix: the gate `--compare` on the baseline's node shows
   ExecuteSequences on the four stock cells within 3% of 439e2f2, and no
   cell regresses beyond the gate thresholds; pac-zstd ExecuteSequences
   stays within 3% of main. The baseline is refreshed.
3. `sbz test`, `sbz test-kernels`, `run_pac_fixtures.py`,
   `run_features.py` and `run_roundtrip.py` pass. The report says whether
   `build/sm_35` and `build/kernels` need rebuilding (T05).

## Reviewer checks

- Rerun the bisect job for f and g, or at least main against the fix, on
  an idle node: the numbers must agree with the report within the
  reported spreads.
- Rerun the gate compare yourself.
