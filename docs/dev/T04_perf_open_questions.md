# T04: Explain the open performance questions

Status: todo
Executor: Claude
Depends on: T03
Branch: `t04-perf-questions`

## Goal

Answer, with measurements from the T03 gate, the performance questions the
recent rapid fixes left open. Fix only what the measurements justify. The
answers matter more than any speed-up.

## Questions

1. **silicon_u32_maskmax-1 ExecuteSequences, +30%.** With pac-zstd 0.1
   frames (no sequence merging) ExecuteSequences took 1776 us/call; with
   pac 0.2.0 frames after 692e3f9 it takes 2321 us/call. Both are GTX 1050
   Ti, 4 frames (`docs/archive/plan_pac_zstd.md:115-131`). Wall time
   improved (1.24 -> 1.21 ms/fr) because pac 0.2.0 frames are smaller and
   there are fewer sequences. Possible causes, none verified:
   - (a) the `offs`-sized chunk loop costs more per byte than one long
     aligned copy;
   - (b) merged matches change the work balance between lanes;
   - (c) a measurement artefact (one run, default reps).

   Measure both frame versions with the current kernel and with b840caa's
   kernel. That is 2 x 2 runs, on the L40S and, if available, the
   1050 Ti.
2. **Small stock regressions in b840caa / a0338bf**, 2-4% on some stock
   cells:
   - silicon_u16_mask0 0.82 -> 0.85 ms/fr
   - silicon_u32_mask0 1.36 -> 1.40 ms/fr
   - ceo2_u32_maskmax-1 4.51 -> 4.54 ms/fr

   Real or noise? Compare gate cells at 439e2f2 (the root commit of the
   current history, before the pac-zstd tuning commits b840caa and
   a0338bf), at b840caa, at a0338bf, and at `main`.
3. **CeO2 per frame at 4 vs 16 frames per call.** `AGENTS.md` still asks
   for a re-check: the earlier claim came from a harness that joined the
   chunks inside the timed loop. Measure with the input built once, at 1, 4,
   16 and 64 frames per call, sync and async.
4. **Fused vs unfused LZ4, and `SLABITZ_LZ4_FUSED=0`, across the gate
   matrix.** This confirms that the fused path is never slower, as was
   claimed for silicon and CeO2 only.

## Method

- **Building old kernels.** For the older commits, make a scratch clone
  (`dev-root new t04-at-<commit> <commit>`, a worktree under
  `$SLABITZ_DEV_ROOT/wt/`) and build its
  kernels there. Each kernel set carries its own `sbz_kernels.id`, so the
  library refuses a mismatched mix. The gate needs its own commit's
  harness: run each commit's own benchmark for pre-gate commits, and record
  the harness used.
- **Repetitions.** Every number is a median of at least 9 repetitions on an
  idle GPU (Slurm).
- **Cause analysis.** For question 1: the per-lane work histogram from a
  debug build that counts loop iterations per lane into an output buffer
  (no printf, so it also works on old cards). Keep that build out of the
  committed kernels, or behind a compile-time define that is off by
  default.

## Files that may change

`docs/dev/reports/T04.md` (main deliverable); `bench/` helpers if needed. A
kernel change only when question 1 or 2 shows a fixable cause, with its own
commit and a gate `--compare` table; `AGENTS.md` (remove the stale CeO2
item, record the answers in the Status table).

## Acceptance

1. `docs/dev/reports/T04.md` answers each of the 4 questions. Each answer
   has a table of numbers, with medians and IQRs, the GPU, and the commits.
   It says one of:
   - "real, cause X, evidence Y";
   - "noise, run-to-run spread Z covers it";
   - "fixed by commit C".
2. **If a kernel fix lands:**
   - `bench/gate.py --compare` shows the target cell improved;
   - no cell is worse than the T03 thresholds;
   - `sbz test`, `sbz test-kernels` and `run_pac_fixtures.py`
     pass;
   - the baseline is refreshed.
3. The CeO2 item is removed from `AGENTS.md` or restated with the new
   numbers.

## Reviewer checks

- Rerun at least one of the 2 x 2 measurements for question 1 and confirm
  it is within the reported IQR.
- If the report says "noise", check that the stated run-to-run spread
  comes from the `--noise` run, not from an assumption.
