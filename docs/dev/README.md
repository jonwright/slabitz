# slabitz development plan (from 2026-11-25)

This directory is the single actionable plan. It replaces the "Next steps"
list that used to be in `AGENTS.md`, and the plans now in `docs/archive/`
(`plan_p4_api.md`, `plan_p5_tests.md` and the readback audit were folded in
here; the others were finished). Each task has its own file, `Tnn_*.md`
(T-1 and T00 come first: the shared agent root and the synthetic data).
Every task file stands alone: it can be executed after a context reset by
someone who has read only `AGENTS.md`, this file and the task file.

## Order and status

Do the tasks roughly in order. A task may start once the tasks in its
"Depends on" line are merged. T-1 (the agent root) comes before
everything: until it is done, agents work as before, in
`/data/id11/jon/hdftest/slabitz-workspace`. After T05, four tracks can run in parallel:
correctness (T06-T11), API (T12-T16), performance (T20-T21) and encoder
(T17-T19).

| # | Task | Executor | Depends on | Status |
|---|---|---|---|---|
| T-1 | [Shared agent root: one env, shared data, one worktree per task](T-1_agent_root.md) | Claude | - | review |
| T00 | [Synthetic test data matched to real data, plus worst cases](T00_synthetic_data.md) | Claude (models) + worker | T-1 | todo |
| T01 | [Kernel argument checks for every Python launcher](T01_kernel_arg_checks.md) | Claude | T-1 | todo |
| T02 | [Test harness: pytest entry, required data, pac fixtures](T02_test_harness.md) | worker | T01, T00 | todo |
| T03 | [Benchmark regression gate and stored baselines](T03_bench_gate.md) | Claude | T01, T00 | todo |
| T04 | [Explain the open performance questions](T04_perf_open_questions.md) | Claude | T03 | todo |
| T05 | [Portable kernels and old cards re-check](T05_old_cards.md) | worker + user (hardware) | T01 | todo |
| T06 | [zstd feature matrix with proof of coverage, round trips](T06_zstd_feature_matrix.md) | worker | T02 | todo |
| T07 | [zstd version corpus: bitshuffle 0.4.0's zstd to 1.6.0](T07_zstd_version_corpus.md) | worker | T06 | todo |
| T08 | [pac-zstd: lower the compat floor (other repo)](T08_pac_zstd_floor.md) | worker | - | todo |
| T09 | [Decoder: matching rejects, error codes, bounds](T09_decoder_rejects_bounds.md) | Claude | T07 | todo |
| T10 | [Decoder: frames without a content size](T10_no_fcs_frames.md) | Claude | T09 | todo |
| T11 | [Corrupt compressed blocks on the GPU, compute-sanitizer](T11_gpu_corrupt_sanitizer.md) | worker (fixes: Claude) | T02 | todo |
| T12 | [P4 step 2c for zstd: chunk walk on the device](T12_zstd_device_walk.md) | Claude | T03 | todo |
| T13 | [Public C API `include/slabitz.h`](T13_public_c_api.md) | Claude | T12 | todo |
| T14 | [Python package `slabitz`](T14_python_package.md) | worker | T13 | todo |
| T15 | [bslz4_to_sparse decoder hook (other repo)](T15_bslz4_to_sparse_hook.md) | Claude + worker | T14 | todo |
| T16 | [Wheel](T16_wheel.md) | worker | T05, T14 | todo |
| T17 | [Encoder shared stages: GPU bitshuffle and plane-aware parse](T17_encoder_shared_stages.md) | Claude | T03 | todo |
| T18 | [LZ4 encoder](T18_lz4_encoder.md) | worker (after T17 contracts) | T17 | todo |
| T19 | [zstd encoder](T19_zstd_encoder.md) | Claude | T18, T07 | todo |
| T20 | [ExecuteSequences parallelism](T20_execute_sequences_parallel.md) | Claude | T03, T04 | todo |
| T21 | [Other decoder performance items](T21_perf_other.md) | Claude | T03 | todo |
| T22 | [Slang target surveys (CPU, Metal, SPIR-V)](T22_slang_target_surveys.md) | worker | - | todo |

Executors:

- **Claude** means the task is subtle: C runtime, concurrency, kernels,
  or design.
- **worker** means DeepSeek through opencode, as described in the user's
  CLAUDE.md. After T-1 it runs with its working directory at
  `$SLABITZ_DEV_ROOT`, in the task worktree `wt/<task>`. The task file is then also the worker's
  brief. Claude reviews the result and re-verifies it before the PR.

## How to execute a task (for the agent doing it)

1. **Start in the root.** After T-1, start at `$SLABITZ_DEV_ROOT`: read
   `AGENT_START.md`, then `. env.sh`, then `cd wt/<task>`. The reviewer has
   already created the worktree with `dev-root new <task>`. Use `sbz build`,
   `sbz test` and so on. Never run pixi, pip or conda, never download, and
   never read or write outside the root. Real data is in
   `$SLABITZ_EXAMPLES`, synthetic data in `$SLABITZ_SYNTH`. Time only through
   `gpu-run`. (The task files say `sbz <task>`; before T-1 that means
   `pixi run <task>` in the old workspace.)
2. **Read first.** Read `AGENTS.md` (all of it), this file, and the task
   file. Then read the files the task lists under "Read first".
3. **Branch.** Work on the worktree's branch `tNN-<short-name>`, made from
   the current `main` (or from the dev branch named in the task). Never
   push: the user does all pushes.
4. **Questions.** This work is non-interactive. Decide small choices
   yourself and record each one in the report. If a design choice is not
   settled by the task file, do NOT guess. Write numbered questions to
   `QUESTIONS.txt`, finish the parts that do not depend on the answers,
   and stop.
5. **Scope.** Keep to "Files that may change". Anything else needs a line
   in the report saying why.
6. **Kernel changes.** If you change a kernel source, follow the
   `AGENTS.md` rules. Run `tools/build_all.py`, commit the regenerated
   `src/sbz_kernels_gen.h`, and say whether `build/sm_35` and
   `build/kernels` need rebuilding.
7. **Report.** Write `docs/dev/reports/TNN.md` with these parts:
   - what was done, one commit per logical part
   - each acceptance check: the exact command, its output (the numbers),
     and pass or fail
   - every decision you took
   - every deviation from the task file
   - open issues
   A check that could not run is reported as "not run" with the reason,
   never as "pass".
8. **Status.** Set the task's Status line to `review`, and update the
   Status column in the table above.

## How to verify a task before merging (for the reviewer)

The reviewer may have no memory of the task. Do not trust the report: rerun.

0. **Where.** After T-1, review in the root: `cd $SLABITZ_DEV_ROOT/wt/<task>`
   (the agent's worktree), or make a fresh one from its branch with
   `dev-root new review-<task> <task>`. Run `dev-root update-main` first,
   and after the merge, `dev-root rm <task>`.
1. **Diff scope.** Read the task file, then `git diff main...tNN-*`. Every
   changed file must be listed under "Files that may change" or explained
   in the report.
2. **Rerun the checks.** Rerun every command in the task's "Acceptance"
   section yourself, on an idle GPU (see "Measurement rules" below). Compare
   the numbers with the report. A difference of more than the stated
   tolerance is a finding.
3. **Mutation checks.** Run the "Reviewer checks" listed in the task. These
   are deliberate breakages that must make the new tests fail, so they show
   the tests can detect a fault. Revert each one afterwards.
4. **Full suite.** Run `CUDA_ARCH=sm_89 sbz test`. Once T01 is merged,
   also run `sbz test-kernels`. Once T03 is merged, and the task touches
   `kernels/`, `src/` or `tools/transform_ds_for_slang.py`, also run
   `python3 bench/gate.py --compare`.
5. **Worker drafts.** If the worker drafted the change, the commit message
   says so.
6. **After merging.** Set Status to `done (<merge commit>)`, add an entry to
   `docs/history.md`, and update the `AGENTS.md` Status table if a row
   changed.

## Measurement rules (all tasks)

- **Timeouts.** Run every GPU command under `timeout`. A hang is a failure.
- **Idle GPU.** Before any timing, check `nvidia-smi`: utilisation near 0
  and no large foreign process. The login node's L40S is shared. Use a
  Slurm job (`gpu-run <cmd>` after T-1, i.e. `--partition=low-gpu
  --gres=gpu:nvidia_l40s:1`) for numbers that go into a baseline or a
  report.
- **Correctness numbers.** Report compared byte counts. A check that
  compared zero bytes is a failure.
- **Timing numbers.**
  - Use the median of at least 7 repetitions after warmup.
  - Build the input once, outside the timed loop.
  - Report device time per kernel (`SLABITZ_PROFILE=1`) and wall time per
    call.
- **Performance claims.** Before and after numbers come from
  `bench/gate.py` (after T03) over the whole matrix, never from the one
  case a change targets. "Unchanged" means within the gate's tolerance
  in every cell.
