# T03: Benchmark regression gate and stored baselines

Status: todo
Executor: Claude (the harness decides what every later performance claim means)
Depends on: T01, T00 (synthetic and worst-case cells)
Branch: `t03-bench-gate`

## Goal

One command measures the whole decode matrix and compares it with a stored
baseline. It fails on a regression in any cell, not only in the cell a
change was aimed at.

## Background: why

All the numbers below are recorded only in prose, in
`docs/archive/plan_pac_zstd.md` and `docs/history.md`.

- **pac `mask=max-1`.** ExecuteSequences was 5.0x slow. Fixed in b840caa
  with vectorised copies.
- **pac-zstd 0.2.0 merged matches.** ExecuteSequences went from 1776 to
  6570 us/call, 3.7x. Fixed in 692e3f9.
- **"Stock unchanged or slightly faster" was not true.** That commit's own
  table shows stock cases 2-4% slower:
  - silicon_u16_mask0 0.82 -> 0.85 ms/fr
  - silicon_u32_mask0 1.36 -> 1.40 ms/fr
  - ceo2_u32_maskmax-1 4.51 -> 4.54 ms/fr

  The measurement was one run on a GTX 1050 Ti with 4 frames and default
  reps.
- **An unexplained gap remains** after the merged-match fix:
  silicon_u32_maskmax-1 ExecuteSequences is 2321 us against 1776 us with
  pac 0.1 (T04).
- **Wrong conclusions from bad harnesses**, for example:
  - CeO2 per-frame time rising with batch size: the chunks were joined
    inside the timed loop;
  - the LZ4 reference timed on its CPU decoder.

Nothing in the repository stores a baseline or compares with one.
`bench/run_all.py` appends to a CSV that nothing reads.

## Read first

`AGENTS.md` (the Timing rules); `bench/pac_matrix.py`; `bench/p4_overlap.py`;
`bench/old_cards.py`; `src/sbz_cuda.c` (`SLABITZ_PROFILE` output format);
`docs/archive/plan_pac_zstd.md` (the 12-case matrix).

## Settled design

- **The harness, `bench/gate.py`.** It reuses `pac_matrix.py`'s case
  construction where it can.
- **The matrix.** Each cell is one codec x data x dtype x mask x mode, at
  16 frames per call:

  | Axis | Values |
  |---|---|
  | codec | pac-zstd, stock zstd (level 3 and level 19, hdf5plugin-written), pac-LZ4, stock bitshuffle-LZ4 |
  | data | silicon, CeO2, thaumatin |
  | dtype | u16, u32 |
  | mask | `mask=0`, `mask=max-1` |
  | mode | sync; async with two contexts; device input (LZ4) |

  Add the synthetic twins of silicon, CeO2 and thaumatin, and the T00
  worst cases `min_matches`, `short_offsets_1_31`, `random_u16` and
  `all_max` (from `$SLABITZ_SYNTH`). These cells run without the real data,
  and the worst cases guard the slow paths that real data does not reach.
  Build the input once, outside the timed loop. When the example data or
  libpac is missing, the cell is reported `missing`, and `--compare` fails
  on `missing` unless `--allow-missing` is given.
- **Per cell it measures:**
  - the wall time per call, as the median and the interquartile range of
    at least 9 repetitions after 2 warmups;
  - the device time per kernel, from one extra `SLABITZ_PROFILE=1` run
    (that mode syncs per op, so it is kept separate from the wall time);
  - the decoded bytes compared against the CPU decode, which must be
    nonzero.
- **Idle-GPU guard.** Before starting and after finishing, it checks
  `nvidia-smi --query-gpu=utilization.gpu,memory.used` and
  `--query-compute-apps`. It refuses to run when utilisation is above 5% or
  another process holds more than 1 GB, and it prints why.
  `--force-busy` overrides this; the output is then marked "not a baseline".
- **Output.** `bench/results/<gpu-slug>-<commit>-<date>.json`, with:
  - the GPU name, driver, CUDA version and SM count;
  - the commit, `SBZ_KERNELS_ID` and the kernel directory;
  - the frames per call;
  - the cells.

  The results directory is gitignored except for baselines.
- **Baselines.** `bench/baselines/<gpu-slug>.json` is committed. It is
  written by `bench/gate.py --write-baseline` and must come from a run on
  an idle GPU.
  - **L40S:** made by the committed Slurm script `bench/gate.sbatch`
    (`--partition=low-gpu --gres=gpu:nvidia_l40s:1`). Its output goes under
    `$SLABITZ_DEV_ROOT/results/`, never into `data/`. `gpu-run` (T-1) is
    the interactive equivalent of the sbatch script.
  - **GTX 1050 Ti and TITAN Black:** the user adds these when the hardware
    is available.
- **Compare.** `bench/gate.py --compare [--baseline FILE]` prints one table
  of cells: baseline, now, ratio. It exits nonzero when either:
  - the wall time is more than 5% slower and the difference exceeds 3x the
    larger of the two IQRs; or
  - any kernel's device time is more than 10% slower and above 20 us.

  Improvements of more than 5% are listed, so that the baseline gets
  refreshed in the same PR. The thresholds are named constants at the top
  of the file.
- **Noise check.** `bench/gate.py --noise` runs the matrix twice and
  reports the largest run-to-run ratio per cell. It runs once on the L40S
  when the baseline is made; the result goes in the report and sets
  whether the 5% threshold is achievable.
- **Rules added to AGENTS.md:**
  - A PR touching `kernels/`, `src/` or `tools/transform_ds_for_slang.py`
    includes the `--compare` table in its report.
  - A PR that is faster refreshes the baseline in the same commit.
  - A cell that regresses needs an explicit, accepted reason in the report.

## Files that may change

`bench/gate.py`, `bench/gate.sbatch`, `bench/baselines/*.json`, `.gitignore`
(`bench/results/`), `bench/pac_matrix.py` (factor out shared helpers only),
`AGENTS.md` (Timing rules), `docs/dev/reports/T03.md`.

## Acceptance

1. `python3 bench/gate.py --list` prints the matrix, with at least 24 cells.
2. On the L40S via `sbatch bench/gate.sbatch`:
   - `bench/baselines/l40s.json` exists;
   - no cell is `missing`;
   - every cell has a nonzero compared-byte count.
3. `python3 bench/gate.py --noise` (in Slurm) gives a largest wall ratio of
   at most 1.03 in at least 90% of the cells. The report lists the
   exceptions.
4. `python3 bench/gate.py --compare` on the same commit exits 0.
5. **Injected slowdown.** `SLABITZ_GATE_SELFTEST=slow` makes the harness
   add a 10% sleep inside the timed loop of one named cell. With it,
   `--compare` exits nonzero and names that cell. This hook exists only in
   `gate.py`, not in the library.
6. **Busy GPU.** Start a background GPU load (any CUDA busy loop, for
   example `bench/gate.py --burn 30`, which the gate provides for this
   purpose), then run the gate without `--force-busy`. It must refuse with
   a message.

## Reviewer checks

- Read `gate.py` and check that the input is built outside the timed loop
  and that compared bytes come from a real comparison.
- Rerun acceptance 4 and 5 yourself.
- Check that `bench/baselines/l40s.json` records the commit, kernel ID and
  the Slurm node name.

## Out of scope

Making anything faster. T04 explains the open questions using this gate.
