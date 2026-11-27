# T05: Portable kernels and old cards re-check

Status: todo
Executor: worker for the builds; the user provides access to the cards
(crunch for sm_35; the machines with the sm_50 960M and the sm_61 1050 Ti)
Depends on: T01
Branch: `t05-old-cards`

## Goal

Make the committed kernel sets match the current sources on every supported
card, from sm_35 to sm_90, after the `pac-zstd` and `matchcopy-overlap`
kernel edits. sm_50 has never been run.

## Background

- **Two kernel sets.**
  - `build/sm_35` is built with `sbz build-sm35` (NVRTC
    11.8).
  - `build/kernels` (22 fatbins: sm_35, sm_50..sm_90, compute_90 PTX) is
    built with `sbz build-fatbins`, after the sm_35 set.
- **Which kernel edits each set has seen:**
  - The sm_61 local cubins passed all 11 test scripts and
    `bench/pac_matrix.py` before 692e3f9.
  - Neither `build/sm_35` nor `build/kernels` has been rebuilt or
    spot-checked since the `pac-zstd` kernel edits (a0338bf, b840caa) and
    692e3f9.
- **Fermi is not supported.** sm_20 (crunch's Quadro 600) is the
  documented floor failure: the library must fail there cleanly with
  `SBZ_ERR_KERNEL` ("no kernel image").
- **Procedures.** `AGENTS.md`, section "Pre-Volta (crunch)", has the exact
  commands for crunch: `make -q` must be clean on the login node first,
  CUDA device 0 is the TITAN Black, and NVRTC 11.8 kernels with `printf` do
  not load under driver 340.

## Steps

1. On the login node: `sbz build && sbz build-sm35 &&
   sbz build-fatbins`. Check that every directory's `sbz_kernels.id`
   equals `SBZ_KERNELS_ID` in `src/sbz_kernels_gen.h`.
2. L40S: `bash tests/run_suite_kernel_dir.sh build/kernels`, and
   `SLABITZ_KERNEL_DIR=build/kernels sbz test-kernels`.
3. crunch (TITAN Black):
   - `. $SLABITZ_DEV_ROOT/env.sh && cd $SLABITZ_DEV_ROOT/wt/t05-old-cards &&
     PYTHON=python3 bash tests/run_suite_kernel_dir.sh $PWD/build/kernels`
     (the root is on GPFS, which crunch mounts);
   - `check_bshuf.py --kernels build/sm_35`;
   - `run_pac_fixtures.py` with each of the two kernel directories.
4. sm_61 (1050 Ti) and sm_50 (960M): the suite with `build/kernels`;
   `run_pac_fixtures.py`; `bench/gate.py` (to make a baseline for that
   card, if the user wants one committed).
   On the 1050 Ti, also rerun T04 question 1 (`docs/dev/reports/T04.md`):
   the original "+30% ExecuteSequences" on `silicon_u32_maskmax-1` came
   from this card and was not reproduced on the L40S. Run the Q1 loop of
   `bench/t04_runs.sbatch q1q2` (without the Slurm lines) with
   `wt/t04-at-b840caa`, `wt/t04-at-a0338bf` and this checkout, and with
   pac-zstd 0.1 and 0.2 frames (rebuild 0.1 first, as described in the
   T04 report's Method, at `$T04_PAC01`); report the 3 x 2 table of
   ExecuteSequences us made by `bench/t04_tables.py`.
5. Fermi: run one test with `build/kernels` and confirm the clean
   `SBZ_ERR_KERNEL` failure message.

## Files that may change

`docs/dev/reports/T05.md`; `AGENTS.md` (Status rows for pre-Volta,
portable kernels, sm_50); `bench/baselines/*.json` (optional, one per card).
No source change is expected. If a card fails, record a minimal
reproducer and stop: the fix is a separate task.

## Acceptance

1. The report has one table: card x kernel directory x script, each cell
   PASS, FAIL or not run with a reason. It covers L40S, TITAN Black, 1050 Ti
   and 960M.
2. Every cell that ran is PASS, and `run_pac_fixtures.py` compared nonzero
   bytes on each card.
3. The Fermi line shows `SBZ_ERR_KERNEL` and no crash.
4. `AGENTS.md` Status rows record the date, and the commit and kernel ID
   that were tested.

## Reviewer checks

- Confirm that the kernel ID printed in the report equals the one in
  `src/sbz_kernels_gen.h` at the merged commit.
- On the L40S, rerun step 2 yourself.
