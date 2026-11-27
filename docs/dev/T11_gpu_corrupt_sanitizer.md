# T11: Corrupt compressed blocks on the GPU, and compute-sanitizer

Status: todo
Executor: worker (finding problems); Claude (fixing them, as follow-up tasks)
Depends on: T02
Branch: `t11-gpu-corrupt`

## Goal

The host walk is fuzzed and bounds-checked, but the GPU kernels still trust
the contents of compressed blocks. Find out what corrupt compressed blocks
do on the GPU: hangs, crashes, out-of-bounds writes. Run the
compute-sanitizer over the chain tests. This is parts 4 and 5 of the old P5
plan (`docs/archive/plan_p5_tests.md`).

## Background

- `tests/fuzz_host_walk.c` covers the host walk: ASan/UBSan, 200k mutants,
  run under `setarch $(uname -m) -R`.
- The LZ4 decoder already has bounds checks and a per-chunk status.
- **Hangs.** A wave-divergence bug shows up as a hang. Run every GPU
  command under `timeout`.
- **Sanitizer.** compute-sanitizer is not in the pixi environment. The
  ESRF module may have it: `bash -lc 'module load jupyter-slurm && which
  compute-sanitizer'`. ncu does not work on the L40S node
  (`RmProfilingAdminOnly: 1`), and the sanitizer may hit the same limit:
  check first.

- **Known faults from T06** (`docs/dev/reports/T06.md`). Both gave an
  illegal memory access (CUresult 700), not an error code or wrong bytes:
  - a malformed RLE-literals section: size format 0 (a 5-bit
    regenerated size, at most 31) with a declared size of 500, so the
    section is shorter than the frame's content size (an early version of
    `hand-rle-literals-sf0`; section "The development-time context
    crash");
  - a wrong FSE table index: the repeat-mode (mode 3) path of
    `PARSE_FSE_TABLE` pointed at the predefined table (a temporary
    transform edit, reviewer check #2), which faults on `zstd-u32-l8`.
    The input-side equivalent is a sequence section whose mode byte says
    "repeat" where the previous table differs, or where there is none.
- **One fault ends the process.** The library uses the device's primary
  CUDA context, so after an illegal memory access no later call in the
  same process can run, even with a new dctx. Hence one subprocess per
  input below.

## Read first

`AGENTS.md`; `src/host_walk.c` (how to find compressed-block byte ranges);
`tests/run_host_walk.py`; `tests/run_bslz4_chain.py` (the corrupt-input
cases for LZ4); `docs/archive/plan_p5_tests.md`.

## Settled design

- **New `tests/run_gpu_corrupt.py`.**
  - Take valid frames: stock zstd at levels 3 and 19, pac zstd fixtures
    from T02, and bitshuffle-LZ4 chunks.
  - Flip bytes only inside compressed blocks, using block offsets from a
    Python port of the walk or from `sbz_dctx_parse` output, so the host
    walk accepts the input.
  - Mutation kinds: single bit flip, single byte set to 0x00 or 0xFF, a
    run of 8 random bytes, and truncating the block's last 4 bytes
    (adjusting the header size).
  - Two named seeds from T06, before the random ones: an RLE-literals
    section whose regenerated size disagrees with the frame content size
    (built like `hand-rle-literals-sf*` in `tests/run_features.py`), and a
    compressed block whose sequence-section mode byte is changed to
    repeat (3) for each of LL, OF and ML in turn, on the first block of a
    frame (no previous table) and on a later block.
  - Decode each input in a subprocess under `timeout 20`.
  - **Required:** no hang, no crash (the process exit code), and no
    out-of-bounds write. Pad the output allocation with canary bytes, and
    use the sanitizer below.
  - **Accepted results:** an error code, or some output. Wrong output
    without an error is counted, not failed, because zstd checksums are not
    verified on the GPU.
  - A fixed seed; at least 2000 inputs by default; `--long` for a soak
    run.
  - Every hang, crash or canary hit is saved under
    `build/corrupt_cases/<kind>-<n>.bin` with a one-line description.
- **New `tests/run_sanitizer.sh`.** memcheck and racecheck over a small
  case of each chain test (one zstd, one bitshuffle-zstd, one
  bitshuffle-LZ4 fused, one unfused), and over `run_gpu_corrupt.py` with
  200 inputs. If the sanitizer cannot run on the node, record the exact
  error and skip this part (that is a result, not a failure).
- **Rules.** Do not change `src/`, `kernels/` or `third_party/`. Fixes are
  follow-up tasks: Claude files one task file per distinct root cause,
  `T11a_*.md` and so on, in `docs/dev/`, with the reproducer.

## Files that may change

`tests/run_gpu_corrupt.py`, `tests/run_sanitizer.sh`, `tests/test_suite.py`
(the slow marker), `.gitignore` (`build/corrupt_cases/`),
`docs/dev/reports/T11.md`.

## Acceptance

1. `timeout 3600 python3 tests/run_gpu_corrupt.py` completes and prints:
   - the counts of inputs, error-code results, wrong-output-without-error
     results, hangs, crashes and canary hits;
   - the total inputs (at least 2000).
2. The report lists every hang, crash and canary hit, with its saved
   input. The same counts must reproduce when the script is rerun with the
   same seed.
3. `bash tests/run_sanitizer.sh` gives 0 errors, or a list of every
   report, or the documented reason why it cannot run.
4. Claude has written follow-up task files (T11a...) for each distinct
   root cause, or the report states that there are none.

## Reviewer checks

- Rerun acceptance 1 with the same seed: the counts must be identical.
- Replay one saved crash or hang input by hand, if there is any.
