# Plan: P5, test suite

> Archived 2026-11-25. Folded into the task files in `docs/dev/` (P4: T12-T16;
> P5: T02, T06, T11). Kept for its history and settled decisions; the task
> files are the instructions.

The P5 tests item in `AGENTS.md`. Today the tests are standalone scripts run by
`pixi run test` / `build_test.sh`. They check real data and hand-picked
cases well, but nothing proves that every zstd feature the decoder must
handle was actually exercised. This job adds a pytest entry point, a
feature matrix with proof of coverage, random round trips, corrupt
compressed blocks on the GPU, and compute-sanitizer runs. Independent of
P4 (it tests the current `sbz_dctx_*` API); can run in parallel with P4
step 1. Suitable for the worker, with these decisions settled.

## Read first

1. `AGENTS.md`, especially "A test that compares nothing must fail".
2. `tests/` (every script; `run_host_walk.py` and `run_chain_frames.py`
   show the ctypes harness and the edu oracle use).
3. `tools/edu_oracle.py` (builds the zstd educational decoder with dump
   hooks; tags today: FSE, FSE_C, HUFW, RAW_HUFW, LITS, SEQ, OFFS, RLE_B,
   RLE_L, RLE_M).
4. RFC 8878 sections 3.1.1.3 (literals section: raw, RLE, compressed,
   treeless; 1 or 4 streams; size formats) and 3.1.1.3.2 (sequence table
   modes: predefined, RLE, FSE-compressed, repeat).

## Design (settled)

### pytest entry point

- `tests/test_suite.py`: one pytest test per existing script, run as a
  subprocess under a timeout, failing on a nonzero exit, and showing the
  script's output on failure. Markers: `gpu` (needs a CUDA device),
  `slow` (> 60 s). `pixi run pytest` task in the default environment
  (pytest is added to `pixi.toml` dependencies only if not already there;
  if adding it needs a `pixi.lock` change, write QUESTIONS.txt instead).
- The scripts stay runnable on their own (crunch runs them without pytest).

### Feature matrix with proof of coverage

- Add oracle tags in `tools/edu_oracle.py` (the hook mechanism it already
  uses) for, per compressed block: the literals section type (raw, RLE,
  compressed, treeless), its stream count (1 or 4) and size format, the
  three sequence table modes, and the sequence count (to find 0-sequence
  blocks). Also per frame: checksum flag, single-segment flag, content size
  field size, and skippable frames seen.
- `tests/run_features.py`: builds a corpus with python `zstandard`
  (levels, `ZstdCompressionParameters` such as strategy, window_log,
  min_match, target_length, write_checksum, write_content_size) and, where
  libzstd will not emit a feature on its own, hand-built frames (raw and
  RLE blocks, skippable frames, 0-sequence blocks). Each corpus entry is
  classified with the new oracle tags; the test then asserts that every
  cell of the matrix (each literals type x stream count; each table mode
  for each of LL, OF, ML; 0-sequence blocks; raw, RLE and compressed
  blocks; checksum and skippable frames; each content size field size) has
  at least one entry, decodes the corpus through `sbz_dctx_decompress` and
  `sbz_dctx_decompress_frames`, and compares byte-exact with `zstandard`.
  It prints the matrix with counts; an empty cell is a failure, not a skip.

### Random round trips

- `tests/run_roundtrip.py`: fixed seed, levels 1-19, sizes from 1 byte to
  4 MB on a log scale, data kinds (random, text, zeros, runs, u16 and u32
  detector-like images with Poisson noise), single frames and concatenated
  batches of up to 64 frames per call. Byte-exact against `zstandard`.
  Default about 2 minutes; `--long` for a soak run.

### Corrupt compressed blocks on the GPU

- The host walk is fuzzed already (`tests/fuzz_host_walk.c`). The GPU
  kernels still trust compressed-block contents. `tests/run_gpu_corrupt.py`:
  valid frames with bytes flipped only inside compressed blocks (offsets
  from the host walk), each decoded in a subprocess under a timeout.
  Required: no hang, no crash, no out-of-bounds write (checked with
  compute-sanitizer, below); the result is an error code or some output.
  Wrong output without an error is recorded, not failed (zstd checksums
  are not verified on the GPU). Report every hang or crash with the input
  saved under `build/corrupt_cases/`; fixing them is a separate job.

### compute-sanitizer

- It is not in the pixi environment; the ESRF module has it
  (`bash -lc 'module load jupyter-slurm && which compute-sanitizer'`).
  First check that it runs on the L40S node (ncu does not: performance
  counters are admin-only; the sanitizer may not need them). If it cannot
  run, record that in the report and skip this part.
- `tests/run_sanitizer.sh`: memcheck and racecheck over a small case of
  each chain test (one zstd, one bitshuffle-zstd, one bitshuffle-LZ4 fused
  and unfused) and over `run_gpu_corrupt.py` with a few hundred inputs.
  Zero errors required; list every report otherwise.

## Rules

- ASCII only. Do not change `src/`, `kernels/` or `third_party/`: this job
  only adds tests and oracle tags. If a test finds a bug, record it with a
  minimal reproducer; do not fix it here.
- Every new test prints what it compared and fails on zero compared.
- Keep the default runtime of the new tests under 10 minutes on the L40S.

## Deliverables

One commit per part (pytest entry, feature matrix, round trips, GPU corrupt
inputs, sanitizer), and `REPORT_p5.md`: the coverage matrix with counts, the
round-trip totals, every corrupt-input finding, the sanitizer results.
