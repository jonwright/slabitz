# T02: Test harness: pytest entry, required data, pac fixtures

Status: review
Executor: worker (DeepSeek), reviewed by Claude
Depends on: T01, T00 (its committed subset `tests/data/synth/` joins `sbz test` here)
Branch: `t02-test-harness`

## Goal

Make "the tests pass" mean something:

- one pytest entry point over all the scripts;
- missing example data becomes a failure when data is required;
- the pac-specific fast paths in ExecuteSequences are exercised by committed
  fixtures inside `sbz test`, instead of only by the manual
  `bench/pac_matrix.py`.

## Background

- `sbz test` runs 11 scripts (`pixi.toml:44-59`). The real-file
  checks in `tests/run_bshuf_chain.py:353-360` and
  `tests/run_bslz4_chain.py:710-717` print `ok ... skipped` when
  `SLABITZ_EXAMPLES` is unset. `bench/pac_matrix.py:118-119` returns 0 when
  its data is missing.
- **Where the pac paths are tested today.** The three ExecuteSequences fast
  paths that pac data needs are:
  - `off == 1` fills;
  - non-overlapping `off == P` copies (P = plane bytes);
  - merged `len > off` matches (pac-zstd 0.2.0 sequence merging, fixed in
    692e3f9).

  Only `bench/pac_matrix.py` exercises them, and it needs `libpac.so` from
  `../pac-zstd` plus the example files.
- **Existing fixtures.** Small fixtures are already committed in `data/`
  (`u16-256k.chunk.bin` plus `.elem` plus `.expected.bin`). The new pac
  fixtures follow this pattern, but store a SHA-256 of the expected output
  instead of the output itself, to keep the repository small.
- **The earlier P5 plan** (`docs/archive/plan_p5_tests.md`, "pytest entry
  point") settled the pytest design; it is repeated below.

## Read first

`AGENTS.md`; `pixi.toml`; `tests/run_bshuf_chain.py`;
`tests/run_bslz4_chain.py`; `bench/pac_matrix.py` (how it calls libpac);
`tools/edu_oracle.py` (the SEQ tag dumps sequences); `data/`.

## Settled design

- **pytest entry, `tests/test_suite.py`.**
  - One pytest test per script in `sbz test`, plus one per non-retired
    script in `sbz test-kernels` (from T01). Each runs as a subprocess
    under a timeout and fails on a nonzero exit, showing the script's
    output on failure.
  - Markers: `gpu` (needs a CUDA device) and `slow` (more than 60 s).
  - `sbz pytest` (and the pixi task of the same name, which calls
    `tools/sbz.sh`). pytest is in the shared env from T-1; do not change
    the env. The scripts stay runnable on their own (crunch runs them
    without pytest).
- **`SLABITZ_REQUIRE_DATA=1`.** Every place that would print "skipped"
  for missing example data or a missing libpac must instead exit nonzero
  with a message naming the missing file. The default stays as it is
  today, so that a fresh clone still passes. `tests/test_suite.py` sets
  `SLABITZ_REQUIRE_DATA=1` when `SLABITZ_EXAMPLES` is set.
- **pac fixtures.**
  - `tools/gen_pac_fixtures.py` uses `libpac.so` (the `PAC_LIB` env, as in
    `pac_matrix.py`) to encode frames from the example files. It writes
    `data/pac/<case>.chunk.bin` (filter-32008 chunks, several frames
    concatenated with an index file `<case>.index`: offsets and lengths),
    `<case>.json` (elem size, block size, frame bytes, the SHA-256 of each
    decoded frame, the pac-zstd version and commit) and nothing else.
  - Cases:
    - silicon u16 and u32, `mask=0` and `mask=max-1`, zstd;
    - CeO2 u32, `mask=max-1`, zstd;
    - silicon u16, LZ4;
    - one all-zero frame (RLE empty frames).
  - Keep `data/pac/` under 3 MB in total: crop frames, or use fewer frames.
    Each case must still contain all three match kinds listed under
    "Acceptance".
- **New `tests/run_pac_fixtures.py`, added to `sbz test`.** It
  decodes each fixture through `sbz_dctx_bshuf_decompress` /
  `sbz_dctx_bslz4_decompress`, sync and async, and compares the SHA-256 of
  each frame. It also decodes on the CPU with `bitshuffle`/`hdf5plugin` and
  compares with that, so the fixtures cannot carry a wrong hash.
- **Proof of coverage.** `tests/run_pac_fixtures.py --stats` uses the
  SEQ dumps from `tools/edu_oracle.py` to count, per zstd case, the
  sequences with `off == 1`, `off == P` with `len <= off`, and `off == P`
  with `len > off`. It prints the counts.
- **Audit of "compared nothing".** Grep every script in `sbz test`
  for its final PASS line, and make sure each prints a compared count and
  fails on zero. List in the report any script you had to fix.

## Files that may change

`tests/test_suite.py`, `tests/run_pac_fixtures.py`,
`tools/gen_pac_fixtures.py`, `data/pac/*`, the "skipped" branches in
`tests/*.py` and `bench/pac_matrix.py`, `pixi.toml` (task list only), `docs/dev/reports/T02.md`.

## Acceptance

1. `timeout 3600 sbz pytest -m "not slow"` passes, with one test per
   script.
2. `SLABITZ_EXAMPLES=/nonexistent SLABITZ_REQUIRE_DATA=1 python3
   tests/run_bshuf_chain.py` exits nonzero, naming the missing file.
   Without `SLABITZ_REQUIRE_DATA` it still exits 0.
3. `du -sh data/pac` is at most 3 MB.
4. `python3 tests/run_pac_fixtures.py --stats` prints, for every zstd pac
   case, nonzero counts in all three columns (`off==1`, `off==P len<=off`,
   `off==P len>off`). It also prints the compared bytes, nonzero, for both
   the GPU-vs-hash and the CPU-vs-hash comparisons.
5. `CUDA_ARCH=sm_89 sbz test` passes and includes
   `run_pac_fixtures.py`.

## Reviewer checks

- **Disable the merged-match path.** In `tools/transform_ds_for_slang.py`,
  make the merged `len > off` aligned copy take the byte fallback with a
  deliberate off-by-one in the source index. Rebuild, then run
  `run_pac_fixtures.py`: it must FAIL.
- **Corrupt a hash.** Flip one byte in a `<case>.json` hash: the
  CPU-vs-hash check must FAIL.

## Out of scope

The zstd feature matrix and round trips (T06), corrupt inputs (T11).
