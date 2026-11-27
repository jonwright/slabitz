# T06: zstd feature matrix with proof of coverage, and round trips

Status: done (9de040d)
Executor: worker (DeepSeek), reviewed by Claude
Depends on: T02
Branch: `t06-zstd-features`

## Goal

Prove that every zstd format feature the decoder must handle has actually
been exercised. Build a corpus, classify each entry with the educational
decoder, check that every cell of the feature matrix is non-empty, and
compare the GPU output byte-exact. Add fixed-seed round trips over sizes,
levels and data kinds. This is parts 2 and 3 of the old P5 plan
(`docs/archive/plan_p5_tests.md`), with its settled decisions.

## Background

- **Decoder support** (audit 2026-11-25):
  - supported: raw, RLE and Huffman literals, treeless, 1 and 4 streams,
    all size formats;
  - supported: predefined, RLE, FSE and repeat tables for LL, OF and ML;
  - supported: zero-sequence blocks, skippable frames, checksum (skipped),
    multiple and empty frames.

  No oracle proves which of these the tests reach. The only explicit
  treeless case is `tests/compare_parse_blocks_cases.py`.
- **Corpus source.** All current test inputs come from python-zstandard
  0.25.0 with libzstd 1.5.7 (`pixi.lock`).
- **Oracle.** `tools/edu_oracle.py` builds the zstd educational decoder
  with dump hooks. Tags today: FSE, FSE_C, HUFW, RAW_HUFW, LITS, SEQ, OFFS,
  RLE_B, RLE_L, RLE_M.
- **Limits** that are T09/T10's business, not this task's. Frames without
  Frame_Content_Size, and frames with windows of 2^29 or more, are
  rejected today. Here they are recorded as "expected reject" cells, not
  as coverage failures.

## Read first

`AGENTS.md`; `tools/edu_oracle.py`; `tests/run_chain_frames.py` and
`tests/run_host_walk.py` (the ctypes harness and oracle use);
`tests/test_suite.py` (from T02); RFC 8878 sections 3.1.1.1 (frame
header), 3.1.1.2 (blocks), 3.1.1.3 (literals: raw, RLE, compressed,
treeless; 1 or 4 streams; size formats), 3.1.1.3.2 (sequence table modes).

## Settled design

- **Oracle tags.** Add tags to `tools/edu_oracle.py`, using the hook
  mechanism it already has.
  - Per compressed block:
    - the literals section type (raw, RLE, compressed, treeless);
    - its stream count (1 or 4) and size format (the 2 bits);
    - the LL, OF and ML table modes;
    - the sequence count.
  - Per frame:
    - the checksum flag and the single-segment flag;
    - the FCS field size (0, 1, 2, 4 or 8), and the window descriptor's
      presence and value;
    - the dictionary-ID field size;
    - whether a skippable frame was seen.
- **New `tests/run_features.py`.**
  - **Corpus sources.**
    - python `zstandard` with levels 1-19 and `-22` (ultra);
      `ZstdCompressionParameters` (strategy, window_log, min_match,
      target_length, write_checksum, write_content_size); and the
      streaming compressor (no FCS).
    - Hand-built frames where libzstd will not emit a feature by itself:
      raw and RLE blocks, skippable frames at the start, middle and end,
      0-sequence blocks, a 1-byte FCS, and an 8-byte FCS above 2^32 (header
      only, expected reject `SBZ_ERR_LIMIT`).
  - **Matrix cells:**
    - each literals type x stream count;
    - each size format;
    - each table mode for each of LL, OF and ML;
    - 0-sequence blocks;
    - raw, RLE and compressed blocks;
    - checksum on and off;
    - skippable frames;
    - each FCS field size;
    - single-segment on and off.
  - **Decoding.** Decode through `sbz_dctx_decompress` and
    `sbz_dctx_decompress_frames`, and compare byte-exact with `zstandard`.
    Each entry carries an expectation: `decode` or `reject:<code>`.
    Entries with no FCS, and with a window of 2^29 or more, are
    `reject:SBZ_ERR_CORRUPT` today. Their expectation becomes `decode`
    after T10; give each one a one-line comment pointing to T09/T10.
  - **Output.** Print the matrix with counts. An empty cell is a failure,
    not a skip. The compared byte count must be nonzero.
  - Add it to `sbz test` and to `tests/test_suite.py`. Default
    runtime at most 3 minutes on the L40S.
- **New `tests/run_roundtrip.py`.** Fixed seed:
  - levels 1-19;
  - sizes from 1 byte to 4 MB on a log scale;
  - data kinds: random, text, zeros, runs, and u16 and u32 detector-like
    images with Poisson noise and masks;
  - single frames, and concatenated batches of up to 64 frames per call;
  - byte-exact against `zstandard`.

  The default run takes about 2 minutes; `--long` is a soak run. Add it
  under the pytest `slow` marker, not to `sbz test`.
- **Rules.** Do not change `src/`, `kernels/` or `third_party/`. If a test
  finds a decoder bug, record a minimal reproducer in
  `tests/data/features_bugs/` and in the report, mark the entry
  `xfail:<reason>`, and continue.

## Files that may change

`tools/edu_oracle.py`, `tests/run_features.py`, `tests/run_roundtrip.py`,
`tests/test_suite.py`, `pixi.toml` (test task list),
`tests/data/features_bugs/` (only if bugs are found),
`docs/dev/reports/T06.md`.

## Acceptance

1. `timeout 600 python3 tests/run_features.py` exits 0 and prints the
   matrix. Every cell count is at least 1, the compared bytes are nonzero,
   and there are 0 unexpected results.
2. `timeout 600 python3 tests/run_roundtrip.py` exits 0. It prints the
   number of cases (at least 500) and the compared bytes (nonzero).
3. `CUDA_ARCH=sm_89 sbz test` passes and includes `run_features.py`.
4. The report lists every `xfail` entry, with its reproducer path.

## Reviewer checks

- **Starve the corpus.** Remove the hand-built RLE-literals entries from
  the corpus: `run_features.py` must fail, naming the empty cell.
- **Break a table mode.** In `tools/transform_ds_for_slang.py`, add a
  temporary patch that makes the repeat (mode 3) table path use the
  predefined table. Rebuild: `run_features.py` must fail. Revert and
  rebuild.
