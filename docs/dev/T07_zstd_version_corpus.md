# T07: zstd version corpus: bitshuffle 0.4.0's zstd to 1.6.0

Status: todo
Executor: worker (DeepSeek), reviewed by Claude
Depends on: T06 (oracle tags and the expectation mechanism)
Branch: `t07-zstd-versions`

## Goal

Show that the GPU decoder reads what every relevant zstd encoder writes.
The range runs from the zstd bundled in the first bitshuffle release with
zstd support up to zstd 1.6.0. Every frame must either decode byte-exact or
be rejected with a documented code. The cells that fail today become the
input to T09 and T10.

## Background

- **The floor.** bitshuffle 0.4.0 ("Add ZSTD support", 9ba2580,
  2022-02-24) bundles zstd as a git submodule at commit
  `18d02cbf2e0654de08093094f1a77cfd231f11d7`. `git describe` gives
  `v1.4.7-403-g18d02cbf`, dated 2021-05-27: a development snapshot between
  1.4.9 and 1.5.0. Files written by bitshuffle 0.4.0 through
  `bshuf_compress_zstd` use that encoder. Its LZ4 is 1.9.x. This floor is
  what the user asked for ("the version used in the first release of
  bitshuffle-zstd").
- **zstd 1.6.0** (verify the details against the upstream CHANGELOG and
  record them in the report):
  - legacy-format support (pre-0.8) is off by default, which does not
    affect us;
  - the format spec clarifies Block_Maximum_Size;
  - it fixes a streaming-decompression CVE in libzstd, which is not our
    code;
  - it adds `ZSTD_compressSequencesAndLiterals`, which is relevant to the
    T19 encoder.
- **Today's corpus.** Every test input comes from libzstd 1.5.7. Nothing
  covers `--long`, `-T`, streaming (no FCS) or dictionary frames.
- **Known decoder limits**, from the 2026-11-25 audit:
  - frames without FCS are rejected;
  - windows of 2^29 or more are rejected, by the GPU parser only;
  - a dictionary ID is rejected;
  - Block_Maximum_Size is not enforced;
  - the output cap is 1 GiB per call.
- **Version builds already exist** in pac-zstd's `tests/check_compat.sh`,
  which fetches and builds several zstd versions offline. Reuse its
  pattern. pac-zstd is in `../pac-zstd`; it is the same repository as
  `../plane_aware_compressors`.

## Read first

`AGENTS.md`; `tests/run_features.py` and `tools/edu_oracle.py` (after T06);
`../pac-zstd/tests/check_compat.sh` and `../pac-zstd/tests/gen_compat_corpus.c`;
`src/host_walk.c` (frame header handling).

## Settled design

- **Builder: `tools/gen_compat_corpus.py`.**
  - It builds the zstd CLI and libzstd for each version. Build directories
    and output go under `$SLABITZ_WS/tmp/compat/`, never into the checkout
    or `workspace/examples`.
  - Versions: `v1.4.7`, `18d02cbf` (bitshuffle 0.4.0's), `v1.4.9`,
    `v1.5.0`, `v1.5.2`, `v1.5.5`, `v1.5.6`, `v1.5.7` and `v1.6.0`.
  - It also builds bitshuffle 0.4.0 itself (its 0.4.0 commit with that
    submodule) into a venv under `$SLABITZ_WS/tmp/compat/`, to write
    filter-32008 zstd chunks with h5py, as a detector pipeline of 2022
    would.
- **Inputs:**
  - detector images (silicon u16, CeO2 u32 from `SLABITZ_EXAMPLES`,
    cropped);
  - text (any public-domain text file committed in tests, or generated);
  - random, zeros, and a 300 MB file for the window cases (generated, not
    stored).
- **Modes per CLI version:**
  - levels 1, 3, 9, 19 and `--ultra -22`;
  - `--long=27`, `--long=30`, `--long=31` (the window cases);
  - `-T0` (multithreaded);
  - streaming: the input through stdin, which writes no FCS;
  - `--no-content-size`, `--check`/`--no-check`;
  - `-B<size>`;
  - `--patch-from`, and a trained dictionary (expected reject).

  A mode the old CLI lacks is recorded as `n/a`, not as a failure.
- **Classification and runner.** Each file is classified with the T06
  oracle tags. A stored manifest gives, per file, the version, mode,
  expectation (`decode` / `reject:<code>`) and SHA-256 of the decoded
  output.
- **Runner: `tests/run_compat.py`.**
  - `--full DIR` decodes the whole corpus through
    `sbz_dctx_decompress_frames`, compares with the version's own libzstd
    output and with 1.6.0, and prints a table of version x mode with
    pass / expected-reject / FAIL counts.
  - The default mode runs on a committed subset, `tests/data/compat/`, at
    most 5 MB. It holds one file per version for levels 3 and 19, the
    no-FCS and window-30 cases (expected reject until T09/T10), and the
    bitshuffle 0.4.0 chunks. Add it to `pixi run test`.
- **Expectations today**, to be updated by T09/T10:
  - no-FCS frames: `reject:SBZ_ERR_CORRUPT`;
  - `--long=30/31`: `reject:SBZ_ERR_CORRUPT`;
  - dictionary frames: `reject:SBZ_ERR_CORRUPT`.

  Anything else that does not decode is a FAIL: record its reproducer and
  report it. Do not fix the decoder in this task.
- **Rules.** Do not change `src/`, `kernels/` or `third_party/`. Never
  read or write outside the checkout and `$SLABITZ_WS/tmp/compat/`.

## Files that may change

`tools/gen_compat_corpus.py`, `tests/run_compat.py`,
`tests/data/compat/*` (5 MB at most), `pixi.toml` (test list),
`tests/test_suite.py`, `docs/dev/reports/T07.md`.

## Acceptance

1. `python3 tools/gen_compat_corpus.py --list` prints the 9 versions and
   the modes. `--build` builds all of them, and the report records the
   time taken and any version that failed to build, with the reason.
2. `timeout 3600 python3 tests/run_compat.py --full
   $SLABITZ_WS/tmp/compat/corpus` prints the version x mode table:
   - every `decode` cell passes;
   - every expected-reject cell gives the expected code;
   - the total compared bytes are nonzero and printed.
   Any FAIL is listed in the report with a reproducer under
   `tests/data/compat_bugs/`.
3. `du -sh tests/data/compat` is at most 5 MB, and `python3
   tests/run_compat.py` (subset) passes inside `pixi run test`.
4. The report quotes the zstd 1.6.0 CHANGELOG items relevant to the
   format, with a link to the upstream file, and lists the spec changes
   between 1.4.7 and 1.6.0 (`doc/zstd_compression_format.md` git log).

## Reviewer checks

- Pick 3 corpus files from different versions at random. Decode them with
  that version's `zstd -d` and with slabitz by hand, and compare the
  hashes.
- **Tamper with the manifest.** Change the expectation of one no-FCS file
  to `decode`: `run_compat.py` must FAIL on it.
