# T08: pac-zstd: lower the compat floor to bitshuffle 0.4.0 (other repo)

Status: todo
Executor: worker (DeepSeek) in a scratch clone of pac-zstd, reviewed by Claude
Depends on: - (independent; T19 depends on its result)
Repository: `../pac-zstd` (GitHub jonwright/pac-zstd; `../plane_aware_compressors`
is another clone of it). Branch there: `compat-floor-bitshuffle040`.

## Goal

pac-zstd's output must decode with the decoders in the first bitshuffle
release that had zstd: bitshuffle 0.4.0, which bundles zstd
`18d02cbf` (`v1.4.7-403`) and LZ4 1.9.x. Its tests must prove this. Today
its documented floor is zstd 1.5.2.

## Background (pac-zstd branch `gemini-minimal-p9fix`, HEAD 162af8b)

- **Current floor.** `tests/check_compat.sh:29-36` runs zstd 1.5.2 by
  default (CI), and 1.5.2, 1.5.4, 1.5.5, 1.5.6, 1.5.7 and dev with
  `COMPAT_MATRIX=1`. It also runs bitshuffle 0.4.0 + zstd 1.5.2 + lz4
  1.9.3, kcb with lz4 1.10.0, and the Go decoders. RESULTS.md:102-103
  reports 530/0 (zstd) and 262/0 (LZ4).
- **Decoder errata** documented in `docs/build.md:47-61` and
  `PROOFREAD.md:95-120`:
  - libzstd 1.5.2 and older reject a compressed block of exactly 131072
    bytes, and a block with 0 literals and 0 sequences;
  - libzstd 1.5.5 and older mishandle some magicless frames;
  - the CLI at 1.4.3 and older rejects a frame whose first block is a
    131072-byte RLE block.
- **Open issue** (PROOFREAD.md:111-120). The grouping in `src/pac.c:596`
  uses `>`, so a compressed block of exactly 131072 bytes can occur, and
  1.5.2 rejects it. No fix was found.
- **Not yet checked:** whether `18d02cbf` has other errata. It predates
  1.5.0, so it may have more than 1.5.2.

## Read first (in pac-zstd)

`AGENTS.md`, `docs/build.md`, `PROOFREAD.md`, `tests/check_compat.sh`,
`tests/gen_compat_corpus.c`, `tests/verify_compat.c`, `tests/check_rfc.c`,
`src/pac.c` (block grouping around line 567-632, the splice at 760-808).

## Settled design

- **Add the floor versions to `check_compat.sh`.** Add zstd `v1.4.7` and
  `18d02cbf` to the full matrix, and make `18d02cbf` the default CI floor
  in place of 1.5.2. Add LZ4 at the version bundled in bitshuffle 0.4.0 to
  the floor. Also check "bitshuffle 0.4.0 as released": build it from its
  0.4.0 commit with its own submodules, rather than 0.4.0 + zstd 1.5.2.
- **Fix the 131072-byte compressed block.** Never emit a compressed block
  of exactly 131072 bytes; change the grouping test or split. Add a test
  that builds data which would have produced one, and checks that no block
  of that size appears (`check_rfc` already flags it).
- **Fix or avoid every other failure** the new floor shows. Each fix needs
  a test and a line in `docs/build.md` naming the decoder version and the
  feature avoided. If avoiding a failure costs compression ratio, report
  the cost on pac's bench matrix (`bench/sweep.py`).
- **Documentation.** Update `AGENTS.md`, `README.md` and `docs/build.md`
  so the stated floor is "bitshuffle 0.4.0 (zstd 18d02cbf / 1.4.7+, LZ4
  1.9.x)".

## Acceptance (run in the pac-zstd clone)

1. `make check-compat` (the CI floor) passes and includes zstd `18d02cbf`
   and bitshuffle 0.4.0 as released.
2. `COMPAT_MATRIX=1 make check-compat-matrix` reports 0 failures over every
   version from `v1.4.7` to `dev`, with the totals printed (expect more
   than 530 zstd cases).
3. The new test for the 131072-byte block passes. With the fix reverted,
   it fails.
4. `make check` (pac's own tests) passes, and the bench ratio change is
   reported (any change, even 0).

## Reviewer checks

- Rerun acceptance 1 in a clean clone.
- Revert the 131072 fix and confirm that acceptance 3 fails.
- `grep -rn "1.5.2" docs/ AGENTS.md README.md` shows no remaining claim
  that 1.5.2 is the floor.

## Notes for slabitz

Record the result in slabitz's `docs/history.md` when merged. T19's zstd
encoder must avoid every feature this task lists as unsafe for the floor.
