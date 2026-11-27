# Plan: harden the host frame walk against corrupt input (done)

Outcome (2026-11-20, worker-drafted 46f7efd, review 2549539): as planned.
The review made the fuzz harness give each mutant an exact-size buffer and
exact-size output arrays (one shared max-size buffer had hidden overreads
from ASan), replaced its two seeds (one 443 kB) by ten small structured
ones, and kept the walk out of `libslabitz.so`'s exports. Removing one
bounds check makes the fuzz run fail at once (heap-buffer-overflow), so it
does catch overreads. 2M mutants run clean in 2.5 s. The remaining
`host_walk.so` mentions in AGENTS.md and here were the worker's one
question; they are history now.

The host-walk hardening item in `AGENTS.md`. `src/host_walk.c` is a verbatim port of the
DirectStorage CPU collector walk. It trusts the input: a truncated or garbled
zstd frame header makes it read past the input buffer, loop on garbage, or
hand the GPU block offsets outside the data. After this job, any input gives
`SBZ_OK`, `SBZ_ERR_CORRUPT` or `SBZ_ERR_LIMIT` from `sbz_dctx_decompress*`, and
the walk never reads outside its buffer.

## Read first

1. `AGENTS.md` (rules, especially: tests that compare nothing must fail).
2. `src/host_walk.c`, `src/host_walk.h`.
3. `src/sbz_chain.c`: `host_prepare()` (the only caller in the library) and
   `sbz_dctx_counters()`.
4. `tests/run_host_walk.py`, and `tests/run_chain_stage01.py` around line
   170 (it loads `build/host_walk.so` as an independent reference).
5. `Makefile` (`build_host`, `host_walk.so`), `pixi.toml` (`test` task).
6. The zstd format: frame header, block header and skippable frames
   (RFC 8878 sections 3.1.1 and 3.1.2).

## Design (settled; do not redesign)

### Scope of the walk

- Keep only what the library uses: `sbz_CountFramesAndBlocks` and
  `sbz_CollectFrames` (and the static `sbz_ParseFrame` with its bit reader).
  Delete `sbz_CollectBlocks` and `sbz_CountCompressedLiteralsAndSequences` and
  the literal/sequence branch inside `sbz_ParseFrame` that only they need
  (the GPU does that work; nothing in `src/` calls them). A compressed block
  is then just skipped by its size.
- Both remaining functions return `int`: `SBZ_OK`, `SBZ_ERR_CORRUPT` or
  `SBZ_ERR_LIMIT` (from `src/sbz_backend.h`; include it in `host_walk.h`).
  `host_prepare()` returns that code through `fail()` with a message naming
  the frame index and byte offset. Remove the "verbatim DS port" wording
  from the comments that would then be false.

### Input contract

- `memoryBlock` holds `memoryBlockSizeInBytes` bytes (a multiple of 4),
  followed by 4 more readable zero bytes (the chain allocates padded + 4, so
  the 32-bit refill one word past the end is in bounds). `contentSizeInBytes`
  <= `memoryBlockSizeInBytes` is the real data; the rest is zero padding.
- Every field read and every skipped block must end at or before
  `contentSizeInBytes`. Check before reading or skipping, not after.

### Checks (each one -> SBZ_ERR_CORRUPT unless stated)

- Fewer than 4 bytes left where a frame should start; a magic number that is
  neither zstd (0xFD2FB528) nor skippable (0x184D2A50..0x184D2A5F).
- Skippable frame: the 4-byte size field or the skipped bytes run past the
  content.
- Frame header: the header (descriptor, window descriptor, dictionary id,
  content size) runs past the content; the reserved descriptor bit (bit 3)
  is set; a nonzero dictionary id (dictionaries are not supported: message
  "dictionary id N not supported").
- Block header (3 bytes) runs past the content; block type 3 (reserved);
  block size above 128 KiB (Block_Maximum_Size; also for RLE blocks, whose
  size is the decoded length); a raw or compressed block's bytes, or an RLE
  block's one byte, run past the content; a compressed block of size 0.
- The content ends before a block with the last-block flag.
- The content checksum (4 bytes when the descriptor flag is set) runs past
  the content.
- `SBZ_ERR_LIMIT`: any running count or byte total that the DS kernels keep
  in 32 bits would exceed 0xFFFFFFF0 (frames, blocks, raw/RLE block bytes,
  frame byte count, a frame's content size). Example: a 128 KB input of RLE
  blocks (4 bytes each, 128 KiB decoded each) declares over 4 GiB.
- `sbz_CollectFrames` never writes more than `frameCount` entries, and returns
  `SBZ_ERR_CORRUPT` if the second walk does not find exactly `frameCount`
  frames.

Not in scope (record in the report, do not attempt): the contents of
compressed blocks (literal and sequence sections) are checked by the GPU
kernels or not at all; a frame content size that disagrees with its blocks.

### Tests

- Drop `build/host_walk.so`: remove the `build_host`/`host` targets and the
  `host_walk.so` rule from the `Makefile` and its mention in comments. Tests
  go through `libslabitz.so`.
- `tests/run_host_walk.py`, rewritten to use `libslabitz.so`:
  - Valid inputs (keep its current corpus: single and concatenated frames,
    raw/RLE/compressed blocks, a skippable frame, checksum frames): decode
    with `sbz_dctx_decompress` and compare byte-exact with `zstandard`; check
    frame and block counts (`Frames`, `Blocks_RAW`, `Blocks_RLE`,
    `Blocks_CMP` from `sbz_dctx_counters`) against counts computed in Python
    from the frame headers, and the GPU's literal/sequence totals (the
    matching `sbz_dctx_counters` fields, e.g. `Seq_Streams_DecodedItems`)
    against the edu oracle, which the current test uses for the deleted
    host-side totals. Print how many values were compared;
    zero compared is a failure.
  - Corrupt inputs, one case per check above (named, so a failure says
    which): each must return `SBZ_ERR_CORRUPT` (or `SBZ_ERR_LIMIT` for the RLE
    case), with no hang (run each under the test's timeout).
- `tests/run_chain_stage01.py`: replace its use of `host_walk.so` (literal
  and sequence totals) with the edu oracle (`tools/edu_oracle.py`), keeping
  the number of compared values the same or higher.
- Fuzz under AddressSanitizer: `tests/fuzz_host_walk.c`, built by a new
  `Makefile` target `build/fuzz_host_walk` straight from `src/host_walk.c`
  with `$(CC) -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all`
  (both work in the pixi env). It reads a corpus file of valid inputs that
  `run_host_walk.py` writes into `build/`, applies a fixed-seed sequence of
  mutations (byte flips, truncation at every length for small inputs,
  rewritten header fields, two inputs spliced), copies each mutant into an
  exact-size `malloc(padded + 4)` buffer, calls both walk functions, and
  requires a return in {OK, CORRUPT, LIMIT}. `run_host_walk.py` builds and
  runs it (at least 200000 mutants, under a timeout) and fails on any
  sanitizer report or nonzero exit. It prints the count of mutants per
  return code; all mutants returning OK is a failure (the mutations did not
  reach the checks).
- `pixi.toml`'s `test` task and `tools/build_all.py` (if it builds
  `host_walk.so`) updated accordingly.

## Build and verify (in the scratch clone)

- Full build and all regression tests: `bash ../build_test.sh` (about 8
  minutes). All must pass.
- Single test: `. ../activate.sh && timeout 900 python3 tests/run_host_walk.py`.
- `git grep -n host_walk.so` must find nothing outside `docs/history.md`.

## Rules

- ASCII only. C89-compatible style as in the existing `src/` files.
- Change only `src/host_walk.c`, `src/host_walk.h`, `src/sbz_chain.c`
  (`host_prepare` and its error messages), `Makefile`, `pixi.toml` (the
  `test` task only), `tests/run_host_walk.py`, `tests/run_chain_stage01.py`,
  the new `tests/fuzz_host_walk.c`, and `tools/build_all.py` only if it
  refers to `host_walk.so`. Do not edit `AGENTS.md`, `README.md` or `docs/`.
- No change to any kernel or to decoding of valid input: every existing test
  keeps passing with its compared counts unchanged.

## Deliverables

1. One commit in the scratch clone:
   `host_walk: bounds-check the frame walk, drop host_walk.so (drafted by
   <agent>)`, with the test results (cases, compared counts, fuzz counts per
   return code) in the message and a `Co-Authored-By:` trailer naming you.
2. `REPORT_host_walk.md` in the clone root (not committed): each check and
   the test case that covers it; the fuzz results; every decision you made
   and why; what is not covered.

## Acceptance (checked by the architect)

- Full suite passes; valid-input results unchanged.
- Every corrupt case returns its code; the fuzz run is clean under ASan and
  UBSan and reaches every return code.
- `host_walk.so` and `make build_host` are gone; nothing else loads
  `host_walk.c` except the library and the fuzz binary.
