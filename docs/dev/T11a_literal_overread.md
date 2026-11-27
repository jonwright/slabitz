# T11a: ExecuteSequences reads past the end of the literals

Status: review (`docs/dev/reports/T11a.md`)
Executor: Claude (host allocation; kernel only if padding is not enough)
Depends on: T11 (the finding and `tests/run_sanitizer.sh`)
Branch: `t11a-literal-overread`

## Goal

compute-sanitizer memcheck reports out-of-bounds reads on a valid frame. Make
`tests/run_sanitizer.sh` report 0 memcheck errors on the valid inputs, with no
slowdown.

## Background

- **Finding** (`docs/dev/reports/T11.md`, cause 3 and review section).
  `tests/data/compat/v1.5.7--level3_text.zst` (2216 bytes, valid) under
  `compute-sanitizer --tool memcheck`:
  - 76 one-byte reads, 1 to about 30 bytes past a 440-byte allocation;
  - the allocation is `DecompressedLiterals` (HUF_Streams_DecodedBytes 438,
    rounded to 4);
  - the kernel is ExecuteSequences (`SLABITZ_DEBUG_SYNC=1` stops at
    `chk:Execute`). Under memcheck the launch then fails (719); without it the
    decode is byte-exact.
- **Cause.** The fast paths added by the transform (`EXEC_VEC_PATCHES` in
  `tools/transform_ds_for_slang.py`):
  - `zstdgpu_ExecuteSequence_FusedCopy` loads `litBuf[litOfs + laneId]` for
    every lane below `llen + mlen`;
  - the coalesced branch of `zstdgpu_ExecuteSequencePair_FusedCopy` does the
    same for every lane below the combined length;

  and both then select the literal or the match value. Lanes copying match
  bytes therefore read literal bytes that do not exist: up to 31 bytes past the
  block's literals (63 in the pair path). At the last block of a call, that is
  past the end of the buffer.
- **Raw literals** come from `CompressedData` (`litBuf` =
  `inCompressedData`), padded only to 4 bytes (`stage0`), so a raw-literals
  block at the end of the input can read past that buffer too.
- The loads are unconditional on purpose ("keep control flow wave-uniform"). T04a
  measured that this loop pays for every extra instruction per match, so a
  per-lane guard in the kernel is the last resort.

## Settled design

- Pad the allocations instead of changing the kernel:
  - `DecompressedLiterals`: allocate `cntLit + 64` bytes;
  - `CompressedData`: 64 bytes (16 words) of slack after the padded input,
    zero-filled once per allocation;
  - the bitshuffle and LZ4 paths that bind other `litBuf` sources: check them
    the same way.

  Record the padding next to `buf_size` with the reason.
- No change to the kernels unless padding leaves memcheck errors. If so, report
  which load and measure the guard with the gate before choosing it.

## Files that may change

`src/sbz_chain.c`, `tests/run_sanitizer.sh` (expectations),
`docs/dev/reports/T11a.md`, `AGENTS.md`.

## Acceptance

1. `bash tests/run_sanitizer.sh` via `gpu-run`: memcheck 0 errors on the zstd
   text frame, the bitshuffle-zstd and both bitshuffle-LZ4 cases; racecheck 0.
2. `sbz test` passes.
3. The gate passes on the baseline node.

## Reviewer checks

- Remove the padding: memcheck reports the reads again.
