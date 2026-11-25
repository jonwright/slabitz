# T10: Decoder: frames without a content size

Status: todo
Executor: Claude (C runtime, device sizing, async path)
Depends on: T09
Branch: `t10-no-fcs`

## Goal

Decode zstd frames that carry no Frame_Content_Size, which is what zstd's
streaming API, `zstd` reading stdin, and `--no-content-size` write. This
closes the main gap to "the decoder handles all zstd cases". After T09
these frames are rejected with UNSUPPORTED "no content size".

## Background

- **Output sizing today.** Output size and frame bases come only from the
  FCS sum (`src/sbz_chain.c:918-942`). A missing FCS gives size 0, and the
  GPU parser marks it `MissingContentSize` (ds:394, 422).
- **Bitshuffle chunks.** The decoded size of every frame is known from
  the chunk header: block size x element size, and the tail. So a no-FCS
  frame inside a filter-32008 chunk needs only "expected size = block
  bytes", not real discovery.
- **Plain zstd.** The size is known only after decoding.
  - A host upper bound per frame: the sum of the raw block sizes, plus the
    RLE regenerated sizes, plus Block_Maximum_Size for each compressed
    block. The host walk already counts blocks by type.
  - A compressed block's decoded size is its literal count plus the sum of
    its match lengths. Both are known on the device after the sequence
    decode, before ExecuteSequences.
- **The async contract** (P4 step 2b, `docs/archive/plan_p4_api.md`): no
  readback in the no-error path. Capacities are learned from exact calls;
  the wait checks once and replays exactly on overflow.

## Read first

`AGENTS.md`; `src/sbz_chain.c` (`host_prepare`, output sizing, frame refs,
`ComputeDestBlockOffsets` inputs, the wait/replay path);
`kernels/ComputeDestBlockOffsets.slang`; the DS code that fills
`inBlockSizePrefix` and `UnCompressedFramesRefs`; `src/host_walk.c`;
`docs/archive/p4_readback_audit.md` (sections 2-4).

## Settled design

- **Step 1 (report only, the first commit).** Trace where DS gets each
  block's decoded size and each frame's base offset. Confirm that for
  compressed blocks the decoded size is computed on the device and not
  read from the frame header. Write the trace into the report. If it
  contradicts this design, write QUESTIONS.txt and stop.
- **Bitshuffle path.**
  - For no-FCS frames inside chunks, the host walk sets the expected
    frame size to the block's decoded bytes.
  - The GPU uses that size as the frame's bound, so the T09 clamp
    applies, and it checks at the end that the decoded size equals the
    expectation (per-frame status).
  - No new readback.
- **Plain zstd path** (`sbz_dctx_decompress`,
  `sbz_dctx_decompress_frames`):
  - Frames with an FCS keep today's behaviour.
  - If any frame lacks an FCS, the call works as follows:
    1. Frame bases come from a device prefix sum over per-frame decoded
       sizes. Each frame's size is the sum of its blocks' decoded sizes,
       computed after the sequence decode.
    2. The output buffer is sized from the host upper bound.
    3. The real per-frame sizes and offsets are written to a device array,
       which the caller reads with a new
       `sbz_dctx_frame_sizes(dc, uint64_t *sizes, size_t n)`.
  - The frames API documents that, with no-FCS frames present, the output
    is still packed back to back, and the offsets must be read with
    `sbz_dctx_frame_sizes`.
  - Exact mode reads the sizes back as part of its existing readbacks.
- **Bounds.** A frame that decodes past its host bound is CORRUPT, through
  the same per-frame overflow status as T09.
- **Tests.** Update the T06/T07 expectations for no-FCS cells to `decode`.
  Add to `tests/run_features.py` a mixed batch (FCS and no-FCS frames
  interleaved, plus empty frames without an FCS), and a no-FCS frame
  inside a bitshuffle chunk, built by recompressing a chunk's frames with
  `write_content_size=False`.

## Files that may change

`src/sbz_chain.c`, `src/sbz_chain.h`, `src/host_walk.c`,
`tools/transform_ds_for_slang.py`, the affected `kernels/*.slang` (with a
regenerated `src/sbz_kernels_gen.h`), `tests/run_features.py`,
`tests/run_compat.py`, `tests/data/compat/` (expectations), `slabitz/loader.py`
(the new function), `docs/dev/reports/T10.md`.

## Acceptance

1. `python3 tests/run_compat.py --full ...`: every no-FCS cell (streaming
   stdin and `--no-content-size`, every version) decodes byte-exact.
2. `python3 tests/run_features.py`:
   - the mixed-batch and no-FCS-chunk entries decode byte-exact, sync and
     async;
   - `sbz_dctx_frame_sizes` returns the true sizes;
   - compared bytes are nonzero.
3. A no-FCS frame whose data decodes larger than the expected chunk block
   size gives CORRUPT, and the neighbour frames are intact (extend
   `tests/run_frame_bounds.py`).
4. `bench/gate.py --compare`: no regression. The no-FCS path may be
   slower, and the report gives its cost on one silicon case.
5. `pixi run test` and `pixi run test-kernels` pass. The kernel sets are
   rebuilt as in T09.

## Reviewer checks

- **Break the size path.** Force the device prefix sum to use the host
  upper bound instead of the decoded sizes (temporary change): the
  mixed-batch test must fail on the frame offsets.
- **Readback check.** Read the async path diff and confirm that it adds no
  host readback when no frame lacks an FCS.
