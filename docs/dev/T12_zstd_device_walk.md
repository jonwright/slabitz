# T12: P4 step 2c for zstd: chunk walk on the device

Status: todo
Executor: Claude (subtle C runtime and kernel work; not for the worker)
Depends on: T03 (gate). Ideally after T09/T10, so the bounds exist; it may
start in parallel if it rebases onto them before merging.
Branch: `t12-zstd-device`

## Goal

Add `sbz_dctx_bshuf_decompress_device`, the zstd twin of
`sbz_dctx_bslz4_decompress_device`. Bitshuffle-zstd chunks that live only
in device memory are decoded with no host pass over the compressed bytes.
This is the "full GPU" option the user asked for (2026-11-20). It is the
basis of the device-input API that the friendly and bslz4_to_sparse
integration needs (T13, T15).

## Background

- **The LZ4 device path** (1ecfae2, `src/sbz_chain.h:136-161`):
  - the caller states a geometry `sbz_bshuf_geom {elemSize, chunkBytes,
    blockBytes}`;
  - the `BshufWalkChunks` kernel walks each chunk's block records on the
    device and checks the header against the geometry;
  - `sbz_dctx_wait` checks the per-chunk status;
  - on a corrupt chunk or a geometry mismatch, the wait copies the input
    back and redoes the call on the host (the slow path).

  The walk is latency-bound: 412 / 1255 us per 16-chunk call on silicon /
  CeO2, overlapped by using two contexts.
- **The design for zstd** (`docs/archive/plan_p4_api.md`, step 2c):
  - the frame refs come from the same chunk walk;
  - DS's `ParseFrames` already counts and collects frames and blocks on
    the device;
  - what is missing is sizing without the host. Size from bounds: blocks
    at most compressed bytes / 3; frames at most chunks x blocks per chunk;
    output equal to the caller's stated frame size;
  - keep the capacity check at the wait.
- **Block-type counts.** Today the zstd async path takes the per-type
  block counts from the host walk (P4 2b, 806b260). On the device path
  they must be bounded instead, and the frame-parsing kernels need
  capacities and guards, so they do not write past bounded buffers.
- **The async contract** (P4 step 2):
  - no readback in the no-error path;
  - `SLABITZ_PROFILE=1` and `SLABITZ_CHECK_SYNC=1` keep working;
  - errors are rare, so the slow path may replay synchronously.

## Read first

`AGENTS.md`; `src/sbz_chain.c` (`bshuf_decode_lz4` device path, the zstd
async path, `host_prepare`, wait/replay); `kernels/BshufWalkChunks.slang`;
the DS ParseFrames / ParseCompressedBlocks kernels;
`tests/run_bslz4_device.py`; `tests/run_zstd_async.py`;
`docs/archive/plan_p4_api.md` (steps 2 and 2c);
`docs/archive/p4_readback_audit.md`.

## Settled design

- **API.** `sbz_dctx_bshuf_decompress_device(dc, dev_src, srcSize, chunks,
  nchunks, geom)` takes the same arguments as the LZ4 device call, and is
  queued the same way. `sbz_dctx_wait` checks the status and capacities.
- **Walk.** Extend `BshufWalkChunks`, or add a sibling kernel, so that for
  zstd it emits the frame refs: each block record is one zstd frame. Then
  DS stage 0 (ParseFrames, ParseCompressedBlocks) runs on those refs with
  bounded buffers.
- **Bounds.** Size all stage 0/1 buffers from upper bounds that depend only
  on `srcSize`, `nchunks` and `geom`. Add capacity guards and status bits
  to the kernels that append to those buffers. An overflow sets the status,
  and the wait replays the call exactly on the host.
- **Recurring calls.** For a stream of calls with the same geometry, reuse
  the learned capacities as P4 2b does.

## Files that may change

`src/sbz_chain.c`, `src/sbz_chain.h`, `kernels/BshufWalkChunks.slang`
(or a new entry), `tools/transform_ds_for_slang.py`, other
`kernels/*.slang` that need guards, `src/sbz_kernels_gen.h` (regenerated),
`slabitz/loader.py`, `tests/run_bshuf_device.py` (new, modelled on
`run_bslz4_device.py`), `pixi.toml`, `bench/p4_overlap.py` (a `device`
mode for zstd), `docs/dev/reports/T12.md`.

## Acceptance

1. `python3 tests/run_bshuf_device.py` passes for:
   - both pac zstd fixtures (T02), the two real zstd files, and the
     hdf5plugin u8-u64 cases;
   - byte-exact output, with nonzero compared bytes;
   - a corrupt chunk giving CORRUPT through the slow path;
   - a chunk with a different geometry decoded correctly through the slow
     path.
2. **No host access to the input.** The test passes the input only as a
   device allocation. The host copy is freed or overwritten before the
   call, then restored only for the slow-path case.
3. **Overflow path.** A forced overflow (set the learned capacities to 1
   through a debug environment variable) replays and still gives correct
   output. `sbz_dctx_zstd_stats` shows the replay.
4. `bench/p4_overlap.py ... device` on the L40S (Slurm) reports the
   calls/s for silicon and CeO2 zstd with device input. The report
   compares it with the host-input async path.
5. `bench/gate.py --compare`: no regression in the existing cells.
6. `sbz test` and `sbz test-kernels` pass. The kernel sets are
   rebuilt, and the T05 suite runs on the L40S with `build/kernels`.

## Reviewer checks

- Confirm by reading that the no-error path has no `read_counters` or
  sync between launch and wait.
- Remove one capacity guard (temporary) and run the forced-overflow case:
  it must fail, or report a canary or sanitizer hit.
