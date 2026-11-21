# Plan: remaining pac ideas for the zstd decode path

Status (2026-11-21): work item, not started. The pac encoder is a separate
project and is out of scope here; this is decode-side only.

## What is already merged

- `ExecuteSequences`: offset-1 matches are fills (one load, then stores only),
  added for pac-written bitshuffle-zstd frames (`e870151`, via
  `tools/transform_ds_for_slang.py`).
- Bitshuffle-LZ4: offset-1 fills and the overlapping plane-repeat copy with an
  incremental index instead of a modulo per byte (`3a94af2`,
  `kernels/lib/lz4_decode.slangh`).

`pac` (plane_aware_compressors) emits, in bit-plane data, mostly two kinds of
match: offset 1 (a run of equal bytes, in practice an empty plane) and offset
P (the plane below, P = block bytes / 8). The first is merged for both
decoders; the second only for LZ4.

## Remaining decoder idea

Add the plane-repeat case to the zstd `MatchCopy` swizzle loop:

- the bitshuffle-zstd path knows the block geometry, so it can pass an optional
  plane size (P) to `ExecuteSequences`; the plain zstd calls pass 0 and keep
  today's behaviour;
- when `seq.offs == P` (and `P < copyLen`, the overlapping case), copy with an
  incremental index like the LZ4 path, without the per-byte `% seq.offs`;
- guard the new path so stock-zstd timing does not change. A previous attempt
  to use an incremental index for every overlapping match made stock zstd 14%
  slower and was reverted; the plane-size guard is what changes that.

The kernel change goes through `tools/transform_ds_for_slang.py` (never edit
the generated `.cu`), and the extra argument is a scalar in the entry
reflection table.

## Tests and measurement

- Byte-exact vs hdf5plugin on `silicon_u16_zstd2_256k.h5` and
  `ceo2_u32_zstd2_256k.h5` from `SLABITZ_EXAMPLES`, 16 frames per call.
- Same on stock-zstd chunks (must not regress), with the `SLABITZ_PROFILE=1`
  per-kernel table and the wall/device split.
- Add a committed test or bench so the case is exercised when the example
  data is present and skipped with a notice when it is not.
- Report pac and stock separately; zero compared bytes is a failure.

## Out of scope

- The pac encoder and its sequence producer.
- Changes to the LZ4 path (already has both cases).
- A general "no modulo" rewrite of every overlapping zstd match (the measured
  stock regression above).

This is subtle C/kernel work; do not hand it to the DeepSeek worker.
