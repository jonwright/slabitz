# Plan: remaining pac ideas for the zstd decode path

Status (2026-11-21): implemented on branch `pac-zstd` (vectorized `MatchCopy` +
12-case detector matrix + real-file zstd coverage in `tests/run_bshuf_chain.py`).
The pac encoder is a separate project (`pac-zstd`) and stays out of scope here;
this is decode-side only and keeps `slabitz` a generic zstd/LZ4 GPU decoder.

## What was already merged

- `ExecuteSequences`: offset-1 matches are fills (one load, then stores only),
  added for pac-written bitshuffle-zstd frames (`e870151`, via
  `tools/transform_ds_for_slang.py`).
- Bitshuffle-LZ4: offset-1 fills and the overlapping plane-repeat copy with an
  incremental index instead of a modulo per byte (`3a94af2`,
  `kernels/lib/lz4_decode.slangh`).

`pac` (plane_aware_compressors) emits, in bit-plane data, mostly two kinds of
match: offset 1 (a run of equal bytes, in practice an empty plane) and offset
P (the plane below, P = block elements / 8, equal to `blockBytes / (8 * e)`).

## Investigation finding: why `seq.offs == P` in the swizzle loop does not apply

1. In DS `zstdgpu_shaders.h` (`zstdgpu_MatchCopy`), the `% seq.offs` swizzle loop
   is only entered when `seq.offs < min(WaveGetLaneCount(), seq.mlen)` (i.e.
   `seq.offs < 32`). Any offset `>= 32` already takes the direct modulo-free path.
2. For bitshuffle blocks, plane length `P >= 256` bytes (15,872 B for 248 kB
   `u16`; 7,936 B for 248 kB `u32`), and `pac_parse.inl` parses at most one
   plane per piece (`P_ov = 0` overlapping `off == P` matches across all 12
   cases). So an incremental-index swizzle change guarded on `seq.offs == P`
   is dead code.
3. Testing the **3 x 2 x 2 = 12 case matrix** (`silicon`, `thaumatin`, `ceo2` x
   `u16`/`u32` x `mask=0`/`mask=uint_max-1` in `bench/pac_matrix.py`) showed
   why non-zero masks slowed decoding by up to 2.1x wall / 5.0x in
   `ExecuteSequences`: with `mask=max-1`, **47.9% to 94.6%** of all frame bytes
   are decoded by **non-overlapping `off == P` matches**, which went through the
   original 1-byte-per-lane-per-iteration `dstData[dstI - seq.offs]` loop in
   `zstdgpu_MatchCopy` (latency-bound by byte loads/stores), whereas `off == 1`
   had already been changed to store-only byte fills.

## Implemented change: generic vectorized `zstdgpu_MatchCopy`

Instead of passing a `planeSize` scalar or adding pac-specific branches:

- Added `inoutUnCompressedFramesData32` (`uint32_t`) and
  `inoutUnCompressedFramesData16` (`uint4`) alias views to
  `zstdgpu_ExecuteSequences_SRT`, `kernels/ExecuteSequences.slang`, and
  `src/sbz_chain.c`, routed into `zstdgpu_MatchCopy` through
  `tools/transform_ds_for_slang.py`.
- In `zstdgpu_MatchCopy`:
  - `seq.offs == 1u` with `copyLen >= 32`: lane 0 reads the repeat byte once,
    then writes unaligned head/tail bytes and 16-byte aligned `uint4` stores across
    the interior (replaces byte-at-a-time stores).
  - Non-overlapping aligned matches (`seq.offs >= copyLen`): 16-byte `uint4`
    loads/stores when `copyLen >= 64` and `(seq.offs & 15) == 0`, or 4-byte
    `uint32` loads/stores when `copyLen >= 32` and `(seq.offs & 3) == 0`, with
    byte head/tail handling (disjoint source and destination ranges).
  - Every other match keeps the existing DS byte loop or uniform-trip-count
    `< 32` swizzle loop.
- No extra kernel parameters from the caller, no change to stock-zstd semantics,
  and works for all zstd inputs (`silicon`/`thaumatin`/`ceo2`, `pac` and `stock`).

## Measurements (GTX 1050 Ti sm_61, 248 kB blocks, 4 frames/call)

Verified all 24 runs (12 cases x `{pac, stock}`) byte-exact (648,854,592 bytes
compared in `bench/pac_matrix.py`). `SLABITZ_PROFILE=1` on `silicon_u32` (4
frames/call):

- `silicon_u32_mask0` (`off=1` 99.7%): `ExecuteSequences` **1313.5 -> 841.6 us** (1.56x)
- `silicon_u32_maskmax-1` (`off=P` 94.6%): `ExecuteSequences` **6623.5 -> 1776.0 us** (**3.73x**)

Wall time per frame (ms/fr and GB/s) before -> after:

| Case | Codec | Ratio | Before (ms/fr, GB/s) | After (ms/fr, GB/s) |
|---|---|---:|---:|---:|
| `silicon_u16_mask0` | pac | 274.87 | 0.62 (14.48) | 0.64 (14.05) |
| `silicon_u16_mask0` | stock | 207.55 | 0.82 (10.85) | 0.85 (10.50) |
| `silicon_u16_maskmax-1` | pac | 171.39 | 1.36 (6.57) | **0.83 (10.76)** |
| `silicon_u16_maskmax-1` | stock | 159.64 | 1.43 (6.23) | 1.38 (6.47) |
| `silicon_u32_mask0` | pac | 449.95 | 0.90 (19.81) | **0.85 (21.08)** |
| `silicon_u32_mask0` | stock | 389.57 | 1.36 (13.18) | 1.40 (12.80) |
| `silicon_u32_maskmax-1` | pac | 295.91 | 2.32 (7.71) | **1.23 (14.49)** |
| `silicon_u32_maskmax-1` | stock | 238.47 | 2.47 (7.23) | 2.46 (7.27) |
| `thaumatin_u16_mask0` | pac | 74.35 | 3.09 (11.71) | **2.92 (12.40)** |
| `thaumatin_u16_mask0` | stock | 59.64 | 3.56 (10.18) | 3.53 (10.24) |
| `thaumatin_u16_maskmax-1` | pac | 66.41 | 5.37 (6.74) | **3.59 (10.08)** |
| `thaumatin_u16_maskmax-1` | stock | 49.08 | 5.55 (6.53) | 5.49 (6.59) |
| `thaumatin_u32_mask0` | pac | 139.59 | 4.42 (16.38) | **4.30 (16.84)** |
| `thaumatin_u32_mask0` | stock | 116.59 | 5.30 (13.65) | 5.28 (13.70) |
| `thaumatin_u32_maskmax-1` | pac | 124.86 | 9.28 (7.80) | **5.80 (12.49)** |
| `thaumatin_u32_maskmax-1` | stock | 84.24 | 9.46 (7.65) | 9.46 (7.65) |
| `ceo2_u16_mask0` | pac | 2.57 | 2.34 (3.82) | **2.28 (3.92)** |
| `ceo2_u16_mask0` | stock | 2.44 | 3.37 (2.66) | 3.32 (2.69) |
| `ceo2_u16_maskmax-1` | pac | 2.55 | 2.76 (3.24) | **2.51 (3.56)** |
| `ceo2_u16_maskmax-1` | stock | 2.43 | 3.53 (2.53) | 3.51 (2.55) |
| `ceo2_u32_mask0` | pac | 5.12 | 2.47 (7.25) | **2.41 (7.42)** |
| `ceo2_u32_mask0` | stock | 4.89 | 3.83 (4.66) | 3.84 (4.65) |
| `ceo2_u32_maskmax-1` | pac | 5.08 | 3.59 (4.99) | **2.68 (6.67)** |
| `ceo2_u32_maskmax-1` | stock | 4.85 | 4.51 (3.97) | 4.54 (3.94) |

## Verification

- `third_party/slang/bin/slangc kernels/ExecuteSequences.slang -target spirv` and `-target metal` both compile with exit code 0.
- `tests/run_bshuf_chain.py` now also tests `$SLABITZ_EXAMPLES/silicon_u16_zstd2_256k.h5` (143,072,512 B compared) and `ceo2_u32_zstd2_256k.h5` (286,145,024 B compared) when present (skips with a notice otherwise).
- Passed after final kernel/lib edits (`CUDA_ARCH=sm_61`): `run_kernel_id.py`, `run_host_walk.py`, `run_chain_stage01.py`, `run_chain_stage2.py`, `run_chain_frames.py`, `run_bshuf_chain.py`, `run_zstd_async.py`.

## Out of scope

- The pac encoder and its sequence producer (`pac-zstd`).
- Changes to the LZ4 path (already has both cases).
- Multi-warp / cross-block parallelism inside `ExecuteSequences` (deferred to P6; one workgroup per frame remains the remaining floor on `maskmax-1` and `CeO2`).
