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

## Implemented changes

1. **Generic vectorized `zstdgpu_MatchCopy` & `ExecuteSequences` tuning**
   (`tools/transform_ds_for_slang.py`, `kernels/ExecuteSequences.slang`,
   `src/sbz_chain.c`, regenerated `src/sbz_kernels_gen.h`):
   - Bound `inoutUnCompressedFramesData32` (`uint32_t`) and
     `inoutUnCompressedFramesData16` (`uint4`) alias views to
     `B_UnCompressedFramesData` inside `ExecuteSequences`.
   - In `zstdgpu_MatchCopy`:
     - `seq.offs == 1u` (`copyLen >= 32`): broadcast single-byte L1 read (no
       wave intrinsic), then unaligned head/tail bytes and 16-byte aligned
       `uint4` stores across the interior.
     - Aligned matches (`seq.offs >= 64` and `(seq.offs & 15) == 0` for
       `uint4` when `copyLen >= 64`, or `seq.offs >= 32` and `(seq.offs & 3) == 0`
       for `uint32` when `copyLen >= 32`), chunked in `seq.offs` steps when
       `copyLen > seq.offs` so overlapping `off == P` matches stay race-free
       and vectorized, with byte head/tail handling.
   - In `zstdgpu_ExecuteSequence_FusedCopy`: added fused short `seq.offs == 1`
     fill (`total <= 32`) when `llen + mlen` fits in one wave, avoiding a
     `dstData` read-after-write stall whenever `llen > 0`.
   - In `zstdgpu_MemCpy_DstSrc`: vectorized literal copies (`copyLen >= 32`)
     with 32-bit coalesced `dstData32` word stores.
2. **Huffman literals tuning** (`tools/transform_ds_for_slang.py`,
   `kernels/InitHuffmanTableAndDecompressLiterals.slang`):
   - Overlapped `GS_PreInit` (322 dwords, dead before table expansion) with
     `GS_HuffmanTable` (1024 dwords), shrinking workgroup LDS from 1614 to 1292
     dwords (6456 -> 5168 bytes).
   - Packed symbol bit length `bits` into `GS_CodeAndSymbol` bits 16..23 so
     table expansion reads `[0, r1)` in O(1) and handles `[r1, codeTableSize)`
     with 1 binary search per state pair instead of 4.

## Measurements (GTX 1050 Ti sm_61, 248 kB blocks, 4 frames/call)

Verified all 24 runs (12 cases x `{pac, stock}`) byte-exact (648,854,592 bytes
compared in `bench/pac_matrix.py`).

Per-kernel `SLABITZ_PROFILE=1` (us/call, 4 frames/call; initial -> Commit 1 -> Commit 2):

- `silicon_u32_mask0` (`pac raw=0`): `ExecuteSequences` **1313.5 -> 841.6** (Commit 1)
- `silicon_u32_maskmax-1` (`pac raw=0`): `ExecuteSequences` **6623.5 -> 1776.0** (Commit 1)
- `ceo2_u32_maskmax-1` (`pac raw=0`): `Literals` **2812.2 -> 2208.1 us** (-21.5%), `ExecuteSequences` **2855.1 -> 2608.4 us** (-8.6%)
- `ceo2_u32_maskmax-1` (`pac raw=10`): `Literals` **62.2 -> 54.2 us** (-12.9%), `ExecuteSequences` **2711.4 -> 2472.2 us** (-8.8%)
- `ceo2_u32_maskmax-1` (`stock`): `Literals` **6255.9 -> 4748.3 us** (-24.1%), `ExecuteSequences` **6322.7 -> 5666.1 us** (-10.4%)
- `thaumatin_u32_maskmax-1` (`pac raw=0`): `Literals` **2723.3 -> 2117.6 us** (-22.2%), `ExecuteSequences` **9138.2 -> 8573.5 us** (-6.2%)
- `thaumatin_u32_maskmax-1` (`stock`): `Literals` **1126.5 -> 858.6 us** (-23.8%)

Wall time per frame (ms/fr and GB/s):

| Case | Codec | Ratio | Initial | Commit 1 (`MatchCopy`) | Commit 2 (`Exec` + `Literals`) |
|---|---|---:|---:|---:|---:|
| `silicon_u16_mask0` | pac | 274.87 | 0.62 (14.48) | 0.64 (14.05) | **0.59 (15.16)** |
| `silicon_u16_mask0` | stock | 207.55 | 0.82 (10.85) | 0.85 (10.50) | 0.83 (10.79) |
| `silicon_u16_maskmax-1` | pac | 171.39 | 1.36 (6.57) | 0.83 (10.76) | **0.80 (11.24)** |
| `silicon_u16_maskmax-1` | stock | 159.64 | 1.43 (6.23) | 1.38 (6.47) | 1.41 (6.33) |
| `silicon_u32_mask0` | pac | 449.95 | 0.90 (19.81) | 0.85 (21.08) | 0.87 (20.45) |
| `silicon_u32_mask0` | stock | 389.57 | 1.36 (13.18) | 1.40 (12.80) | 1.40 (12.79) |
| `silicon_u32_maskmax-1` | pac | 295.91 | 2.32 (7.71) | 1.23 (14.49) | 1.24 (14.38) |
| `silicon_u32_maskmax-1` | stock | 238.47 | 2.47 (7.23) | 2.46 (7.27) | 2.49 (7.17) |
| `thaumatin_u16_mask0` | pac | 74.35 | 3.09 (11.71) | 2.92 (12.40) | **2.69 (13.46)** |
| `thaumatin_u16_mask0` | stock | 59.64 | 3.56 (10.18) | 3.53 (10.24) | **3.51 (10.30)** |
| `thaumatin_u16_maskmax-1` | pac | 66.41 | 5.37 (6.74) | 3.59 (10.08) | **3.38 (10.69)** |
| `thaumatin_u16_maskmax-1` | stock | 49.08 | 5.55 (6.53) | 5.49 (6.59) | 5.55 (6.52) |
| `thaumatin_u32_mask0` | pac | 139.59 | 4.42 (16.38) | 4.30 (16.84) | **3.99 (18.15)** |
| `thaumatin_u32_mask0` | stock | 116.59 | 5.30 (13.65) | 5.28 (13.70) | **5.23 (13.85)** |
| `thaumatin_u32_maskmax-1` | pac | 124.86 | 9.28 (7.80) | 5.80 (12.49) | **5.51 (13.15)** |
| `thaumatin_u32_maskmax-1` | stock | 84.24 | 9.46 (7.65) | 9.46 (7.65) | 9.50 (7.62) |
| `ceo2_u16_mask0` | pac | 2.57 | 2.34 (3.82) | 2.28 (3.92) | **2.08 (4.30)** |
| `ceo2_u16_mask0` | stock | 2.44 | 3.37 (2.66) | 3.32 (2.69) | **3.13 (2.86)** |
| `ceo2_u16_maskmax-1` | pac | 2.55 | 2.76 (3.24) | 2.51 (3.56) | **2.36 (3.79)** |
| `ceo2_u16_maskmax-1` | stock | 2.43 | 3.53 (2.53) | 3.51 (2.55) | **3.23 (2.77)** |
| `ceo2_u32_mask0` | pac | 5.12 | 2.47 (7.25) | 2.41 (7.42) | **2.23 (8.02)** |
| `ceo2_u32_mask0` | stock | 4.89 | 3.83 (4.66) | 3.84 (4.65) | **3.34 (5.36)** |
| `ceo2_u32_maskmax-1` | pac | 5.08 | 3.59 (4.99) | 2.68 (6.67) | **2.46 (7.27)** |
| `ceo2_u32_maskmax-1` | stock | 4.85 | 4.51 (3.97) | 4.54 (3.94) | **4.01 (4.46)** |

## Update for `pac-zstd` 0.2.0 (`1dba63f`, 2026-11-22)

- `pac-zstd` 0.2.0 added sequence merging (`84fda9e`) and removed
  `pac_set_raw_literals` (`3796c1f`). Merged `off == P` matches produce
  `copyLen > seq.offs`, which initially fell back to the 1-byte-per-lane loop
  (`silicon_u32_maskmax-1` `ExecuteSequences` 6570.6 us/call, wall 2.27 ms/fr).
- Generalizing `zstdgpu_MatchCopy` to chunk aligned matches by `seq.offs`
  (`tools/transform_ds_for_slang.py`, kernel ID `6446e283157f4598`) recovered
  and improved both `pac` and `stock` `maskmax-1` speeds (`bench/pac_matrix.py`,
  24/24 runs ok, 648,854,592 B compared, GTX 1050 Ti `sm_61`):
  - `silicon_u16_maskmax-1`: `pac` ratio 174.70, 1.32 -> **0.81 ms/fr** (11.10 GB/s)
  - `silicon_u32_maskmax-1`: `pac` ratio 316.63, 2.27 -> **1.21 ms/fr** (14.80 GB/s; `ExecuteSequences` 6570.6 -> **2321.1 us/call**)
  - `thaumatin_u16_maskmax-1`: `pac` ratio 66.87, 4.84 -> **3.34 ms/fr** (10.85 GB/s)
  - `thaumatin_u32_maskmax-1`: `pac` ratio 128.28, 8.69 -> **5.45 ms/fr** (13.27 GB/s; `stock` 9.46 -> **7.34 ms/fr**)
  - `ceo2_u32_maskmax-1`: `pac` ratio 5.08, 3.27 -> **2.43 ms/fr** (7.36 GB/s; `ExecuteSequences` 5890.1 -> **2609.5 us/call**)
- `bench/pac_matrix.py` guards `pac_set_raw_literals` with `hasattr` for 0.2.0.

## Verification

- `third_party/slang/bin/slangc` compiles both `kernels/ExecuteSequences.slang` and `kernels/InitHuffmanTableAndDecompressLiterals.slang` for `-target spirv` and `-target metal` with exit code 0.
- `tests/run_bshuf_chain.py` now also tests `$SLABITZ_EXAMPLES/silicon_u16_zstd2_256k.h5` (143,072,512 B compared) and `ceo2_u32_zstd2_256k.h5` (286,145,024 B compared) when present (skips with a notice otherwise).
- Passed after final kernel/lib edits (`CUDA_ARCH=sm_61`, kernel ID `bdf608e50497a7ca`): `run_kernel_id.py`, `run_host_walk.py`, `run_chain_stage01.py`, `run_chain_stage2.py`, `run_chain_frames.py`, `run_bshuf_chain.py`, `run_zstd_async.py`.

## Out of scope

- The pac encoder and its sequence producer (`pac-zstd`).
- Changes to the LZ4 path (already has both cases).
- Multi-warp / cross-block parallelism inside `ExecuteSequences` (deferred to P6; one workgroup per frame remains the remaining floor on `maskmax-1` and `CeO2`).
