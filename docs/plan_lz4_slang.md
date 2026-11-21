# Plan: bitshuffle-LZ4 decoding in Slang

Brief for a subcontractor (agent or person). Self-contained: read it all
before starting. Nobody answers questions while you work. For small choices,
decide and record the decision in your report. If an instruction here is
unclear, or a design choice is not settled here, do NOT guess: write numbered
questions to `QUESTIONS.txt`, finish the parts that do not depend on the
answers, and stop.

## Goal

slabitz decodes bitshuffle-zstd HDF5 chunks (filter 32008, compression 3) on
the GPU, with every kernel written in Slang. Add bitshuffle-LZ4 (filter 32008,
compression 2, the most common format from Dectris and Lambda detectors) so
that this one library decodes every bitshuffle plugin variant on the GPU, from
one Slang source, for CUDA now and Vulkan/Metal later.

The LZ4 decoder exists today as CUDA C in a sibling project,
bslz4_to_sparse (`src/cuda/`). It is a port of NVIDIA's open-source nvCOMP
LZ4 decompressor (BSD-3). Port it to Slang here; do not change
bslz4_to_sparse.

Output: a public C entry point `sbz_dctx_bslz4_decompress`, byte-exact against
hdf5plugin on every test case, at least as fast as the bslz4_to_sparse CUDA
`decompress` kernel on the same GPU.

## Read first (in this order)

1. `AGENTS.md`: the project rules. All of them apply. In particular: one
   kernel per Slang entry file; entry points take `uint2 groupId2 :
   SV_GroupID` through `zstdgpu_ConvertTo32BitGroupId`; a test that compares
   nothing must fail; run every GPU test under `timeout`.
2. `kernels/BitUnshuffleWarp.slang`, `kernels/BitUnshuffleSlow.slang` and
   `kernels/MemsetMemcpy.slang`: the house style for Slang entries with byte
   buffers, uint32/uint4 views of the same buffer, and scalar parameters.
3. `src/sbz_chain.c`, from `bshuf_walk_chunk` to the end: the host side of
   bitshuffle-zstd (chunk walk, frame list, unshuffle launches, raw tails),
   and the buffer registry (`SBZ_BUFFER_LIST`, `kBufInfo`, the `raw` view
   rule), `launch()` and `sbz_bind` overrides.
4. `src/sbz_chain.h`: the public API.
5. `tests/run_bshuf_chain.py`: how chunks are made with h5py + hdf5plugin and
   compared.
6. The reference LZ4 decoder, read-only, in a local clone of bslz4_to_sparse (`src/cuda/`):
   - `nvcomp_lz4.cuh`: `decompressStream` (one warp per LZ4 block: tokens,
     LSIC lengths, literal copy, match copy, the 256-byte input staging buffer
     `BufferControl`, bounds checks that make a corrupt block return false).
   - `bslz4_split.cu`: `unblock` (block headers) and `decompress` (decode one
     8 kB block into shared memory, then unshuffle it in registers).
   - `shuffle_warp.cuh` (the 8 kB register unshuffle).

## The format (bitshuffle-LZ4 chunk)

The container is identical to bitshuffle-zstd. With elements of e bytes:

- `[BE u64 total bytes][BE u32 block bytes]`. Block bytes default to 8192
  (any multiple of 8e is legal). B = block bytes / e elements per block,
  S = total / e.
- floor(S/B) full blocks, then one last block of (S%B) - (S%B)%8 elements if
  nonzero. Each block is `[BE u32 c][c bytes]`: one raw LZ4 block (LZ4 block
  format, no LZ4 frame header), which decodes to the block's elements
  bit-transposed.
- Then (S%8)*e raw bytes, stored untransposed.

`bshuf_walk_chunk` in `sbz_chain.c` already walks exactly this layout. Reuse
it; do not write a second walker.

LZ4 block format: a sequence is a token byte (4-bit literal length, 4-bit
match length), the literal length continues with LSIC bytes (each 255 means
"add and continue") if the nibble is 15, then the literals, then a 2-byte
little-endian offset (1..65535; 0 is corrupt) and the match length (nibble +
4, continued with LSIC if the nibble is 15). The last sequence has literals
only. A match may overlap its own output (offset < length): bytes repeat with
period `offset`.

## Design (settled; do not redesign)

### Phase 1: straight port, correct everywhere

- One new Slang kernel, `kernels/Lz4DecodeBlocks.slang`: one wave of 32 lanes
  per LZ4 block, the decode loop of `decompressStream` ported line by line. It
  writes each block's decoded (still bit-transposed) bytes to a global buffer
  laid out exactly like `UnCompressedFramesData` for zstd (blocks back to back
  in chunk order). The existing unshuffle launches in `bshuf_unshuffle()` then
  run unchanged on that buffer. This keeps decode and unshuffle separate, as
  for zstd: the unshuffle costs 27 us per 8.9 MB chunk on an L40S, so fusing
  them is not worth a second unshuffle implementation.
- Thread group: 64 threads (2 waves, 2 blocks per group, as the reference
  found faster) or 32; measure both on the L40S and record the result.
- Shared memory (`groupshared`): the per-wave input staging buffer (256
  bytes, as `BufferControl`) and the block's decode buffer. The decode buffer
  size is the block size. For block sizes that do not fit, decode straight
  into the global output buffer instead. Pick one size limit, record it, and
  make the kernel take the path per launch (a scalar), not by `#ifdef`.
- Wave operations: use only Slang's portable wave intrinsics
  (`WaveActiveBallot`, `WaveReadLaneAt`, `WaveReadLaneFirst`, `WaveGetLaneIndex`).
  `__ballot_sync` in `lengthOfMatch` becomes `WaveActiveBallot`. Do not use
  `WavePrefixCountBits`: slangc 2026.22 has no Metal lowering for it (it is
  why 4 of the zstd kernels do not compile for Metal yet). Keep every
  loop that contains a wave operation uniform across the wave (same trip
  count on all lanes): this is a hard rule in AGENTS.md, and violating it
  hangs the GPU.
- Barriers: inside one wave no barrier is needed on CUDA, but Vulkan and
  Metal require one before lanes read groupshared bytes written by other
  lanes. Use `GroupMemoryBarrierWithWaveSync()` or the closest Slang
  equivalent that compiles on CUDA, SPIR-V and Metal, and say which.
- Buffer lengths: pass every byte count as a scalar parameter. Do not call
  `GetDimensions`: Metal compute does not support it (slangc error E36107).
- Corrupt input must never read or write outside its block's input or
  output range. A failing block writes a nonzero status per chunk (a new
  `uint32_t` buffer, one entry per chunk). The host turns any nonzero status
  into `SBZ_ERR_CORRUPT` with the chunk index in the message. Keep every bounds
  check of the reference: offset 0, offset beyond the decoded start, a match
  or literal run past the block's end, and a decoded size different from the
  block size.
- No `#ifdef` on target or architecture. The pre-Volta CUDA shim
  (`tools/slang_cuda_prevolta.h`) is already applied to every Slang kernel by
  the build. Nothing else is needed for sm_35, sm_50 or sm_61.
- Host side, in `src/sbz_chain.c`:
  - `sbz_dctx_bslz4_decompress(dc, src, srcSize, chunks, nchunks, elemSize)`,
    with the same arguments and output (`BshufOutput`) as
    `sbz_dctx_bshuf_decompress`.
  - Factor the shared parts (chunk walk, output layout, unshuffle, raw tails)
    out of `sbz_dctx_bshuf_decompress` into a static helper that takes the
    decode stage as the only difference. Do not copy-paste the 150 lines.
  - Upload the chunks once. Build the block table (input offset, compressed
    size, decoded size, output offset, chunk index) on the host from the walk,
    upload it, and launch the decode once for all blocks of all chunks.
  - Declare it in `src/sbz_chain.h`.
- Licence: the Slang kernel is a derivative of NVIDIA's BSD-3 code. Put the
  full NVIDIA copyright and licence text at the top of the kernel file,
  followed by a line saying it was ported from nvCOMP via bslz4_to_sparse.
  Add an entry to `THIRD_PARTY_NOTICES.md`.
- The build: add the kernel to `KERNEL_MAP` in `tools/build_all.py`. Commit
  the regenerated `src/sbz_kernels_gen.h`.

### Phase 2: speed-ups from the DirectStorage zstd design (done)

Outcome (worker run of 2026-11-19, reviewed; L40S, `Lz4DecodeBlocks` device
time, 16 frames/call):

- Step 0, the 256-byte input staging buffer (nvCOMP `BufferControl`): kept,
  d62b28c. Silicon 275 -> 270 us/call, CeO2 894 -> 837. The review added a
  group barrier before each refill: unlike the reference, a refill can come
  from the match-length LSIC, after the literal copy has read the window.
- Step 1, short-offset matches from registers (`WaveReadLaneAt(src, i %
  offset)`): 110% slower on silicon, 37% on CeO2. Reverted.
- Step 2, pairs of short sequences in one store: 3% slower. Reverted.
- Step 3, parse then execute: dropped. The decode alone is already faster
  than the reference's fused decode+unshuffle kernel.

Where the time goes after Phase 2 (device us per frame, 16 frames/call):

| | silicon u16 | CeO2 u32 |
|---|---|---|
| slabitz `Lz4DecodeBlocks` | 16.9 | 52.3 |
| slabitz BitUnshuffleSlow + Warp | 36.5 | 64.6 |
| reference `decompress` (decode + unshuffle fused) | 19.7 | 66.9 |
| upload (PCIe, both decoders) | 11 | 183 |

The gap to the reference is the separate unshuffle pass: a second read and
write of every output byte. The Phase 1 argument against fusion ("27 us per
8.9 MB chunk, not worth it") holds for zstd, where the unshuffle is 12-20% of
the device time, but not for LZ4, where it is two thirds. Phase 3 fuses them.

How the reference was timed: `bench/ref_bslz4_cuda.py` (its docstring says
how to set it up). The GPU reference is
`bslz4_to_sparse.cuda.chunk2sparse(mask, dtype).decompress(chunks)`; the
top-level `bslz4_to_sparse.decompress` is the CPU decoder. ncu does not work
on the L40S node (`RmProfilingAdminOnly: 1`); kernel times come from CUDA
events (`SLABITZ_PROFILE=1`, `ref_bslz4_cuda.py --kernels`).

### Phase 3: fused decode + unshuffle, batched unshuffle launches (done)

Outcome (worker run of 2026-11-20, reviewed and re-measured; L40S, 16
frames/call, device us per frame with every op synced):

| | silicon u16 | CeO2 u32 |
|---|---|---|
| unfused (decode + unshuffle), same build, `SLABITZ_LZ4_FUSED=0` | 17.4 + 29.0 = 46.3 | 53.8 + 55.5 = 109.3 |
| fused `Lz4DecodeUnshuffle` + partial last blocks | 23.2 + 1.8 = **25.0** | 57.6 + 1.3 = **58.9** |
| reference `decompress`, same session | 19.6 | 66.9 |

- Steps 1-3 kept (d6ac917, cfa1022, cabca01, test 0ea3a3a). The review
  replaced the fixed 32-entry unshuffle bucket array, which failed calls
  with more than 32 distinct partial-block sizes, by a growing list
  (8188582, test `u16-many`).
- Fused and unfused output are byte-identical on every test and on both real
  files (32 frames). L40S and TITAN Black (sm_35).
- CeO2 meets the target. Silicon does not (25.0 vs 19.6). The fused kernel
  stores each lane's 32 elements directly (lane stride 32*e bytes, not
  coalesced across the warp); the reference stages them through a second
  shared buffer and flushes coalesced warp stores. A second 8 KB buffer next
  to the tile and the 256-byte input window is 16.6 KB, over the 16 KB rule.
  Accepted as is: at the wall slabitz is faster than the reference on both
  files (16 frames/call: silicon 694 vs 828 us/call, CeO2 5.1 vs 7.5 ms;
  1 frame/call: 97 vs 402 us, 340 vs 820 us), and with uploads overlapped
  (see "Streaming" below) CeO2 is PCIe-bound anyway. If silicon-like data
  ever needs it: decode with the tile itself as the input window, so the
  store staging fits under 16 KB.
- zstd: decode kernels unchanged (ExecuteSequences 1395 / 1797 us/call
  before and after). Unshuffle 32 -> 2 launches per 16-frame call; 533 ->
  416 us/call silicon, 1001 -> 891 CeO2 (synced, so the old small launches
  look slower than they are).
- Step 4: the zstd "fixed cost" of about 1.5 ms per call at 1 frame/call
  (silicon) is device time, not launch overhead: ExecuteSequences 1.0 ms and
  DecompressSequencesMultiStream 0.29 ms of it; the other ~15 kernels are
  6-8 us each, synced. CUDA graphs would save at most ~0.2 ms there. LZ4's
  fixed cost is ~0.1 ms per call.
- Compile targets: `Lz4DecodeUnshuffle` and `Lz4DecodeBlocks` build for
  SPIR-V and Metal; the two unshuffle kernels still fail on Metal
  (`GetDimensions`), as before.
- Timing pitfall: the test harnesses' `decode()` joins the chunks on every
  call; at 16 CeO2 frames (63 MB) that makes a call 45 ms instead of 5 ms.
  Build the input once outside the timed loop.

Streaming: the intended use is a stream of frames, with uploads overlapped
with decoding, the decoded frames kept on the GPU and reduced there, and only
results smaller than the compressed input copied back. So the throughput is
set by max(upload, device time) per frame, not their sum. On the L40S
(upload at ~21 GB/s, pageable): silicon LZ4 11 vs 25 us (device-bound), CeO2
LZ4 183 vs 59 us (PCIe-bound), silicon zstd ~150 us device with a small
upload, CeO2 zstd 193 vs ~300 us (device-bound). Further LZ4 kernel work therefore pays only
for well-compressed data; for zstd, ExecuteSequences is the limit (AGENTS.md
P6).

Goal (as planned): LZ4 device time per frame at or below the reference's
fused kernel (19.7 / 66.9 us per frame for silicon / CeO2 at 16
frames/call), with each stage still written once.

#### The fusion rule (also in AGENTS.md)

Fuse a producer kernel into its consumer only when all three hold:

1. Same work unit: one workgroup produces exactly the tile the consumer
   needs.
2. The tile fits in groupshared memory on every target: at most 16 KB per
   group in all (the Vulkan guaranteed minimum; Metal 32 KB, CUDA 48 KB).
3. The intermediate's round trip through device memory is a large share of
   the time, measured.

LZ4 with bitshuffle blocks of at most 8 kB passes. LZ4 with 248k/256k blocks
fails 2 and keeps the separate kernels. zstd `ExecuteSequences` -> unshuffle
fails 1 and 2 (one workgroup per frame, 256 kB bitshuffle blocks): no zstd
fusion in Phase 3. Never merge kernels only to save launches (that is a
launch-batching or graph question, not fusion).

#### Step 1: stage functions shared by fused and unfused kernels (no behaviour change)

- Move the stage code into include files under `kernels/lib/` (not entry
  points; `KERNEL_MAP` does not list them):
  - `kernels/lib/lz4_decode.slangh`: `lzDecodeBlock` and its helpers, generic
    over where the decoded bytes go. The decoder reads back its own output
    (match copies), so the target interface has load and store:
    `interface ILz4Target { uint8_t load(uint32_t i); void store(uint32_t i,
    uint8_t v); }`, with two structs: `GlobalTarget` (the
    `RWStructuredBuffer<uint8_t>` and a base offset) and `TileTarget` (a
    file-scope `groupshared` tile).
  - `kernels/lib/bit_unshuffle.slangh`: the transpose core of
    `BitUnshuffleWarp` and the per-lane "load 8*e plane words of one 32-element
    group, transpose, store 32 elements" step, generic over the source of the
    plane words: `interface IPlaneSource { uint32_t word(uint32_t
    byteOffset); }`, with `GlobalPlanes` (the uint32 view) and `TilePlanes`
    (the groupshared tile). The store path (uint4 / uint32 / bytes by
    alignment) is shared as is.
- `Lz4DecodeBlocks.slang` and `BitUnshuffleWarp.slang` become thin entry
  files that instantiate these with the global structs.
- First, before moving any code: a compile spike. One small generic function
  over an interface with a buffer-backed and a groupshared-backed struct,
  compiled for CUDA, SPIR-V and Metal with slangc 2026.22. If any target
  rejects it, or the CUDA output is not a direct call (check the `.cu`), stop
  and write QUESTIONS.txt. Do not fall back to copying the code.
- Acceptance for this step: generated kernels behave identically; full test
  suite passes; device times within 1% of before (`SLABITZ_PROFILE=1`, both
  LZ4 files and both zstd pac files). One commit.

#### Step 2: `kernels/Lz4DecodeUnshuffle.slang`

- One 32-lane group per bitshuffle block, as `Lz4DecodeBlocks`. Groupshared:
  the 256-byte input window and one tile of `LZ4_FUSED_MAX_BLOCK` = 8192
  bytes (8.5 KB per group).
- Per group: decode the block into the tile through `TileTarget`; if the
  block failed, write the chunk status and return (uniformly: every lane
  takes the same branch); else `AllMemoryBarrierWithGroupSync()`, then
  unshuffle from `TilePlanes` straight into `BshufOutput`. The 32 lanes loop
  over the block's 32-element groups (blockElems / 32 of them, at most 256);
  the trip count is the same on every lane.
- Eligible blocks: full blocks with blockElems * e <= 8192, e in {1, 2, 4}
  and blockElems % 32 == 0 (the cases `BitUnshuffleWarp` handles). Every other
  block (the partial last block of a chunk, e = 8, larger blocks) goes through
  `Lz4DecodeBlocks` + the separate unshuffle, as now. Both launches can occur
  in one call; the host splits the block table.
- The fused path never writes `UnCompressedFramesData`. Environment variable
  `SLABITZ_LZ4_FUSED=0` (read once per dctx, like `SLABITZ_PROFILE`) forces
  the unfused path for every block: that is the debugging path
  (`sbz_dctx_read_buffer` of the decoded bytes) and the test oracle.
- Corrupt input: the same checks and the same per-chunk status as
  `Lz4DecodeBlocks`; a failing block writes nothing to `BshufOutput`.
- Add to `KERNEL_MAP`, regenerate and commit `src/sbz_kernels_gen.h`.
- Thread group: 32 threads (settled). Do not try 64: two tiles would be
  17 KB, over the Vulkan limit.
- One commit, with before/after device times per frame and the reference's.

#### Step 3: one unshuffle launch per kernel per call

Today `bshuf_unshuffle()` launches per chunk: 2 launches per frame, 32 per
16-frame call, each with a few us of fixed cost. This step is shared by the
zstd and LZ4 paths.

- `BitUnshuffleWarp` and `BitUnshuffleSlow` take a block table: a new buffer
  of `uint2 {inByteOffset, outByteOffset}` per block (uint32 offsets, as now),
  plus `blockElems` and `elemSize` as scalars. The host builds the table for
  all chunks of the call and launches each kernel once per distinct
  (kernel, blockElems) pair: in practice once for the full blocks and once
  for the partial last blocks.
- Regenerate `src/sbz_kernels_gen.h`. One commit with the launch count and
  the device time per call before and after (zstd and LZ4, 1 and 16
  frames/call).

#### Step 4: measure the fixed cost per call (report only, no code kept)

Without `SLABITZ_PROFILE` (which syncs after every op), time the wall per
call and the device-busy time per call (a CUDA event at the start and at the
end of the call's stream work) at 1, 4 and 16 frames/call, for the two zstd
pac files and the two LZ4 files. Report wall - device per call. This number
decides whether CUDA graphs are worth doing later (AGENTS.md, P6).

### Non-goals

- Do not change bslz4_to_sparse or the zstd kernels (`third_party/`, the DS
  transform). Phase 3 step 3 changes only the two unshuffle kernels and the
  host.
- No fusion on the zstd path.
- No CUDA graphs, no indirect dispatch (later, AGENTS.md P6).
- No LZ4 frame format (only the raw LZ4 blocks that bitshuffle writes).
- No compression.
- No Vulkan or Metal host backend: those targets are compile-checked only.

## Tests (new `tests/run_bslz4_chain.py`, modelled on `run_bshuf_chain.py`)

- Chunks made with h5py + `hdf5plugin.Bitshuffle(cname="lz4")`, read back raw
  with `read_direct_chunk`, decoded on the GPU, and compared byte for byte
  with h5py's own read. Cover:
  - e = 1, 2, 4, 8;
  - block sizes 8192 (the default), 256 kB and a non-power-of-two multiple
    of 8e;
  - chunks with a partial last block, a raw tail, and both;
  - several chunks per call;
  - zeros, random (incompressible: the LZ4 block is all literals), smooth
    detector-like data, and data with many short repeats (offsets 1..40, to
    exercise overlapping matches).
- The real files, at least the first 32 frames of each, compared with h5py
  (`silicon_lz4.h5` and `ceo2_lz4.h5` from the read-only example data,
  `SLABITZ_EXAMPLES`). Read them only; never write there.
- Corrupt input must give `SBZ_ERR_CORRUPT`, with no hang and no crash. Cover:
  a truncated chunk, a block size record past the end, offset 0, an offset
  before the block start, a match running past the block end, and a block
  that decodes short.
- Print the number of bytes compared per case. Zero compared is a failure.
- Add the test to the `test` task in `pixi.toml` and to `../build_test.sh`'s
  list if you use that script.

Phase 3 additions to `tests/run_bslz4_chain.py`:

- Run every case twice, with `SLABITZ_LZ4_FUSED=0` and `=1` (one process
  each), and compare both with h5py and with each other. Print, per case,
  how many blocks took the fused path; the 8 kB cases with e in {1, 2, 4}
  must report a nonzero count with `=1` and zero with `=0`.
- At least one call that mixes both paths: 8 kB blocks with a partial last
  block and a raw tail, several chunks per call.
- A corrupt block on the fused path (offset 0 and decode-short inside an
  eligible 8 kB block): `SBZ_ERR_CORRUPT`, the right chunk index, no hang.
- The real files (first 32 frames of `silicon_lz4.h5` and `ceo2_lz4.h5`, both
  paths) compared with h5py. Phase 1 compared them only inside
  `bench/old_cards.py`; put the check in the test, skipped with a printed
  notice when the files are not readable.
- The zstd tests (`run_bshuf_chain.py`) cover the batched unshuffle of step 3;
  they must keep passing with their compared-byte counts unchanged.

## Build, verify, measure

Work in a scratch clone of this repository (an external scratch checkout;
Phases 1-2 used separate scratch clones). Then replace `third_party/slang`
with a symlink to the workspace's copy. Never edit the
main workspace or the maintainer's checkout. In the clone:

- Full build and all regression tests: `bash ../build_test.sh`, about 8
  minutes. All must pass, plus the new test.
- Single test after a build: `. ../activate.sh && timeout 600 python3 tests/run_bslz4_chain.py`.
- Kepler compile check, which must succeed: `. ../activate_kepler.sh && python3
  tools/build_kubins.py --arch sm_35 --out build/sm_35 --nvrtc
  $CONDA_PREFIX/lib/libnvrtc.so`. The architect runs the result on the
  Kepler machine. You cannot reach it; do not try.
- Other targets, compile only: `third_party/slang/bin/slangc
  kernels/Lz4DecodeBlocks.slang -target spirv` (and `-target metal`), with the
  same `-I` and `-entry` flags as `tools/build_all.py`. Both must succeed.
- Speed:
  - Add LZ4 support to `bench/old_cards.py`: detect the compression from the
    dataset's filter options and call the matching entry point.
  - Run it on the L40S for `silicon_lz4.h5` and `ceo2_lz4.h5`.
  - For the reference, use `bench/ref_bslz4_cuda.py` (set-up in its
    docstring; build the package of symlinks under `./scratch_work`). It
    times `bslz4_to_sparse.cuda.chunk2sparse.decompress`, the GPU decoder,
    not the top-level `decompress` (CPU). Record the exact command used.
  - Use `SLABITZ_PROFILE=1` for per-kernel times. ncu does not work on this
    node (admin-only performance counters): do not try to make it work.
  - Phase 3: compile-check `Lz4DecodeUnshuffle.slang` and the changed
    unshuffle kernels for SPIR-V and Metal as well (BitUnshuffleWarp does not
    compile for Metal today because of `GetDimensions`; record whether the
    changed file still uses it, and do not add new uses).

## Rules

- ASCII only.
- Never read or write outside the scratch clone, except to read the paths
  named above. Use `./scratch_work` for temporary files and delete it before
  committing.
- Do not push and do not ssh. Do not `git add` `third_party/slang` (a
  symlink) or logs.
- Do not edit `third_party/` or generated `.cu` files.
- Commits:
  - Phase 1: one commit, `bitshuffle-LZ4: Slang LZ4 block decoder and
    sbz_dctx_bslz4_decompress (drafted by <agent>)`.
  - Phase 2: one commit per kept step.
  - Phase 3: one commit per step 1-3 (step 4 is report only), each saying
    `(drafted by <agent>)`.
  - Every message gives the test results and the timing numbers and ends with
    a `Co-Authored-By:` trailer naming you.

## Deliverables

1. The commits above in the scratch clone.
2. `REPORT_lz4.md` in the clone root (Phase 3: `REPORT_lz4_phase3.md`; not
   committed), containing:
   - the test results per case, with bytes compared;
   - a timing table: slabitz LZ4 vs the bslz4_to_sparse `decompress` kernel,
     per frame, for both files, at 1, 4 and 16 frames per call;
   - the SPIR-V and Metal compile results;
   - every decision you made and why;
   - anything left undone.

Do not end your turn until `REPORT_lz4.md` exists. If a step fails, record the
failure in the report and move on. Finish with a short summary on stdout.

## Acceptance (checked by the architect)

- All regression tests and the new test pass on the L40S and, with the sm_35
  build, on the Kepler TITAN Black.
- Output is byte-exact against h5py on the real files.
- Corrupt input gives an error, never a hang, crash or out-of-bounds access.
- The decoder is no slower than the bslz4_to_sparse `decompress` kernel on the
  L40S, comparing decode plus unshuffle against its fused kernel, at 16 frames
  per call.
- The new kernel compiles for SPIR-V and Metal.

Phase 3, in addition:

- Each stage is written once: `Lz4DecodeBlocks`, `BitUnshuffleWarp` and
  `Lz4DecodeUnshuffle` are thin entry files over `kernels/lib/`. No copied
  decode or transpose code.
- Fused and unfused paths give identical output on every test case.
- L40S, 16 frames/call: LZ4 device time per frame (all kernels of the call)
  at or below the reference's `decompress` kernel: 19.7 us silicon, 66.9 us
  CeO2, re-measured in the same session.
- zstd device times unchanged by steps 1 and 2; step 3 lowers the launch
  count per call to a constant (independent of the number of frames).
- The step 4 table (wall - device per call) is in the report.
