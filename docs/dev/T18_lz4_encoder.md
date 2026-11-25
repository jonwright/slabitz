# T18: LZ4 encoder (bitshuffle-LZ4 chunks on the GPU)

Status: todo
Executor: worker (DeepSeek), once T17's data contract is merged; Claude
reviews and owns the compaction kernel if the worker struggles
Depends on: T17
Branch: `t18-lz4-encoder`

## Goal

Write filter-32008 bitshuffle-LZ4 chunks on the GPU from frames in device
memory. The output must be byte-identical to pac-zstd's LZ4 output.
pac's LZ4 path is deterministic and written by hand, so any difference is a
bug.

## Background

- **pac's LZ4 emitter.** `pac_emit_lz4` (`../pac-zstd/src/pac.c:644-689`)
  turns the sequence list into one raw LZ4 block per bitshuffle block, with
  no LZ4 frame format:
  - offsets are 16-bit little-endian;
  - the last 5 bytes are literals, and no match starts in the last 12
    bytes;
  - matches that break those rules become literals.
- **Filter-32008 chunk layout**, as the slabitz decoder reads it (see
  `docs/archive/plan_lz4_slang.md`, "The format"):
  - a 12-byte header: 8-byte big-endian total bytes, then the 4-byte
    big-endian block size in bytes;
  - then, per block, a 4-byte big-endian compressed length followed by
    the LZ4 block;
  - the tail (fewer than 8 elements) is stored raw.
- **Output sizes** vary per block. That needs per-block sizes, then a
  prefix scan, then a compacting copy. Worst case per block: the LZ4 bound
  (`n + n/255 + 16`) plus 4.
- **Reference.** T17's `tools/pac_ref.py` already reproduces pac's LZ4
  output byte for byte.

## Settled design

- **Kernels.** `kernels/Lz4EncodeBlocks.slang` takes one workgroup per
  block. It reads the T17 sequence list and the transposed data, and
  writes the LZ4 block into a worst-case slot plus its size. A prefix sum
  follows (reuse the existing PrefixSum entry if its contract fits).
  `kernels/ChunkCompact.slang` then writes chunk headers, block lengths and
  blocks contiguously into the caller's output. The kernel structure
  follows `kernels/lib/` (stage functions written once).
- **Public API.** It goes in `include/slabitz.h` if T13 is merged,
  otherwise in `src/sbz_encode.h`, moved later:

  ```
  int sbz_bshuf_encode_device(sbz_ctx *ctx, int codec /* SBZ_CODEC_LZ4 */,
      uint32_t elem_size, uint32_t block_bytes,
      uint64_t dev_src, uint32_t nframes, uint64_t frame_bytes,
      uint64_t dev_dst, uint64_t dst_capacity,
      uint64_t dev_chunk_offs /* u64[nframes+1] */, void *stream);
  ```

  The call is queued with no sync. `sbz_ctx_wait` reports overflow of
  `dst_capacity` (the caller sizes it with `sbz_bshuf_encode_bound`).
- **Python.** Add `slabitz.Encoder.encode_device(...)` and
  `encode(frames) -> list of bytes`, if T14 is merged.

## Files that may change

`kernels/Lz4EncodeBlocks.slang`, `kernels/ChunkCompact.slang`,
`kernels/lib/*`, `src/sbz_encode.c`, `src/sbz_encode.h` or
`include/slabitz.h`, `src/sbz_kernels_gen.h`, `tools/build_all.py`,
`tests/run_lz4_encode.py` (new), `slabitz/*.py`, `pixi.toml`,
`docs/dev/reports/T18.md`.

## Acceptance

1. **Byte-identical to pac.** `python3 tests/run_lz4_encode.py` checks
   that the GPU chunks are byte-identical to `pac_compress_chunk(PAC_LZ4)`
   (with the same narrow setting) for:
   - the 12-case matrix;
   - random u8 to u64 data with tails;
   - block sizes of 8 kB, 56 kB (pac's default) and 248 kB;
   - an all-zero frame.

   It prints the chunks and bytes compared.
2. **Independent decoders.** The same chunks decode correctly with
   hdf5plugin / bitshuffle (CPU), with lz4 1.9.x's `LZ4_decompress_safe`
   per block, and with slabitz's own GPU decoder (a full round trip on the
   device).
3. **Throughput.** On the L40S (Slurm), report GB/s of input for silicon
   u16 and CeO2 u32 at 16 frames per call. Compare it with pac on the CPU,
   single thread and all cores of the node, and with nvCOMP LZ4 compress
   if it is available (`bench/baselines.py` has nvCOMP setup).
4. `pixi run test`, `pixi run test-kernels` and `bench/gate.py --compare`
   pass. The decode cells must be unchanged.

## Reviewer checks

- Remove the "no match in the last 12 bytes" rule (temporary): acceptance
  1 must fail, and acceptance 2 may fail with lz4.
- Pick a random chunk, split it into blocks by hand from the header and
  the block lengths, decode each block with the `lz4.block.decompress`
  Python binding (or a small C call to `LZ4_decompress_safe`), and compare
  with `bitshuffle` CPU decode.
