# T13: Public C API `include/slabitz.h`

Status: todo
Executor: Claude
Depends on: T12 (device input for zstd); T09/T10 (error codes)
Branch: `t13-public-api`

## Goal

Add a stable public header that streaming callers, the Python package (T14)
and bslz4_to_sparse (T15) build on. The caller owns the device memory and
the stream, output goes to a strided destination, status is per chunk on
the device, and no call syncs. This is P4 step 3 of
`docs/archive/plan_p4_api.md`, extended with what the bslz4_to_sparse seam
needs.

## Background

- **Settled with the user on 2026-11-20** (`docs/archive/plan_p4_api.md`):
  - caller-owned device memory for the input and output;
  - the library keeps only its own scratch;
  - Python uses ctypes over bslz4_to_sparse's `_backend_drv.py`;
  - a wheel with the fatbins;
  - no HDF5 helper.
- **The step 3 shape settled there:**
  - `sbz_ctx_create(device, kernel_dir, err, errlen)` uses the calling
    thread's current CUDA context. A NULL `kernel_dir` means
    `$SLABITZ_KERNEL_DIR`, else `kernels/` next to the loaded library
    (found with `dladdr`).
  - Output sizing on the host comes from the chunk headers.
  - Scratch belongs to the context and is reused.
  - Each call records an event; the next call's stream waits on it.
  - One context per thread at a time.
  - The old `sbz_dctx_*` calls become thin synchronous wrappers.
  - Only CUDA 6.5 driver calls (Kepler); anything newer goes through
    optional `dlsym`.
- **The bslz4_to_sparse seam** (`../bslz4_to_sparse-dev/src/cuda/_gpu.py`,
  `chunk2sparse._decompress(s0, n, dense, stride)`, about lines 422-439).
  The decode must:
  - fill `dense + f*stride` with unshuffled frames;
  - run on the caller's stream (its `Stream.handle` from the driver
    backend, or the `.ptr` of a cupy stream);
  - write a per-chunk int32 error array `d_err` (0 = ok; the downstream
    kernels skip nonzero chunks).

  Both projects use the driver API on the primary context, so they share
  contexts and streams cleanly.
- **The pipelining lesson** (P4 2a trace): put every input upload on one
  upload stream and record an event after each. The decode stream waits on
  that event. Uploads on two streams at once share PCIe and finish
  together, after which PCIe idles.

## Settled design

- **`include/slabitz.h`.** Only types from `<stdint.h>`/`<stddef.h>`;
  `SBZ_API` visibility; a version macro and a `sbz_version()` call.
- **Context:** `sbz_ctx *sbz_ctx_create(int device, const char
  *kernel_dir, char *err, size_t errlen)` and `void
  sbz_ctx_destroy(sbz_ctx *)`.
- **Geometry and sizing:** `sbz_bshuf_geom` (as today), and
  `int sbz_bshuf_output_bytes(const void *host_src, const sbz_chunk *chunks,
  uint32_t n, uint64_t *bytes)` for callers with host copies.
- **Decode from device memory.** Codec is `SBZ_CODEC_LZ4` or
  `SBZ_CODEC_ZSTD`; returns `SBZ_OK` once queued:

  ```
  int sbz_bshuf_decode_device(sbz_ctx *ctx, int codec,
      const sbz_bshuf_geom *geom,
      uint64_t dev_src, const sbz_chunk *chunks, uint32_t n,
      uint64_t dev_dst, uint64_t dst_stride, uint64_t dst_capacity,
      uint64_t dev_err /* int32[n], may be 0 */,
      void *stream /* CUstream, NULL = legacy default */);
  ```
- **Decode with a host copy:** `sbz_bshuf_decode(ctx, codec, elem_size,
  host_src, dev_src, chunks, n, dev_dst, dst_stride, dst_capacity, dev_err,
  stream)`. The host walk reads `host_src` during the call only.
- **Plain zstd:** `sbz_zstd_decode(...)`, with frames back to back, and
  `sbz_zstd_frame_sizes` for frames without an FCS (T10).
- **Completion:** `int sbz_ctx_wait(sbz_ctx *)` checks capacities and
  replays the slow path; `int sbz_ctx_query(sbz_ctx *)` is non-blocking.
  The documented error codes, including the T09 UNSUPPORTED, are mirrored
  in `dev_err`.
- **Strided output.** `dst_stride` is at least the frame bytes, and need
  not be a multiple of anything beyond the element size. The unshuffle
  kernels take a destination stride argument (the default stride equals
  the frame bytes, which is today's behaviour).
- **External stream.** Every launch and copy of a call goes on `stream`.
  The context's event ordering follows the step 3 settlement.
- **Old API.** The `sbz_dctx_*` functions become thin wrappers, and every
  existing test passes unchanged.

## Files that may change

`include/slabitz.h` (new), `src/*.c`, the unshuffle and LZ4 fused
kernels (stride argument), `src/sbz_kernels_gen.h`, `Makefile` (install the
header), `slabitz/loader.py`, `tests/run_api.py` (new), `pixi.toml`,
`README.md` (API section), `docs/dev/reports/T13.md`.

## Acceptance

1. `tests/run_api.py` (ctypes):
   - decodes both pac zstd files and both LZ4 files through
     `sbz_bshuf_decode_device`, with two contexts on two caller streams in
     alternation, plus one upload stream with events;
   - output into a strided buffer, with stride = frame bytes + 4096 and
     the gaps filled with canary bytes;
   - compares with h5py, byte-exact, nonzero bytes, and 0 canary bytes
     changed;
   - corrupt chunks give their documented code in `dev_err`, and the
     others are intact.
2. **Calls do not block.** At 16 frames per call, the host time of the
   call is less than 25% of the device time of the call (measured in the
   test, printed).
3. **Header portability.** The header compiles as C89 and as C++
   (`gcc -std=c89 -pedantic -fsyntax-only`, `g++ -fsyntax-only`).
4. `nm -D --defined-only build/libslabitz.so` lists only `sbz_*` symbols.
5. `sbz test`, `sbz test-kernels` and `bench/gate.py --compare`
   pass.
6. **Kepler.** On crunch (TITAN Black, driver 340), `tests/run_api.py`
   passes with `build/kernels`, or the report states that crunch was not
   available.

## Reviewer checks

- Pass `dst_stride` = frame bytes - 1: the call must return `SBZ_ERR_ARGS`.
- Remove the event wait between calls on one context (temporary): with
  two streams, `run_api.py` should then show corruption. If it does not,
  note that the test does not cover the ordering, and add a case that
  does.
