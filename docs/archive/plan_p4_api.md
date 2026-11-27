# Plan: P4, streaming API, Python wrapper, wheel

> Archived 2026-11-25. Folded into the task files in `docs/dev/` (P4: T12-T16;
> P5: T02, T06, T11). Kept for its history and settled decisions; the task
> files are the instructions.

The P4 streaming item in `AGENTS.md`. Today every `sbz_dctx_*` call copies its input from
host memory, reads counters back to the host several times, and syncs before
returning. The target use is a stream of frames: the caller uploads batch
n+1 while batch n decodes, keeps the decoded frames on the GPU, reduces them
there, and copies back only small results. So the throughput is max(upload,
device time) per frame, and a decode call must queue its work on the
caller's stream and return without waiting for the GPU.

Decided with the user (2026-11-20):

- Caller-owned device memory: the caller passes device pointers for the
  compressed input and the decoded output, and a stream. The library keeps
  only its own scratch.
- Python: ctypes, reusing bslz4_to_sparse's driver backend
  (`src/cuda/_backend_drv.py` in a local clone, MIT, same author)
  for device arrays (`__cuda_array_interface__`, `__dlpack__`), streams,
  pinned memory and the memory pool. A vendored c2py23 binding is
  considered later, only if call overhead shows up.
- Packaging: a pip wheel carrying `libslabitz.so` and the `build/kernels`
  fatbins as package data; the kernel directory defaults to the package,
  `SLABITZ_KERNEL_DIR` overrides.
- No HDF5 helper in P4 (later).

The steps are in order; each ends in one commit with its tests. Steps 1 and
2 are subtle C runtime work (counts, buffer bounds, stream ordering): keep
them with a strong agent, not the DeepSeek worker. Steps 4 and 5 suit the
worker once steps 1-3 are in.

## Read first

1. `AGENTS.md` (all of it), `docs/history.md` items 13, 19, 23.
2. `src/sbz_chain.c`: `read_counters()`, `stage0()`, `stage1()`,
   `stage2()`, `bshuf_decode_lz4()`, `bshuf_common()`, `host_prepare()`.
3. `src/sbz_cuda.c` (context and stream handling, `load_kernel`),
   `src/sbz_backend.h`, `src/sbz_chain.h` (the current API).
4. The DS shaders in `third_party/directstorage_zstdgpu/` (only the
   shaders are vendored; the D3D12 host code is not. Its design: buffers
   sized from the CPU walk, indirect dispatch for counts only the GPU
   knows. `src/host_walk.c` is the port of its CPU walk).
5. A local clone of bslz4_to_sparse (`src/cuda/_backend_drv.py`) and its use in
   `_gpu.py` (how a Python class launches on the current stream and hands
   out arrays).

## Step 1: readback audit and measurement (no library code)

Status (2026-11-20): done, waiting for the user's review:
`docs/archive/p4_readback_audit.md` and `bench/p4_overlap.py` (929609a). A first
attempt worked in the maintainer's checkout against a stale `build/` and reported the
crash as a decoder bug; the library now refuses mismatched kernels
(`sbz_kernels.id`). Work in the pixi workspace or a scratch clone. Time only
on an idle GPU: the L40S login-node GPU is shared, so use a Slurm job on
the `low-gpu` partition (`--gres=gpu:nvidia_l40s:1`).

Write `docs/archive/p4_readback_audit.md`:

- One row per host readback or sync inside a call. Today: `read_counters()`
  at the end of stage 0, stage 1 (plus the `FrameStatus` download), in the
  middle of stage 2 (`DecompressLiteralsGroups`) and at its end; the LZ4
  status download in `bshuf_decode_lz4()`; the final sync in
  `bshuf_common()`.
- For every value read back: what it is used for (a buffer size, a grid
  size, a `workItemCount` scalar, a consistency check, an error), and its
  replacement:
  - a host upper bound from the host walk (frames and blocks per type are
    known exactly; per compressed block the decoded literals are at most
    the block's decoded size, at most 128 KiB; give the bound used for
    sequences, FSE tables, Huffman weight streams and literal groups, with
    the zstd format rule that justifies it), with the worst-case memory for
    the pac files (silicon and CeO2, 16 frames per call);
  - or the count read on the device: the kernel is launched with the
    worst-case grid and reads its real count from the `Counters` buffer,
    exiting early (the CUDA form of DS's indirect dispatch). List which
    Slang entries need that change, and whether it goes in the entry file
    or the DS transform.
  - Consistency checks and errors become device-side status (below).
- Measure before redesigning: two contexts (`sbz_dctx`) used from two host
  threads, each with its own stream, decoding alternate batches of
  silicon zstd, CeO2 zstd and both LZ4 files at 16 frames per call. Does
  the upload of one overlap the decode of the other today? Report frames
  per second against max(upload, device) from `SLABITZ_PROFILE=1`.
- Stop after this step for review by the user (it decides the scope of
  step 2).

## Step 2: no host wait inside a call

Decisions from the user's review of the audit (2026-11-20):

- Sizing: the target is an Eiger2 16M with 16-bit pixels, about 35 MB per
  frame; the number of frames per call is chosen to fit the card, and the
  smallest cards have about 1 GB. Worst-case scratch of the order of the
  output size is therefore acceptable. Size buffers from the host bounds of
  the audit; only cache-aware tuning would care about tighter bounds.
- Errors are extremely rare (one corrupt chunk seen in five years). The
  error path may be slow: on a nonzero device status the library may redo
  the call synchronously with the old readbacks and checks to locate and
  report the error. Only the no-error path must not wait.
- Host work inside a call (the chunk walk and the block tables, as for
  bslz4) is acceptable for now, but a "full GPU" option is wanted: chunk
  headers walked and block tables built on the device, so a call needs no
  host pass over the compressed bytes (needed for device-only input). Plan
  it after step 2b as step 2c.
- Keep going: step 2b (the zstd readbacks) is approved.

Following the reviewed audit:

- LZ4 first (simplest): the per-chunk status stays on the device; the call
  returns after queueing the decode and unshuffle. Done 2026-11-20 (912d07e,
  step "2a"): `sbz_dctx_bslz4_decompress_async` and `sbz_dctx_wait` in
  `src/sbz_chain.h`, a per-context page-locked staging buffer for the
  library's own tables, `run_bslz4_chain.py --async two-ctx`, results in the
  commit message and `bench/p4_overlap.py`. Then `SLABITZ_TRACE=1` and
  `bench/p4_trace_lz4.py` (6bfe47c): the input upload moved before the host
  walk; the remaining gap (CeO2 about 3.3 ms per call against 2.35 ms of
  upload) comes from the two contexts uploading at the same time, sharing
  PCIe, then running their kernels together while PCIe idles.
- zstd: worst-case sizes from the host walk, counts read on the device,
  every readback removed. Keep `SLABITZ_PROFILE=1` working (it syncs per op
  by design) and keep a debug switch that restores the checks with
  readback (`SLABITZ_CHECK_SYNC=1`) for diagnosing corrupt input.
  Done 2026-11-20 (806b260, step "2b"): block counts from the host walk;
  grids from host bounds, with InitFseTable and PropagateFseIndex reading
  their counts from `Counters` (`workItemCount` ~0u) and group bounds added
  to InitFseTable and the literals kernel (the other DS kernels already
  read their counts on the device). The literal and sequence buffers are
  not bounded by the compressed size (a sequence can cost zero bits), so
  they use per-context capacities learned from exact calls (+25%); the
  kernels touching them exit if the counts exceed them, and `sbz_dctx_wait`
  checks the counters once and replays the call exactly on an overflow or
  corrupt input (the slow path). First call of a context, `sbz_dctx_parse`
  and `SLABITZ_CHECK_SYNC=1` run exactly. `sbz_dctx_bshuf_decompress_async`,
  `sbz_dctx_zstd_stats`, `tests/run_zstd_async.py`. Also fixed: two
  contexts in one thread could share a CUDA context the first one owned;
  now the primary context is retained (own contexts only on driver 340).
  Idle L40S, one thread alternating two contexts, 16 frames per call: silicon
  zstd 2.58 -> 1.98 ms (three contexts 1.65), CeO2 zstd 9.42 -> 5.63 ms.
  Then (zero-copy input): the host walk reads bytes in place and the zstd
  paths upload the caller's buffer as it is, with frame refs into it; with
  pinned input the CeO2 call holds the host 0.43 ms instead of 4.8 (serial
  9.42 -> 7.62 ms, three contexts 4.13 ms).
- Step 2c, full-GPU option. LZ4 done 2026-11-21 (1ecfae2): the design below
  with a BshufWalkChunks kernel; `tests/run_bslz4_device.py`. The walk is a
  chain of dependent reads (412 / 1255 us per 16-chunk call on silicon / CeO2);
  groupshared windows and touch-ahead lanes did not help; a real prefetch
  (PTX, CUDA only) is the idea left. zstd: to do.
  Design: a call that needs no host
  pass over the compressed bytes, so the input may live only on the device.
  What the host walk gives today, and its device replacement:
  - bitshuffle chunk headers (total bytes, block size) and the block
    records (BE u32 size + data): a kernel with one thread (or wave) per
    chunk walks its records, which are sequential within a chunk, and writes
    the block table; a prefix sum over chunks gives the output offsets. The
    caller states the chunk count and an upper bound of the chunk size
    (detector geometry), which sizes everything.
  - zstd frame and block counts and content sizes (for sizing and output
    offsets): DS's ParseFrames already counts and collects on the device;
    what is missing is sizing without the host, so size from bounds
    (blocks <= compressed bytes / 3, frames <= chunks x blocks per chunk,
    output = the caller's stated frame size) and keep the capacity check at
    the wait.
  - LZ4: the block table from the same chunk walk kernel.
  The host then only launches; errors and overflows go to the device status
  and the exact replay (which needs the bytes on the host: copy them back
  on that rare path).
- Per-frame (zstd) and per-chunk (bitshuffle) status (still to do; today the
  wait reports errors and the status stays internal): a device `uint32`
  array the caller provides, 0 = OK, nonzero = an error code documented in
  the header. Errors found by the host walk are still returned at once.
- Acceptance: all tests pass with byte-identical output, using the new
  status for every corrupt-input case; per-kernel device times unchanged
  (within 2%); a call at 16 frames returns before the GPU finishes (measure
  host time of the call vs device time).

## Step 3: public C API, `include/slabitz.h`

- Settled shape (names may be adjusted, the semantics not):
  - `sbz_ctx *sbz_ctx_create(int device, const char *kernel_dir, char *err,
    size_t errlen)`: uses the calling thread's current CUDA context (the
    caller's, e.g. the primary context the Python side makes current);
    creates one only if none is current. `kernel_dir` NULL means
    `$SLABITZ_KERNEL_DIR`, else the `kernels/` directory next to the
    loaded `libslabitz.so` (find it with `dladdr`).
  - Output sizing on the host, from the chunk headers only:
    `sbz_bshuf_output_bytes(host_src, chunks, n, &bytes)`.
  - `sbz_bshuf_decode(ctx, codec /*SBZ_ZSTD or SBZ_LZ4*/, elem_size,
    host_src, dev_src, chunks, n, dev_dst, dst_capacity, dev_status,
    stream)`. The host walk needs the compressed bytes on the host, which
    a streaming caller has anyway (it uploads them): `host_src` is read
    during the call only; `dev_src` holds the same bytes on the device (the
    caller's upload may be queued on `stream` just before). Returns
    `SBZ_OK` once the work is queued; host-walk errors are returned
    directly.
  - Plain zstd frames: the same pattern (`sbz_zstd_decode`), output frames
    back to back as `sbz_dctx_decompress_frames` does today.
  - Scratch is per context and reused (grows only). Stream ordering: the
    context records an event at the end of each call, and the next call's
    stream waits on it (`cuStreamWaitEvent`) before reusing scratch. For
    overlap the caller uses two contexts. The library's own small host to
    device uploads (block tables) go through a per-context pinned staging
    buffer (`cuMemHostAlloc`, CUDA 6.5), so the call does not wait for
    them (done in step 2a).
  - Pipelining guide for the caller (from the step 2a trace): put every
    input upload on ONE upload stream, record an event after each, and make
    the decode wait on it (`cuStreamWaitEvent`) before the decode stream's
    work. Uploads issued on two streams at once share PCIe and finish
    together, so the decodes coincide and PCIe then idles. Offer this as a
    helper in the Python package (step 4) and show it in `tests/run_api.py`.
  - One context is used by one thread at a time (documented).
- The `sbz_dctx_*` functions stay for the existing tests (they become thin
  synchronous wrappers over the new calls with internal buffers), so every
  existing test keeps passing unchanged.
- Kepler: only CUDA 6.5 driver calls; anything newer (primary-context
  calls, stream-ordered allocation) only through optional `dlsym` with a
  fallback, as today.
- Tests: a new `tests/run_api.py` (ctypes) that decodes both pac files and
  both LZ4 files through the new API with two contexts and two streams in
  alternation, compares with h5py, and checks the status arrays (corrupt
  chunks give their documented code). Run it on the L40S and, with
  `build/kernels`, on the TITAN Black.

## Step 4: Python package `slabitz`

- Layout: `python/slabitz/` (`__init__.py`, `_lib.py` for the ctypes
  binding of `include/slabitz.h`, `_backend_drv.py` copied from
  bslz4_to_sparse with its MIT notice and a line saying where it came from
  and at which commit).
- API (settled): `slabitz.Decoder(device=0)` with
  `decode(chunks, codec, dtype, shape, stream=None, out=None)`. `chunks`
  is a list of bytes (or one buffer plus offsets); the decoder uploads them
  through its pinned staging on `stream` and returns a device `ndarray`
  (from the vendored backend: `__cuda_array_interface__` and `__dlpack__`)
  of `shape`, plus a device status array. `decode_device(host_chunks,
  dev_bytes, ...)` for callers that upload themselves. No sync inside;
  `ndarray.get()` or the consumer's stream ordering does that.
- Tests: `tests/test_python.py`: round trip of both LZ4 and both zstd
  files, hand-off to numpy (`.get()`) and, when installed, to cupy and
  torch through `__dlpack__` (skipped with a notice otherwise).

## Step 5: wheel

- `pixi run wheel` builds the kernels (`build-fatbins`, which needs the
  sm_35 set from the kepler environment first), the library, and a wheel
  with `libslabitz.so` and `kernels/<sub>/<stem>.fatbin` as package data.
  Pure ctypes, no compiled Python extension: tag
  `py3-none-manylinux_2_28_x86_64`. The library must keep no RPATH.
- Test: install the wheel into a fresh venv under the workspace (not
  /tmp), run `tests/test_python.py` against it with `SLABITZ_KERNEL_DIR`
  unset.

## Non-goals

HDF5 integration, dictionaries, checksums on the GPU, compression, the
Vulkan backend, CUDA graphs.
