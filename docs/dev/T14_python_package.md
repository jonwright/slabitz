# T14: Python package `slabitz`

Status: todo
Executor: worker (DeepSeek), reviewed by Claude
Depends on: T13
Branch: `t14-python`

## Goal

Give slabitz a Python package. It should work two ways:

- **Standalone:** `decode` returns a device array.
- **As a backend:** `decode_device` takes caller-owned arrays and streams,
  owns nothing, and allocates nothing in the hot path. bslz4_to_sparse
  (T15) calls this one.

This is P4 step 4 of `docs/archive/plan_p4_api.md`, rewritten to meet the
bslz4_to_sparse seam.

## Background

- **Today.** A ctypes loader exists in `slabitz/loader.py`, alongside
  `slabitz/zstd_frame.py`. The tests call the library through ctypes
  directly.
- **The driver backend to vendor.** bslz4_to_sparse's driver backend is
  `../bslz4_to_sparse-dev/src/cuda/_backend_drv.py` (MIT, same author). It
  provides:
  - a retained primary context;
  - non-blocking streams, kept per thread (`_tls.stream`);
  - a private memory pool (`cuMemPoolCreate`, with a fallback to
    `cuMemAlloc`);
  - pinned memory (`cuMemHostAlloc`);
  - `ndarray`, with `__cuda_array_interface__` v3 and `__dlpack__`;
  - `device_bytes`, which wraps foreign arrays.
- **Settled on 2026-11-20:** ctypes, reuse `_backend_drv.py`, no compiled
  extension.

## Read first

`AGENTS.md`; `include/slabitz.h` and `tests/run_api.py` (from T13);
`slabitz/loader.py`; `../bslz4_to_sparse-dev/src/cuda/_backend_drv.py` and
`_gpu.py` (how a class launches on its stream and hands out arrays).

## Settled design

- **Layout.**
  - Keep the package at `slabitz/` (it already exists).
  - `slabitz/_lib.py` holds the ctypes binding of `include/slabitz.h`. It
    replaces the internals of `loader.py`, which stays as a compatibility
    shim for the tests.
  - `slabitz/_backend_drv.py` is copied from bslz4_to_sparse with its MIT
    notice and a line giving the source commit.
- **`slabitz.Decoder(device=0, kernel_dir=None)` has two calls.**
  - `decode_device(src, offsets, lengths, out, *, codec, geom, stride=None,
    err=None, stream=None)` is the backend call. `src`, `out` and `err`
    are any objects with `__cuda_array_interface__`. `stream` is an int
    handle, a driver-backend `Stream`, or a cupy stream (its `.ptr`). It
    performs no sync and no allocation after the first call with a given
    size.
  - `decode(chunks, codec, dtype, shape, stream=None, out=None)` is the
    standalone call. It takes a list of bytes, or one buffer plus offsets.
    It uploads through pinned staging on `stream` and returns a device
    `ndarray` of `(n, *shape)` and a device status array.
- **Other functions:**
  - `slabitz.codec_of_chunk(buf)`: LZ4, ZSTD, or raise, using the magic at
    bytes 16:20, as bslz4_to_sparse does;
  - `slabitz.geom_from_chunk(buf, dtype)`.
- **Errors.** `wait()` raises `slabitz.CorruptChunk` (with the chunk
  indices) or `slabitz.Unsupported`.

## Files that may change

`slabitz/*.py`, `tests/test_python.py` (new), `pixi.toml` (pytest task
membership), `README.md` (Python section), `docs/dev/reports/T14.md`.

## Acceptance

1. `pixi run pytest tests/test_python.py` passes:
   - round trip of the pac zstd and LZ4 fixtures and the real files,
     through `decode` and `decode_device`;
   - hand-off to numpy with `.get()`;
   - when installed, hand-off to cupy and torch through `__dlpack__` /
     `__cuda_array_interface__`. Without them, the test skips with a
     printed notice, and the report says which ones ran.
2. **No allocation in the hot path.** A test calls `decode_device` 100
   times with the same sizes. The library's allocation counter, a debug
   getter added in T13 or here, does not grow after the first call.
3. **No sync.** A test calls `decode_device` on a cupy stream and checks,
   with `cupy.cuda.Stream.done` just after the call, that it returned
   before the GPU finished (16 frames per call).

## Reviewer checks

- Run `tests/test_python.py` in a fresh pixi environment with cupy
  installed. Check that the cupy path really ran: there must be no skip
  message.
- `git diff` of `_backend_drv.py` against its source commit shows only the
  header lines.
