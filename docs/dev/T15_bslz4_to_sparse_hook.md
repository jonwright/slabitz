# T15: bslz4_to_sparse decoder hook (other repo)

Status: todo
Executor: Claude designs the hook; worker drafts the tests; Claude reviews
Depends on: T14
Repository: bslz4_to_sparse (use the most recent clone,
`../bslz4_to_sparse-dev`, or a fresh clone; branch `slabitz-decoder`).

## Goal

This is the "user friendly API" the user asked for: slabitz plugs into
bslz4_to_sparse's CUDA interface. `chunk2sparse` and `chunk2sparseCSC`
should accept bitshuffle-zstd chunks, and LZ4 chunks of any block size,
by delegating the decode to slabitz. Everything downstream of the decode
(sparse, CSC, integrate) stays as it is.

## Background (bslz4_to_sparse-dev, branch cuda-lz4-bounds, c37617f)

- **Public API** (`src/cuda/_gpu.py`). `chunk2sparse(mask, dtype, codec,
  ...)` is at about line 207. Its methods are:
  - `multi`, `from_buffer`, `__call__` and `coo`, which take host chunks;
  - `from_device(buf, offsets, lengths, cut)`;
  - `decompress` and `decompress_from_device`, which return
    `(n, ni, nj)` device arrays.

  `chunk2sparseCSC` adds `integrate`.
- **The seam** is `chunk2sparse._decompress(s0, n, dense, stride)` (about
  lines 422-439). It is the only place that decodes. It fills
  `dense + f*stride` with unshuffled frames on `self.stream` and sets
  `d_err` per chunk. The downstream kernels (`count_above`, `scan_*`,
  `collect_write`, `csr1d`, `permute_gather`) read only `dense` and
  `d_err`.
- **Setup before it.** `_unblock` (the block-header walk) and `_upload`
  are LZ4-specific. The `unblock` kernel flags the zstd magic as err 5
  (`bslz4_split.cu:61`). `_check` refuses non-LZ4 chunks. `BLOCK = 8192`
  is hard-coded (about line 72).
- **No registry.** There is no codec or backend registry. The only switch
  is `BSLZ4_CUDA_BACKEND` (cupy or drv).
- **What `src/cuda/__init__.py:7-11` expected:** zstd would be decoded on
  the CPU with the shuffle on the GPU.
- **Shared resources.** Both backends use the primary context, and every
  entry calls `cu.use_device`. Streams are private non-blocking ones:
  `self.stream.handle` (drv) or `.ptr` (cupy). Foreign device memory comes
  in through `__cuda_array_interface__` (`device_bytes`).

## Settled design

- **A `decoder=` argument on `chunk2sparse` and `chunk2sparseCSC`**,
  with values `"auto"` (the default), `"builtin"` or `"slabitz"`.
  - `auto` uses builtin for 8 kB-block LZ4, and slabitz for zstd or other
    LZ4 block sizes, if `import slabitz` works. Otherwise it raises the
    same error as today.
  - The environment variable `BSLZ4_DECODER` overrides the default.
- **On the slabitz path:**
  - `_unblock` is skipped, because slabitz walks the chunks itself;
  - `_decompress` calls `slabitz.Decoder.decode_device(slab, d_off,
    lengths, dense, codec=..., geom=..., stride=stride, err=d_err,
    stream=self.stream)`;
  - the geometry is read from the first chunk's header and the dataset
    dtype;
  - the `Decoder` is created once per `chunk2sparse` object, on the same
    device;
  - `d_err` keeps its meaning (0 ok, nonzero skipped).
- **No change to the builtin LZ4 path or its results.**

## Files that may change (in bslz4_to_sparse)

`src/cuda/_gpu.py`, `src/cuda/__init__.py` (docstring), `test/` (new test
file), `README.md` (decoder option), `pyproject.toml` (optional extra
`slabitz`).

## Acceptance

1. **LZ4 identical through both decoders.** `pytest
   test/test_slabitz_decoder.py` runs `chunk2sparse` over N frames of an
   8 kB-block LZ4 dataset (the silicon example) with
   `decoder="builtin"` and with `decoder="slabitz"`. `values` and
   `indices` are `np.array_equal` for every frame; print N and the total
   nonzeros, both nonzero.
2. **zstd equal to the CPU path.** On a bitshuffle-zstd dataset (CeO2 or
   pac silicon zstd), `decoder="auto"` gives results equal to the CPU path
   (`bslz4_to_sparse.chunk2sparse` CPU, or h5py decode plus a numpy
   threshold).
3. **Integration path.** `chunk2sparseCSC.integrate` gives identical
   results with both decoders on LZ4.
4. **Corrupt chunk.** One corrupt chunk in a zstd batch is skipped
   (`d_err` nonzero) and the other frames are correct.
5. **Throughput.** Frames per second for each decoder on the L40S
   (Slurm), at 16 frames per call, for silicon LZ4. Report it; slabitz
   should be within 10% of builtin, or the report explains why.

## Reviewer checks

- Run acceptance 1 and 2 yourself.
- Uninstall slabitz: zstd must raise the old clear error, and LZ4 must
  still work through builtin.
