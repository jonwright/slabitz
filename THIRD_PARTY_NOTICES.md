# Third-party notices

slabitz contains or is derived from the following third-party code.

## Microsoft DirectStorage GPU zstd decoder -- MIT

https://github.com/microsoft/DirectStorage (`zstd/`), Copyright (c) Microsoft
Corporation, MIT License. Full text: `third_party/directstorage_zstdgpu/LICENSE`.

- `third_party/directstorage_zstdgpu/`: unmodified headers (see its README).
- Derived from it (transcriptions/ports; the MIT notice applies):
  `kernels/*.slang`, `kernels/cpu_ref_*.py`, `src/host_walk.c`,
  `src/host_walk.h`, `tools/slang_compat.h`, `tools/transform_ds_for_slang.py`.

## zstd educational decoder -- BSD-3-Clause (or GPLv2)

https://github.com/facebook/zstd (`doc/educational_decoder/`), Copyright (c)
Meta Platforms, Inc. and affiliates. Used under BSD-3-Clause. Full text:
`third_party/zstd_educational_decoder/LICENSE`. The vendored copy is
Microsoft's DirectStorage variant (see its README). It is used only by the
test oracle `tools/edu_oracle.py`, not by the library.

## NVIDIA nvCOMP LZ4 decompressor -- BSD-3-Clause

https://github.com/NVIDIA/nvcomp (`src/lowlevel/LZ4CompressionKernels.cu`),
Copyright (c) 2020, NVIDIA CORPORATION. Full text is at the top of
`kernels/Lz4DecodeBlocks.slang`.

- Derived from it, via the bslz4_to_sparse port
  (`src/cuda/nvcomp_lz4.cuh`, `decompressStream`): `kernels/Lz4DecodeBlocks.slang`.

