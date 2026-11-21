# DirectStorage zstdgpu headers (vendored)

Unmodified copies of the HLSL/C headers from Microsoft's DirectStorage GPU
zstd decoder, which the gpuzstd Slang kernels (`kernels/*.slang`) include.

- Upstream: https://github.com/microsoft/DirectStorage, directory `zstd/zstdgpu/`
- Commit:   303a7d239e27c615281d6304ed13f01dc35061ef (2026-11-12)
- License:  MIT, Copyright (c) Microsoft Corporation -- see `LICENSE`
  (the repository root LICENSE, copied verbatim).

The files are the blobs as committed upstream (LF line endings; upstream's
`.gitattributes` converts `*.h` to CRLF on checkout). Do not edit them here.
`tools/transform_ds_for_slang.py` copies them into the build directory and
applies the Slang/CUDA patches there.

Files: `zstdgpu_assert.h zstdgpu_shared_structs.h zstdgpu_structs.h
zstdgpu_shaders.h zstdgpu_lds.h zstdgpu_lds_hlsl.h zstdgpu_lds_decl_base.h
zstdgpu_lds_decl_size.h zstdgpu_lds_decl_undef.h srt_headers/zstdgpu_srt_structs.h`

To update: check out a newer upstream commit, copy the same files with
`git show <commit>:zstd/zstdgpu/<file>`, update the commit above, rebuild with
`tools/build_all.py` and rerun the tests. To build against an upstream
checkout without copying, pass `--in <DirectStorage>/zstd/zstdgpu` to
`tools/transform_ds_for_slang.py` or set `GPUZSTD_DS_DIR`.
