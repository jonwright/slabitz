# zstd educational decoder (vendored, DirectStorage variant)

The reference decoder used by `tools/edu_oracle.py` to dump intermediate
stages (Huffman weights, literals, sequences, offsets) as ground truth.

- Original: facebook/zstd `doc/educational_decoder/` at
  f8745da6ff1ad1e7bab384bd1f9d742439278e99 (the commit Microsoft's
  NOTICES.txt cites). Copyright (c) Meta Platforms, Inc. and affiliates.
  Dual-licensed BSD-3-Clause or GPLv2; we use it under BSD-3-Clause, see
  `LICENSE` (copied verbatim from that zstd commit).
- This copy: as shipped in microsoft/DirectStorage
  `zstd/ThirdParty/zstd_educational_decoder/` at
  303a7d239e27c615281d6304ed13f01dc35061ef. Microsoft added
  `zstdgpu_reference_store.h` hooks, all inside
  `#ifdef ZSTDGPU_REFERENCE_STORE_H` / `#if !defined(DISABLE_ZSTDGPU_REFERENCE_STORE)`.
  The oracle builds with `-DDISABLE_ZSTDGPU_REFERENCE_STORE`, so the hooks
  compile out; its own text-patch anchors are written against this copy.
  Microsoft's changes are under the DirectStorage MIT license
  (`../directstorage_zstdgpu/LICENSE`).

Files are unmodified upstream blobs. Override the location with `DS_EDU_DIR`.
