# T19: zstd encoder (bitshuffle-zstd chunks on the GPU)

Status: todo
Executor: Claude (entropy coders on the GPU; design-heavy)
Depends on: T18 (encode API and compaction), T07 (version corpus tools),
T08 (pac-zstd's floor result: the list of features that are unsafe for old
decoders)
Branch: `t19-zstd-encoder`

## Goal

Write filter-32008 bitshuffle-zstd chunks on the GPU with the same parse as
pac. The encoder needs only what pac-zstd emits, using a fixed, GPU-friendly
subset of entropy modes. The chunks must decode on every decoder from
bitshuffle 0.4.0's zstd (`18d02cbf`, `v1.4.7+`) to 1.6.0. The compression
ratio must be close to pac's.

## Background: what pac emits (`../pac-zstd`, HEAD 162af8b)

- **Frames.** One frame per bitshuffle block, behind a 4-byte big-endian
  length. The header (`pac_write_zstd_fh`, `src/pac.c:691-708`) is:
  magic, Single_Segment=1, FCS present (1, 2 or 4 bytes), no dictionary,
  no checksum, no window descriptor.
- **Block types:**
  - raw blocks for leading planes that the B128 probe finds
    incompressible (`pac_probe_plane`, lines 476-557; the threshold is
    2.0%);
  - RLE blocks for an all-constant block (lines 710-727);
  - every other block is compressed by libzstd 1.5.7's
    `ZSTD_compressSequences` at level 1, with explicit block delimiters
    and repcode resolution. **libzstd picks every literal and table mode**,
    so byte identity with pac is not achievable without reimplementing
    libzstd's heuristics. The user chose not to require it (2026-11-25).
- **The magicless splice trick** (lines 760-808) is a CPU detail. Do not
  copy it: some decoders up to 1.5.5 mishandle magicless frames, and on
  the GPU the raw prefix and the compressed tail can be written directly.
- **Rules for old decoders.** Never emit:
  - a compressed block of exactly 131072 bytes (libzstd 1.5.2 and older);
  - a block with 0 literals and 0 sequences (1.5.2 and older);
  - a frame whose first block is a 131072-byte RLE block (CLI 1.4.3 and
    older);
  - anything T08 adds to this list.
- **pac's validator.** pac's `tests/check_rfc.c` is an RFC 8878 validator
  plus a decode with the educational decoder. Use it as an oracle.
- **New in zstd 1.6.0:** `ZSTD_compressSequencesAndLiterals`. That is
  useful as a CPU reference, but it is not on the floor.

## Settled design

- **Subset emitted.**
  - Frame: pac's header.
  - Block types: raw, RLE and compressed.
  - Literals:
    - raw, when Huffman does not pay;
    - 4-stream Huffman with a new table per block, using the 4-bit direct
      weight form or FSE-compressed weights, whichever is smaller;
    - 1-stream Huffman for literal sections under 1 KiB;
    - no treeless (repeat) literals in phase 1.
  - Sequences: for each of LL, ML and OF, predefined or FSE-compressed,
    chosen by estimated cost. RLE mode where a single symbol occurs. No
    repeat mode in phase 1.
  - Repcodes: none emitted explicitly. pac relies on repcode resolution
    inside libzstd, so measure whether emitting repcode 1 for repeated
    offsets matters for ratio, and add it if it does.
- **Kernels.**
  - Per block: histogram (literals, LL/ML/OF codes); Huffman table build
    (a length-limited code, max 11 bits); FSE table build (normalised
    counts, accuracy log within the decoder's caps LL 9, OF 8, ML 9);
    bitstream writers (the backward FSE stream; Huffman streams with the
    jump table).
  - One workgroup per block. Each stage is a function in `kernels/lib/`,
    shared with tests through small entry points.
- **Probe.** Port pac's B128 probe from the T17 oracle.
- **Size per block.** Use the T18 compaction.

## Files that may change

New `kernels/Zstd*Encode*.slang` and `kernels/lib/zstd_encode*.slangh`,
`src/sbz_encode.c`, the public header, `src/sbz_kernels_gen.h`,
`tools/build_all.py`, `tests/run_zstd_encode.py` (new),
`tools/pac_ref.py` (probe), `slabitz/*.py`, `pixi.toml`,
`docs/dev/reports/T19.md`.

## Acceptance

1. **Decodes everywhere.** `python3 tests/run_zstd_encode.py` encodes the
   12-case matrix plus random u8 to u64 data with tails, at block sizes of
   8 kB and 248 kB. Every chunk decodes byte-exact with:
   - slabitz on the GPU;
   - hdf5plugin;
   - libzstd at `18d02cbf`, v1.4.7, v1.5.2 and v1.6.0 (built by T07's
     `tools/gen_compat_corpus.py`);
   - pac's `check_rfc` validator, which must report 0 violations.

   It prints the chunks and bytes compared.
2. **Ratio.** It prints the compressed size against pac-zstd per case and
   the total. Target: at most 5% larger than pac in total over the matrix,
   and at most 10% larger in any case. If the target is missed, the report
   gives the per-mode breakdown and the proposed next mode (treeless,
   repeat tables, repcodes). The user then decides whether to accept.
3. **Erratum guards.** Hand-built inputs that would produce an exactly
   131072-byte compressed block, a 0/0 block, or a first 131072-byte RLE
   block are encoded differently, and the validator confirms it.
4. **Throughput.** On the L40S (Slurm), report GB/s for silicon u16 and
   CeO2 u32 at 16 frames. Compare it with pac CPU, single thread and all
   cores, and with nvCOMP zstd compress if it is available.
5. `pixi run test`, `pixi run test-kernels` and `bench/gate.py --compare`
   pass. The decode cells must be unchanged.

## Reviewer checks

- Run acceptance 1 with libzstd `18d02cbf` yourself, on 3 random cases.
- Break the guard in acceptance 3 (temporary): `check_rfc` must flag the
  exactly-131072 block.
