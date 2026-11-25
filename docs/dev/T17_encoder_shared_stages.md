# T17: Encoder shared stages: GPU bitshuffle and plane-aware parse

Status: todo
Executor: Claude (kernel design and buffer contracts); the reference oracle
may be worker-drafted
Depends on: T03 (gate, for timing discipline). Independent of T06-T16.
Branch: `t17-encoder-stages`

## Goal

This is the first step of GPU compression. Write the stages that the LZ4
and zstd encoders share:

- the bitshuffle transpose;
- pac's plane-aware match finding, producing a sequence list per block in
  device memory.

Each stage is written once, as device functions in `kernels/lib/`. It is
proven equal to pac-zstd's CPU encoder through a reference oracle.

## Background: pac-zstd's encoder

Repository `../pac-zstd`, branch `gemini-minimal-p9fix`, HEAD 162af8b.

- **Pipeline** (`src/pac.c`, `pac_encode_block`, around line 752):
  - transpose, then parse (`pac_build`, around lines 567-632), then emit;
  - the paths split in `pac_encode_block`: LZ4 goes to `pac_emit_lz4`
    (lines 644-689); zstd goes to probe, raw, RLE, then
    `ZSTD_compressSequences`.
- **Transpose.**
  - It uses kcb `bitshuf_encode_block`, or the user's function from
    `pac_set_transpose`.
  - In "narrow" mode (`pac_set_narrow`, lines 861-891, on by default only
    on x86), pac ORs all the words of a block. If the upper bytes are all
    zero, it transposes only the low bytes and zero-fills the rest.
  - Tail: fewer than 8 leftover elements are stored as they are.
- **Parse** (`src/pac_parse.inl:153-202`):
  - candidates are only offset 1 and offset = plane (the plane before);
  - it builds 64-byte equality masks and ANDs shifted masks to find match
    starts of at least 4 bytes;
  - the longer candidate wins, extended by `match_len`.
- **`pac_build`:**
  - dense planes (at least 1024 bytes and more than 1/20 literals) get
    their own zstd block, and so their own Huffman table;
  - other planes are packed together up to 128 KiB;
  - adjacent same-offset continuations are merged (lines 611-618);
  - after the last nonzero byte, the rest is one offset-1 run
    (lines 580-586);
  - for LZ4, offset = plane is disabled when the plane is larger than
    65535 (line 575).
- **No public API exposes the sequence list.** `pac_build` is static.
  `pac_compress_chunk` and `pac_set_narrow` are public.
- **Block sizes.** `PAC_ZSTD_BLOCK` is 253952 and `PAC_LZ4_BLOCK` is 57344
  (`src/pac.h:29-38`). The block size is a nonzero multiple of 8 x elem,
  at most 2^27.
- **The slabitz decode side.** The unshuffle kernels are the inverse of
  the transpose (`kernels/BitUnshuffle*.slang`, `kernels/lib/`), and
  `kernels/lib/` has the tiling pattern.
- **The fusion rule** (`AGENTS.md`): a block plane is block / (8 x elem)
  bytes, for example 15.5 kB for 248 kB u16. A whole block does not fit
  in 16 KB of groupshared, so transpose and parse stay separate kernels
  at first.

## Settled design

- **Reference oracle: `tools/pac_ref.py`.** It is pure Python and numpy,
  slow but exact, and ports the following:
  - the transpose (narrow and normal);
  - the generic (non-SIMD) parse;
  - `pac_build`, for both the LZ4 and the zstd grouping;
  - `pac_emit_lz4`.

  It must reproduce the output of `pac_compress_chunk(PAC_LZ4)` byte for
  byte, through `libpac.so` (as in `bench/pac_matrix.py`). This proves
  that the oracle's sequence lists are pac's. For zstd it emits the
  sequence list and block grouping only.
- **Device data contract.** These structures go in `kernels/lib/encode.slangh`
  and are documented in the header comment:
  - **Input:** frames in device memory, `n` frames of `frame_bytes`, with
    element size and block size.
  - **Transposed buffer:** the same size as the input. This is the same
    layout the decoder's unshuffle consumes.
  - **Per-block sequence buffer:** fixed capacity per block (the worst case
    is block bytes / 4 matches plus 1), entries `{lit_len, match_len,
    offset}` as u32, and a per-block count.
  - **Per-block group/plane metadata** for the zstd grouping (dense flag,
    block boundaries).
- **Kernels.** `kernels/BitShuffleEncode.slang` (transpose) and
  `kernels/PacParse.slang` (parse plus build, one workgroup per block).
  Lanes scan planes with ballots for match starts, and a serial merge
  per group. Wave loops must keep a uniform trip count (`AGENTS.md` rule;
  the `ZSTDGPU_FOR_WORK_ITEMS` pattern).
- **Host API (internal for now):** `sbz_ectx_*` in a new
  `src/sbz_encode.c`. The public encode call comes with T18.

## Files that may change

`tools/pac_ref.py`, `kernels/lib/encode.slangh`,
`kernels/BitShuffleEncode.slang`, `kernels/PacParse.slang`,
`tools/build_all.py` (new entries), `src/sbz_kernels_gen.h`,
`src/sbz_encode.c`, `src/sbz_encode.h`, `Makefile`,
`tests/run_encode_stages.py` (new), `pixi.toml`, `docs/dev/reports/T17.md`.

## Acceptance

1. **The oracle reproduces pac's LZ4 output.** `python3 tools/pac_ref.py
   --selftest` produces output byte-identical to
   `pac_compress_chunk(PAC_LZ4)` for:
   - the 12-case matrix (silicon, thaumatin, CeO2 x u16/u32 x
     mask0/maskmax-1);
   - random u8 to u64 data with tails;
   - narrow on and off.

   It prints the cases and bytes compared.
2. **GPU transpose.** `python3 tests/run_encode_stages.py` checks that the
   GPU transpose is byte-identical to `bitshuffle.bitshuffle` (the CPU
   package) for u8 to u64, with tails and narrow mode. It also checks that
   the decoder's unshuffle of it gives back the input.
3. **GPU sequence lists.** The GPU sequence lists and block grouping are
   identical to `tools/pac_ref.py` for every block of the matrix. The test
   prints the number of blocks and sequences compared.
4. **Throughput.** Transpose plus parse on the L40S (Slurm), in GB/s of
   input, for silicon u16 and CeO2 u32 at 16 frames. Report it, and
   compare with pac on the CPU, single thread and all threads.
5. `pixi run test` and `pixi run test-kernels` pass. The new kernels
   compile for SPIR-V and Metal (`slangc -target spirv|metal`), or the
   report lists which intrinsic blocks it.

## Reviewer checks

- Change `match_len`'s minimum from 4 to 5 in the GPU kernel (temporary):
  acceptance 3 must fail.
- Run `tools/pac_ref.py --selftest` against a fresh `libpac.so` build.
