# T09: Decoder: matching rejects, error codes, bounds

Status: todo
Executor: Claude (C runtime and DS transform; not for the worker)
Depends on: T07 (corpus and expectations to flip), T03 (gate)
Branch: `t09-rejects-bounds`

## Goal

Every zstd input the decoder cannot handle must be rejected by the host
walk, before any GPU work, with a specific error code and message. Every
frame must be kept inside its own output range, so that a frame that
decodes larger than it declared cannot overwrite its neighbour.

## Background (audit 2026-11-25)

Refs: `hw` = `src/host_walk.c`, `ds` =
`third_party/directstorage_zstdgpu/zstdgpu_shaders.h`, `ch` =
`src/sbz_chain.c`.

- **Host walk and GPU parser reject different things.**
  - The GPU header parser rejects:
    - a dictionary-ID flag that is not zero, even with ID 0 (ds:356, 414);
    - a missing FCS (ds:394, 422);
    - a window of 2^29 or more (ds:418-420; the bit base of 29 is at
      `zstdgpu_structs.h:818`, because offsets share a 32-bit word with a
      repeat-encode bit).
  - The host walk rejects only a dictionary ID that is not zero (hw:192),
    and has no window check.
  - Exact calls therefore fail at the block-count cross-check (ch:703-710)
    with "block counts GPU ... != host". Async calls detect it at the wait
    and replay (ch:1940-1950), after stage 1 and 2 kernels have already run
    on block refs the GPU never wrote.
- **Not checked:**
  - decoded size against FCS. Destination offsets are the frame base plus
    the block prefix, with no bound (`kernels/ComputeDestBlockOffsets.slang:42`),
    so a frame decoding larger than its FCS overwrites the next frame or
    runs out of bounds;
  - Block_Maximum_Size = min(window, 128 KiB): only 128 KiB is checked
    (hw:238). zstd 1.6.0 clarified this rule in the spec;
  - literals above 128 KiB per block: `ZSTDGPU_BREAK` (ds:1199-1201) is
    an assert that does nothing in these builds (`zstdgpu_assert.h:46-51`).
- **Output caps do not match.** The host allows 4 GiB of output per call
  (ch:921). The GPU accepts only 0x3FFFFFFF (1 GiB) per request
  (`zstdgpu_shared_structs.h:58-64`); beyond that, frames get status
  `RequestTooLarge` and the call returns CORRUPT.
- **Error codes** are in `src/sbz_backend.h:37-42`: BACKEND, NOMEM, ARGS,
  KERNEL, CORRUPT, LIMIT.

## Read first

`AGENTS.md` (all of the rules, especially editing DS only through
`tools/transform_ds_for_slang.py`); `src/host_walk.c`; `src/sbz_chain.c`
(`host_prepare`, `stage0`, `read_counters`, the wait/replay path);
`kernels/ComputeDestBlockOffsets.slang`; the DS frame-header parser
(ds:350-460); `tests/run_host_walk.py` (named corrupt cases, fuzz);
`tests/run_compat.py` and `tests/run_features.py` (from T06/T07).

## Settled design

- **New code `SBZ_ERR_UNSUPPORTED = -7`.** It means valid zstd that this
  decoder does not handle. Document it in `sbz_backend.h` and
  `sbz_chain.h`.
- **The host walk applies all frame-level rejects first**, with distinct
  messages:
  - a dictionary-ID field present (any value, including 0) gives
    UNSUPPORTED, "frame N: dictionary";
  - a window of 2^29 bytes or more gives UNSUPPORTED, "frame N: window
    2^k exceeds 2^29".

    Decision (2026-11-25): reject rather than widen the DS offset packing.
    A window that large means a frame over 512 MiB, far beyond the
    detector use case. Record it as a known limit in `AGENTS.md`.
  - missing FCS gives UNSUPPORTED, "frame N: no content size". T10 turns
    this into support.

  The GPU parser keeps its checks as a second line of defence. The
  cross-check message is extended to name the GPU frame status
  (`MissingContentSize`, `WindowTooLarge` and so on) when the counts
  disagree.
- **Block_Maximum_Size.** The host walk checks that each block's size is
  at most min(window, 128 KiB), with window = FCS for single-segment
  frames, and returns CORRUPT otherwise. For compressed blocks it checks
  the compressed size; for RLE blocks, the regenerated size.
- **Per-frame output bound on the GPU.** Every kernel that writes decoded
  bytes (MemsetMemcpy, ExecuteSequences) clamps writes to its frame's
  `[base, base + FCS)` and sets a per-frame overflow status bit instead of
  writing past the end. The frame's end offset comes from the frame refs
  the host walk already has. The wait then reports CORRUPT, "frame N
  decodes past its content size". The change goes in through the
  transform or the Slang entries. Performance must stay within the T03
  gate: a per-sequence compare against a bound in a register should be
  negligible; measure it.
- **Output cap.** Either lift the GPU's 1 GiB request limit, or make the
  host return LIMIT, "call output exceeds 1 GiB", before any GPU work.
  Choose whichever is simpler and record the choice. Callers split
  batches anyway.
- **Tests.**
  - Named host-walk cases in `tests/run_host_walk.py`: dictionary ID 0
    present, window 2^29, window 2^31, no FCS, a block larger than the
    window, and an RLE block larger than the window.
  - A GPU case in a new `tests/run_frame_bounds.py`: a hand-built frame
    whose FCS is smaller than its decoded size, placed between two good
    frames. Pad the output allocation with canary bytes. Check the CORRUPT
    code, that the neighbours are intact and that the canaries are
    untouched. Run it sync and async.
  - Update the T06 and T07 expectations from `reject:SBZ_ERR_CORRUPT` to
    `reject:SBZ_ERR_UNSUPPORTED` where it applies.
  - **T06's one xfail.** `hand-dict-id-1` in `tests/run_features.py`
    (reproducer `tests/data/features_bugs/did_field.zst`, 150 bytes) is a
    frame with a `Dictionary_ID` field of value 0. libzstd decodes it, the
    host walk accepts it today, and the GPU parser rejects it at the
    block-count cross-check ("block counts GPU raw/rle/cmp 0/0/0 != host
    1/0/0"). Under this design its expectation becomes
    `reject:SBZ_ERR_UNSUPPORTED` with "frame 0: dictionary"; then remove
    the reproducer and its README line, or keep the frame as a host-walk
    case.
  - **Output cap reproducer** (T04 question 3): the gate's CeO2 zstd2
    twins at 64 frames per call (`bench/gate.py --only twin/ceo2_zstd2
    --frames 64`) need 1,144,580,096 output bytes and return CORRUPT
    today ("frame 0: status 0xa07a0006"). After this task they must
    return LIMIT before any GPU work, or decode if the cap is lifted.

## Files that may change

`src/host_walk.c`, `src/sbz_chain.c`, `src/sbz_chain.h`, `src/sbz_backend.h`,
`tools/transform_ds_for_slang.py`, the affected `kernels/*.slang`,
`src/sbz_kernels_gen.h` (regenerated), `tests/run_host_walk.py`,
`tests/fuzz_host_walk.c` (seeds), `tests/run_frame_bounds.py`,
`tests/run_features.py` and `tests/run_compat.py` (expectations),
`tests/data/features_bugs/*`, `pixi.toml`,
`AGENTS.md`, `docs/dev/reports/T09.md`.

## Acceptance

1. `python3 tests/run_host_walk.py` passes, with the 6 new named cases
   each returning the stated code and message. The fuzz run (200k mutants)
   is clean.
2. `python3 tests/run_frame_bounds.py` passes, sync and async:
   - the code is CORRUPT;
   - the neighbour frames are byte-exact;
   - 0 canary bytes are changed.
3. `python3 tests/run_compat.py --full ...` (T07): all former
   `reject:SBZ_ERR_CORRUPT` cells now report UNSUPPORTED with the matching
   message, and nothing that decoded before fails.
4. `bench/gate.py --compare`: no cell regresses beyond the gate
   thresholds.
5. `sbz test` and `sbz test-kernels` pass. `build/sm_35` and
   `build/kernels` are rebuilt and the T05 suite is rerun on the L40S with
   `build/kernels`.

## Reviewer checks

- Remove the new clamp in ExecuteSequences (temporary transform edit)
  and rebuild: `run_frame_bounds.py` must report changed canary bytes or a
  damaged neighbour.
- Remove the window check from the host walk: the window-2^29 named case
  must fail.
