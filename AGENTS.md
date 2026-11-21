# AGENTS.md: slabitz status and next steps

GPU decoders for bitshuffle-zstd and bitshuffle-LZ4 HDF5 chunks (filter 32008,
e.g. pac output). Microsoft DirectStorage (DS) GPU zstd shaders (HLSL),
transformed and compiled with Slang to CUDA, driven by a C host library
through the CUDA driver API; the LZ4 path is a Slang port of the nvCOMP
decoder used by bslz4_to_sparse. History and lessons: `docs/history.md`.
Build: `README.md`.

## Build and test

The checkout is git-only. Builds, pixi, caches and temp files live in an
external workspace, normally reached through a gitignored symlink:

```
./              this checkout (no build, no .pixi)
./workspace     -> external pixi workspace (large disk; gitignored)
```

If `workspace/` is absent, create it with `tools/setup_workspace.sh` on a
large filesystem and link it; do not search the filesystem for test data.
Treat `workspace/` as read-only: build outputs go to `build/` or a scratch
directory, never into the example data. `workspace/env.sh` sets
`SLABITZ_EXAMPLES` to the read-only example data.

```bash
. workspace/env.sh && cd workspace/slabitz   # build/test checkout
CUDA_ARCH=sm_89 pixi run test                # kernels, lib, 11 test scripts
pixi run python3 tests/<name>.py             # any single test; all run from any cwd
```

Run every GPU test under `timeout`: a wave-divergence bug shows up as a hang.

## Status (2026-11-21)

| Area | State |
|---|---|
| All kernels on the decode path (22 Slang entries, incl. the 2 bit-unshuffle kernels, the 2 LZ4 kernels and the device LZ4 chunk walk; shared stage code in `kernels/lib/`) | ported, each validated vs CPU ref / oracle |
| C chain: `sbz_dctx_decompress` (concatenated frames) | byte-exact vs libzstd, 11 cases |
| `sbz_dctx_decompress_frames` (explicit frame list) | byte-exact, real hdf5plugin chunks |
| `sbz_dctx_bshuf_decompress` (bitshuffle-zstd chunks) | matches hdf5plugin, u8-u64, 8k/248k/256k blocks, tails, corrupt input -> error |
| `sbz_dctx_bslz4_decompress` (bitshuffle-LZ4 chunks, `kernels/Lz4DecodeBlocks.slang`, fused `kernels/Lz4DecodeUnshuffle.slang` for full blocks of at most 8 kB) | matches hdf5plugin, u8-u64, 8k-256k blocks, tails, corrupt input -> error, fused == unfused; L40S and TITAN Black. Phases 1-3 of `docs/plan_lz4_slang.md` done; plane-aware match copy (runs as fills, plane repeats without a modulo; pac's idea, 3a94af2). L40S, 16 frames/call, device us/frame: silicon 20.4 vs reference 19.6, CeO2 53 vs 67; wall faster than the reference on both (silicon 0.54 ms per 16-frame call, 263 GB/s). `SLABITZ_LZ4_FUSED=0` forces the unfused path |
| Real pac files (silicon u16, CeO2 u32), 16 frames/call | L40S 41 / 6.5 GB/s; TITAN Black 10 / 2.7 GB/s (`bench/old_cards.py`) |
| Volta+ (sm_70 and newer) | works (tested on sm_89) |
| Pre-Volta (sm_35 to sm_61) | works on the sm_35 TITAN Black: check_bshuf 17/17, all per-stage and chain tests; sm_50/sm_61 not yet run |
| Portable kernels (`pixi run -e kepler build-sm35 && pixi run build-fatbins` -> `build/kernels/`) | one fatbin per kernel: sm_35 (NVRTC 11.8), sm_50..sm_90 (nvcc 12.6), compute_90 PTX; 22 files, 5.1 MB. `tests/run_suite_kernel_dir.sh build/kernels` passes on the L40S (same speed as local cubins) and on the TITAN Black under driver 340 |
| Other Slang targets (compile only, slangc 2026.22) | Last survey at 21 kernels: SPIR-V and HLSL 19/19; Metal 13/19: `WavePrefixCountBits` has no Metal lowering (4 DS kernels: PrefixSum, ComputePrefixSum, ParseCompressedBlocks, InitFseTable) and Metal lacks `GetDimensions` (both unshuffle kernels; pass byte counts as scalars instead). The 2 LZ4 kernels (`Lz4DecodeBlocks`, `Lz4DecodeUnshuffle`) build for SPIR-V and Metal. `BshufWalkChunks` was added later: re-run the survey |
| pac-written zstd (a local clone of plane_aware_compressors) | decodes byte-exact (pac_repack_chunk of the LZ4 test files, in `$SLABITZ_WS/tmp/pac`). ExecuteSequences fills offset-1 runs (e870151): pac silicon 2155 -> 837 us per 16-frame call, pac CeO2 2840 -> 1568; stock zstd unchanged. pac silicon 1.69 ms per 16-frame call, stock zstd 2.8 |
| `src/host_walk.c` on corrupt input | bounds-checked: any input -> `SBZ_OK` / `SBZ_ERR_CORRUPT` / `SBZ_ERR_LIMIT` (15 named corrupt cases; ASan/UBSan fuzz, `make build/fuzz_host_walk`, 200k mutants in the test). Compressed-block contents are still trusted by the GPU kernels |
| Python wrapper, packaging, pytest suite | not started (tests are standalone scripts; a ctypes loader is in `slabitz/`) |

Output lives on the device (`UnCompressedFramesData` / `BshufOutput`);
`sbz_dctx_read_buffer` copies a named buffer back (also the debug hook for
diffing one stage against `kernels/cpu_ref_*.py`).

## Rules that keep it correct

- Edit DS code only through `tools/transform_ds_for_slang.py` (patches must
  match exactly once) or the Slang entries. Never edit generated `.cu`.
  `third_party/` is vendored upstream code: do not modify.
- One cubin per kernel, each its own CUDA module. A merged module is silently
  wrong (include-guarded DS header compiles once under the first kernel's
  macros).
- Any loop containing wave ops must have the same trip count on every lane
  (pattern: `ZSTDGPU_FOR_WORK_ITEMS` in the transform). Otherwise the CUDA
  warp deadlocks.
- Byte/halfword DS buffers are real `uint8_t`/`uint16_t` buffers (no masked
  read-modify-write: it races). FseProbs is int32-element.
- Entry points take `uint2 groupId2 : SV_GroupID` through
  `zstdgpu_ConvertTo32BitGroupId`; a scalar group id breaks grid.x > 1.
- After changing any kernel source (entries, `kernels/lib/`, the transform,
  the Slang preludes), rerun `tools/build_all.py` and commit the regenerated
  `src/sbz_kernels_gen.h`: it holds the launch-checking table and
  `SBZ_KERNELS_ID`, a hash of the kernel sources. Every kernel directory
  carries a `sbz_kernels.id`; the library refuses a directory with another ID
  (a stale kernel set once loaded silently and crashed on a changed argument
  list). Rebuild `build/sm_35` and `build/kernels` after kernel changes.
- Host C uses only CUDA 6.5 driver API calls, `libcuda` via dlopen (Kepler).
- Build only through pixi (or the Makefile with PATCHELF): the libraries must
  have no RPATH. A conda compiler bakes one in, and a cvmfs RPATH overloaded
  cvmfs on crunch (every dlopen of libcuda searched it).
- ASan on this node: the pixi gcc 13.4 runtime sometimes cannot start under the
  Linux 6.8 kernel's address randomisation (loops on "DEADLYSIGNAL"); run
  sanitizer binaries under `setarch $(uname -m) -R` (run_host_walk.py does).
  A fuzz harness must give the code under test exactly its input, no padding,
  or out-of-bounds reads go unseen.
- A test that compares nothing must fail: print compared counts and check
  they are nonzero. Several `0 == 0` checks have passed before.
- Pre-Volta: every Slang `.cu` is compiled with `tools/slang_cuda_prevolta.h`
  pre-included (build_all.py, build_kubins.py). Before sm_70 it makes all
  `*_sync` and `_wave*` calls use `__activemask()` instead of Slang's tracked
  lane masks, which assume Volta semantics; on sm_70+ it changes nothing (the
  sm_89 cubins are byte-identical). `WaveIsFirstLane` reads the mask directly,
  so `tools/slang_compat.h` replaces it. A new wave intrinsic that reads the
  mask without a `*_sync` call needs the same treatment.
- Worker (DeepSeek via opencode) output is re-verified before commit; say in
  the commit message when a change was worker-drafted.
- Kernel fusion only when all three hold: (1) one workgroup produces exactly
  the tile the consumer needs; (2) the tile fits in groupshared on every
  target, at most 16 KB per group (Vulkan minimum); (3) the intermediate's
  round trip through device memory is a large, measured share of the time.
  Never merge kernels only to save launches (batch them, or later graphs).
  Each stage is written once, as generic device functions in `kernels/lib/`
  over a source/target interface (global buffer or groupshared tile); fused
  and unfused entry points are thin files over them. The unfused kernels stay,
  as fallback (blocks too big for a tile) and as the test oracle (an env
  switch forces them; tests compare both paths).
- Timing: the login node's L40S is shared with other users' jobs. Check
  `nvidia-smi` first (utilisation near 0, no large foreign process); a busy
  GPU makes every number and every ratio meaningless. For a clean GPU, run a
  short Slurm job: `--partition=low-gpu --gres=gpu:nvidia_l40s:1` (example:
  `$SLABITZ_WS/tmp/p3chk/slurm/overlap.sbatch`).
- Timing: ncu does not work on the L40S node (`RmProfilingAdminOnly: 1`); do
  not spend time on environments for it. Use `SLABITZ_PROFILE=1` (CUDA events,
  syncs after every op, so small kernels look slower than they are). LZ4
  reference: `bench/ref_bslz4_cuda.py` (times bslz4_to_sparse's GPU
  `chunk2sparse.decompress`; the top-level `decompress` is its CPU decoder).

## Pre-Volta (crunch)

crunch (TITAN Black sm_35, driver 340) mounts the workspace. Build here:
`pixi run build && pixi run -e kepler build-sm35` (`build/sm_35/`, NVRTC
11.8). On crunch, from the same workspace:
`python3 check_bshuf.py --lib build/libslabitz.so --kernels build/sm_35 --data
data`; the workspace pixi env runs there too, so any test works with
`SLABITZ_KERNEL_DIR=$PWD/build/sm_35 .pixi/envs/default/bin/python3
tests/<name>.py` (per-kernel tests take the cubin as argv). The portable
set: `PYTHON=$PWD/.pixi/envs/default/bin/python3 bash
tests/run_suite_kernel_dir.sh $PWD/build/kernels`. The tests call `make`:
build the library on the login node first (`make -q` must be clean), or
make relinks it on crunch, where there is no patchelf. CUDA device 0
is the TITAN Black (nvidia-smi lists it second). `tools/dump_parse.py`
dumps the parse-stage buffers for diffing two GPUs. Kernels built by NVRTC
11.8 that call `printf` do not load under driver 340; debug by writing values
to an output buffer.

## Next steps (in order)

1. **Old cards: spot-check** sm_50 (960M) and sm_61 (1050 Ti) with the
   portable fatbins: `tests/run_suite_kernel_dir.sh build/kernels` on each.
   Deferred until the user is home with those machines. sm_35 is the floor:
   Fermi (sm_20, crunch's Quadro 600) is not supported (no warp shuffles,
   no toolkit after CUDA 8); the library fails there cleanly with
   `SBZ_ERR_KERNEL` ("no kernel image").
2. **P4 streaming API, Python wrapper, wheel**: `docs/plan_p4_api.md`
   (five steps; decisions settled with the user 2026-11-20: caller-owned
   device memory, ctypes over bslz4_to_sparse's `_backend_drv.py`, a wheel
   with the fatbins, no HDF5 helper yet). Step 1 (audit of the mid-call
   readbacks, overlap measurement) is done and waits for the user's review:
   `docs/p4_readback_audit.md`. Two threads sharing one CUDA context already
   overlap up to x1.8 today; separate contexts do not. Step 2a (LZ4) done,
   912d07e: `sbz_dctx_bslz4_decompress_async` + `sbz_dctx_wait`, one thread
   alternating two contexts: silicon LZ4 0.38 ms per 16-frame call (GPU-bound),
   CeO2 about 3.3 ms against a 2.35 ms upload. `SLABITZ_TRACE=1` (device
   timeline from CUDA events, no syncs; `bench/p4_trace_lz4.py`) showed why:
   the input upload now goes before the host walk (it was queued after it,
   leaving the GPU idle 1.45 ms per call), and the two contexts' uploads
   share PCIe at the same time, so their kernels also coincide while PCIe is
   idle. Back-to-back uploads need one upload stream plus events (P4 step 3:
   the caller's uploads). Review decisions (2026-11-20, in the plan): scratch of
   the order of the output is fine (Eiger2 16M u16, ~35 MB per frame, cards
   from 1 GB), errors are rare enough for a slow error path, a full-GPU
   option (chunk walk on the device) is wanted later (step 2c). Step 2b
   (zstd) done, 806b260: no readback after capacities are learned, capacities for the
   literal/sequence buffers, one check at `sbz_dctx_wait` with an exact
   replay on overflow or error; primary CUDA context. Zero-copy zstd input
   (2026-11-20): walked and uploaded in place, pinned input makes the zstd
   call async too (CeO2 serial 9.42 -> 7.62 ms). Step 2c for LZ4 done
   (1ecfae2): `sbz_dctx_bslz4_decompress_device` takes device-resident chunks
   and a stated geometry; a kernel walks the block records (latency-bound:
   ~0.4 / 1.3 ms per call, overlapped with the other context); CeO2 1.31 ms
   per call with resident input. Next: 2c for zstd (frame refs from the same
   walk; block-type counts need capacities and guards in the frame-parsing
   kernels), then step 3 (public header). Steps 1-2 are subtle C runtime
   work, not for the DeepSeek worker.
   Design for streaming (the intended use): frames arrive as a stream, the
   upload of batch n+1 overlaps the decode of batch n, decoded frames stay on
   the GPU and are reduced there, and only results smaller than the
   compressed input come back. Throughput is then max(upload, device time)
   per frame, so a call must not block the host: pinned (or caller-owned
   device) input, no mid-call readback except exact mode (the first call, a
   replay and `SLABITZ_CHECK_SYNC=1` still read counters; the LZ4 status is
   read at `sbz_dctx_wait`), per-frame
   status left on the device for the caller to read later, and buffers
   reused across calls (double-buffered for overlap).
   Packaging: the 22 per-kernel fatbins are intentional (one CUDA module per
   kernel); if a single distribution file is wanted later, add a host-side
   bundle loaded by slice, never a merged module.
3. **Pac ideas for zstd (decode side)**: only offset-1 fills are in
   ExecuteSequences (e870151). The remaining decoder idea is the plane-repeat
   case (match offset = one plane): pass the optional plane geometry from the
   bitshuffle-zstd path and copy with an incremental index instead of a modulo
   per byte, guarded so stock zstd is unchanged. Add a committed test/bench on
   the pac zstd files (`silicon_u16_zstd2_256k.h5`, `ceo2_u32_zstd2_256k.h5`,
   read-only in `SLABITZ_EXAMPLES`) and measure against stock zstd. The pac
   encoder is a separate project and stays out of scope. Subtle C/kernel work,
   not for the DeepSeek worker.
4. **Slang CPU target survey (compile only)**: Slang's CPU target is
   experimental (C++ kernel/host, exe/lib) and compute-only. Compile the 22
   entries for it with the pinned slangc, record which wave/`groupshared`/
   barrier intrinsics fail (the Metal survey is the template), and decide
   whether a CPU fallback is worth a host backend. No runtime work yet.
5. **P5 tests**: `docs/plan_p5_tests.md` (pytest entry point, feature
   matrix with oracle-proved coverage, random round trips, corrupt
   compressed blocks on the GPU, compute-sanitizer). Independent of P4;
   suits the worker; P4 step 1 is done, so it can start any time.
6. **P6 performance** (`SLABITZ_PROFILE=1` prints a per-kernel table): the
   remaining cost is DecompressSequences (one thread per compressed block) and
   ExecuteSequences (one workgroup per frame): a latency floor per call (1
   silicon frame: 1.55 ms wall, of which ExecuteSequences 1.0 ms and
   DecompressSequencesMultiStream 0.29 ms are device time). "CeO2 gets slower
   per frame from 4 to 16 frames/call" was measured with harnesses that join
   the chunks inside the timed loop (45 ms of a 16-frame CeO2 call); re-check
   with the input built once. RAW/RLE
   block-size prefixes are now unused by MemsetMemcpy (still checked by
   run_chain_stage01). Then: time each stage on batches, compare the full chain
   with nvCOMP; overlap uploads with decode; reuse buffers across calls. Large
   frames (one workgroup per frame) are slow but not an issue at pac's 248 kB.
   Measured 2026-11-20 (L40S, 16 frames/call, device us per frame): silicon
   zstd ExecuteSequences 87, DecompressSequencesMultiStream 18, unshuffle 34,
   ~13 small kernels ~15 (synced); CeO2 upload 193 (PCIe-bound, 21 GB/s),
   ExecuteSequences 112, literals 71, sequences 45, unshuffle 63.
   - Fusing the unshuffle into ExecuteSequences fails the fusion rule (one
     workgroup per frame, 256 kB blocks) and would run the unshuffle on 16
     SMs instead of the whole GPU. Revisit only after ExecuteSequences has
     more parallelism.
   - With overlapped uploads (P4 streaming) LZ4 is PCIe-bound on CeO2 (upload
     183 vs device 59 us/frame) and device-bound on silicon (11 vs 25); zstd
     is device-bound on both. ExecuteSequences parallelism is the main lever.
   - CUDA graphs: low priority. Phase 3 step 4 found the per-call fixed cost
     is device time (above), not launches: graphs would save at most ~0.2 ms
     per zstd call and less for LZ4. The async path already defers the readback to the wait (P4 2b);
     exact mode still reads counters mid-call. If graphs are tried later, the
     preconditions are: finish removing the exact-mode mid-chain
     `read_counters()` readback (`src/sbz_chain.c`, after stage 0) with the
     indirect-dispatch pattern DS was written for (worst-case buffer sizes
     bounded by the input, kernels read their counts from the device and exit
     early; Vulkan's `vkCmdDispatchIndirect` and command buffers are the same
     design); stable buffer addresses (rebuild the graph when a buffer
     grows); graph calls optional via dlopen (driver 410+), plain launches on
     Kepler's driver 340. Capture with stream capture around the existing
     launch code.

Out of scope for v1: dictionaries, GPU checksum verification (checksums are
skipped), compression. Vulkan backend later (behind `src/sbz_backend.h`).
