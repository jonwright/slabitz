# AGENTS.md: slabitz status, rules and plan pointer

GPU decoders for bitshuffle-zstd and bitshuffle-LZ4 HDF5 chunks (filter 32008,
e.g. pac output). Microsoft DirectStorage (DS) GPU zstd shaders (HLSL),
transformed and compiled with Slang to CUDA, driven by a C host library
through the CUDA driver API; the LZ4 path is a Slang port of the nvCOMP
decoder used by bslz4_to_sparse. History and lessons: `docs/history.md`.
Build: `README.md`.

## Build and test

Agents work in the shared root `$SLABITZ_DEV_ROOT` (set in the user's
`~/.bashrc`; maintainer: `/data/id11/jon/hdftest/slabitz`, on GPFS, visible
from the login node, the Slurm GPU nodes and crunch). `$HOME` is nearly
full: never put builds, environments, data or worktrees there. The layout
and tools are defined in `docs/dev/T-1_agent_root.md`:

```
$SLABITZ_DEV_ROOT/AGENT_START.md    read first
. $SLABITZ_DEV_ROOT/env.sh          shared env (no pixi on PATH), data variables, sbz, gpu-run
cd $SLABITZ_DEV_ROOT/wt/<task>      your worktree (made by the reviewer: dev-root new <task>)
sbz build && sbz test               kernels linked from the shared cache by source key; lib built locally
gpu-run sbz test                    same on an idle Slurm L40S (all timing goes through this)
```

Shared and read-only: the environment (`env/`), slangc (`shared/slang`),
real data (`data/real`, `$SLABITZ_EXAMPLES`), synthetic data
(`data/synth/current`, `$SLABITZ_SYNTH`), toolchains, related repositories
(`ext/`: pac-zstd with `$PAC_LIB`, bslz4_to_sparse), and `main/`. Write only
in your worktree and `tmp/<task>`. Never run pixi, pip or conda, and never
download: a missing dependency goes into QUESTIONS.txt. Do not search the
filesystem for test data.

Until T-1 is done, the old single workspace is
`/data/id11/jon/hdftest/slabitz-workspace` (`. env.sh && cd slabitz &&
CUDA_ARCH=sm_89 pixi run test`). Its `examples` link lacks the thaumatin
files, which live in
`/data/id11/jon/hdftest/zenodo_2412270_dls_i04_eiger2_16m_thaumatin/`.
Outside ESRF, `tools/setup_workspace.sh` still makes a single-checkout
workspace (see README.md).

Run every GPU test under `timeout`: a wave-divergence bug shows up as a hang.

## Status (2026-11-27)

| Area | State |
|---|---|
| All kernels on the decode path (22 Slang entries, incl. the 2 bit-unshuffle kernels, the 2 LZ4 kernels and the device LZ4 chunk walk; shared stage code in `kernels/lib/`) | ported, each validated vs CPU ref / oracle |
| C chain: `sbz_dctx_decompress` (concatenated frames) | byte-exact vs libzstd, 11 cases |
| `sbz_dctx_decompress_frames` (explicit frame list) | byte-exact, real hdf5plugin chunks |
| `sbz_dctx_bshuf_decompress` (bitshuffle-zstd chunks) | matches hdf5plugin, u8-u64, 8k/248k/256k blocks, tails, corrupt input -> error |
| `sbz_dctx_bslz4_decompress` (bitshuffle-LZ4 chunks, `kernels/Lz4DecodeBlocks.slang`, fused `kernels/Lz4DecodeUnshuffle.slang` for full blocks of at most 8 kB) | matches hdf5plugin, u8-u64, 8k-256k blocks, tails, corrupt input -> error, fused == unfused; L40S and TITAN Black. Phases 1-3 of `docs/archive/plan_lz4_slang.md` done; plane-aware match copy (runs as fills, plane repeats without a modulo; pac's idea, 3a94af2). L40S, 16 frames/call, device us/frame: silicon 20.4 vs reference 19.6, CeO2 53 vs 67; wall faster than the reference on both (silicon 0.54 ms per 16-frame call, 263 GB/s). `SLABITZ_LZ4_FUSED=0` forces the unfused path |
| Real pac files (silicon u16, CeO2 u32), 16 frames/call | L40S 41 / 6.5 GB/s; TITAN Black 10 / 2.7 GB/s (`bench/old_cards.py`) |
| Volta+ (sm_70 and newer) | works (tested on sm_89) |
| Pre-Volta (sm_35 to sm_61) | works on the sm_35 TITAN Black (check_bshuf 17/17, all per-stage and chain tests) and on the sm_61 GTX 1050 Ti (2026-11-27, commit `bc43e16`, kernel ID `6446e283157f4598`: `run_suite_kernel_dir.sh` 9/9, `run_pac_fixtures.py` 7/7, all 15 `sbz test` scripts pass against `build/kernels`; `run_kernel_scripts.sh` 30/30 pass against `build/`; `docs/dev/reports/T05.md`); TITAN Black and sm_50 pending in T05 |
| Portable kernels (`pixi run -e kepler build-sm35 && pixi run build-fatbins` -> `build/kernels/`) | one fatbin per kernel: sm_35 (NVRTC 11.8), sm_50..sm_90 (nvcc 12.6), compute_90 PTX; 22 files, 5.4 MB. Rebuilt at `6446e283157f4598` and verified on GTX 1050 Ti (`sm_61`); L40S and TITAN Black re-check pending in T05 |
| Other Slang targets (compile only, slangc 2026.22) | Survey 2026-11-28 at 23 kernels: SPIR-V and HLSL 23/23; Metal 17/23 - the same 6 failures as before: `WavePrefixCountBits` has no Metal lowering (PrefixSum, ComputePrefixSum, ParseCompressedBlocks, InitFseTable) and Metal lacks `GetDimensions` (BitUnshuffleSlow, BitUnshuffleWarp; pass byte counts as scalars instead). `BshufWalkChunks` builds for SPIR-V, HLSL and Metal. CPU target (`-target cpp`) compiles 11/23; 12 fail on `error[E36107]` over wave intrinsics (`WaveGetLaneCount` x8, `WaveActiveSum` x1, `WavePrefixSum` x1) and the group barrier (`AllMemoryBarrierWithGroupSync` x2, the 2 LZ4 kernels); groupshared/LDS itself compiles. `docs/dev/reports/T22.md`, `T22_targets.md` |
| pac-written zstd (local clone of pac-zstd) | done on branch `pac-zstd` (`docs/archive/plan_pac_zstd.md`): verified byte-exact across the 12-case matrix (`bench/pac_matrix.py`: silicon/thaumatin/CeO2 x u16/u32 x mask=0/max-1, 648.9 MB compared) and real zstd files (`tests/run_bshuf_chain.py`). `ExecuteSequences` uses generic `uint4` fills (`off=1`), `uint4`/`uint32` aligned copies (`off=P`), fused short `off=1` fills, and `uint32` literal copies (`maskmax-1` `ExecuteSequences` 3.73x faster on `silicon_u32`); `InitHuffmanTableAndDecompressLiterals` overlaps `GS_PreInit`/`GS_HuffmanTable` (1614 -> 1292 dwords) and uses 1-search table expansion (`Literals` -21% to -24%) |
| `src/host_walk.c` on corrupt input | bounds-checked: any input -> `SBZ_OK` / `SBZ_ERR_CORRUPT` / `SBZ_ERR_LIMIT` / `SBZ_ERR_UNSUPPORTED` (22 named cases, Block_Maximum_Size = min(window, 128 KiB); ASan/UBSan fuzz, `make build/fuzz_host_walk`, 200k mutants in the test). Compressed-block contents are still trusted by the GPU kernels |
| Python wrapper, packaging | not started (a ctypes loader is in `slabitz/`) |
| Test harness | `sbz pytest` runs every `sbz test` and `sbz test-kernels` script (generated from the lists; markers `gpu`, `slow`); `SLABITZ_REQUIRE_DATA=1` turns missing data into failures; committed pac fixtures `data/pac` (7 cases, 272 KB) checked by `tests/run_pac_fixtures.py`; `docs/dev/reports/T02.md` |
| P4 streaming API | async LZ4 + zstd calls, device-input LZ4 (`sbz_dctx_*_async`, `_device`, `sbz_dctx_wait`); zstd device input, public header, per-chunk status: T12, T13 |
| zstd feature matrix (T06) | `tests/run_features.py` (in `sbz test`): 366 frames (libzstd 1.5.7 plus hand-built RLE literals, RLE and raw blocks), every frame-header, literals and sequence-mode cell covered by a byte-exact decode, 46.6 MB compared per pass, per-entry and shared-context passes; no xfail (the `Dictionary_ID` 0 frame is an expected UNSUPPORTED since T09); `tests/run_roundtrip.py` (slow): 1362 cases, 657 MB; `docs/dev/reports/T06.md` |
| zstd version coverage (T07) | `tests/run_compat.py` (in `sbz test`, 58 committed entries; `--full` on `data/compat/`, 283 entries, 2.06 GB compared): 9 zstd versions from bitshuffle 0.4.0's bundled zstd (`18d02cbf`, v1.4.7-403) to 1.6.0 plus 4 bitshuffle 0.4.0 chunks; every decode byte-exact, every expected reject (no FCS, window 2^30/2^31, dictionary) returns UNSUPPORTED and `--patch-from` frames CORRUPT (T09); `docs/dev/reports/T07.md` |
| Per-kernel Python scripts | every launcher calls `tests/kparams.check_args` (signatures parsed from `src/sbz_kernels_gen.h`) before launch; `sbz test-kernels` runs 22 scripts + 8 modes, 30 pass with nonzero compared counts, 2 retired (`run_parse_blocks`, `smoke_initfsetable`); `docs/dev/reports/T01.md` |
| Performance regression gate | `bench/gate.py`: 164 cells, L40S baseline `bench/baselines/l40s.json` (node frcrg-gpu5l40s-01); `docs/dev/reports/T03.md`: T03 |
| Open performance questions (T04) | L40S, `docs/dev/reports/T04.md`: pac 0.2.0 frames are 3.5% faster than 0.1 frames on the current kernel (rechecked 2026-11-27 on GTX 1050 Ti in `docs/dev/reports/T05.md`: `b840caa/0.1` 8218.0 us vs `main/0.2` 7700.5 us, -6.3% - "+30% ExecuteSequences" did not reproduce); CeO2 zstd time per frame falls with frames per call (1 -> 64: 5-9x), stock LZ4 with host input rises beyond 16 frames (host side, device input does not); fused LZ4 never slower (1.05-2.69x); b840caa made ExecuteSequences 8-19% slower on stock zstd with many plane repeats (wall +5-7%): not the views but per-match code in front of the byte loop in a latency-bound one-wave-per-frame kernel, no fix in the transform keeps the pac gains (T04a), left to T20 |
| Compression | not started; LZ4 then zstd: T17-T19 |
| Agent environment | shared root `$SLABITZ_DEV_ROOT`: one env, shared data and kernel cache, one worktree per task (`dev-root`, `sbz`, `gpu-run`); `AGENT_START.md`, `docs/dev/reports/T-1.md` |
| Synthetic data | fitted generator `tools/synth` (version 2), twins of the 8 real files, 36 extensions, 24 worst cases in `data/synth/current` (= `v2`); compare 250 of 256 checks pass (6 misses, all accepted by the maintainer: al2o3 x3 as known; hplc class 2-3 and eiger4m_u32 zstd 3 byte split as near enough), GPU decode time of 5 twin/real pairs within 5%; `docs/dev/reports/T00.md` |

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
- Work goes by task files in `docs/dev/` (see its README): one branch per
  task, a report `docs/dev/reports/TNN.md` with every acceptance command and
  its numbers, and the reviewer reruns the acceptance and reviewer checks
  before a merge. A check that did not run is "not run", never "pass".
- Every Python script that launches a kernel directly checks its argument
  list with `tests/kparams.py` `check_args` (from T01); run `sbz
  test-kernels` after any kernel signature change. A short `cuLaunchKernel`
  list is not an error at launch: the kernel reads garbage (2026-11-25: six
  scripts were behind for weeks).
- Performance claims come from `bench/gate.py --compare` over the whole
  matrix (from T03), never from the one case a change targets; "unchanged"
  means within the gate thresholds in every cell.
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
- Timing: performance claims come from the regression gate, `bench/gate.py`
  (`sbz gate`), over the whole matrix (164 cells: 5 codecs x 3 datasets x
  u16/u32 x 2 masks x sync/async/device, plus synthetic twins and worst cases),
  never from the one case a change targets. `--compare` checks a run against
  `bench/baselines/<gpu>.json`; `sbatch bench/gate.sbatch` writes the L40S
  baseline on an idle GPU. L40S nodes differ by up to 16% in many cells, so
  a compare runs on the baseline's node (the gate refuses another node unless
  `--any-node`). Rules:
  - A PR touching `kernels/`, `src/` or `tools/transform_ds_for_slang.py`
    includes the `--compare` table in its report, from
    `sbatch -w $(SLABITZ_GATE_GPU=l40s python3 bench/gate.py --baseline-node)
    bench/gate.sbatch --compare`.
  - A PR that makes cells faster refreshes the baseline
    (`bench/gate.py --write-baseline` in Slurm) in the same commit.
  - A cell that regresses needs an explicit reason in the report, accepted
    by the maintainer, and then a refreshed baseline.
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

## Plan and next steps

The single actionable plan is `docs/dev/README.md`: an ordered task list
(T01-T22, one file per task, each with measurable acceptance checks and
reviewer checks), plus how to execute and how to verify a task. Start there.
Finished and folded plans are in `docs/archive/` (history, not instructions).

Intended use, which every API and performance decision serves: frames arrive
as a stream, the upload of batch n+1 overlaps the decode of batch n, decoded
frames stay on the GPU and are reduced there (e.g. bslz4_to_sparse's sparse
and integrate kernels), and only results smaller than the compressed input
come back. Throughput is then max(upload, device time) per frame, so a call
must not block the host: caller-owned device (or pinned) input, no mid-call
readback except exact mode, per-chunk status left on the device, buffers
reused across calls.

Known limits (tasks that change them in brackets). Rejected before any GPU
work with `SBZ_ERR_UNSUPPORTED` (T09): frames without Frame_Content_Size
in plain zstd input (`sbz_dctx_decompress`, `_frames`) [T10 part 2,
deferred; inside bitshuffle-zstd chunks they decode since T10, sized from
the chunk's block size], a window of 2^29 bytes or more (kept: the GPU packs offsets
in 29 bits), any Dictionary_ID field (value 0 too). More than 1 GiB of zstd
output per call returns `SBZ_ERR_LIMIT` (the DS request limit; callers
split). On the GPU (T09), a frame that decodes past or short of its content
size, a block over 128 KiB, and a match reaching before its frame's start
(`--patch-from` frames) set a per-frame status and return `SBZ_ERR_CORRUPT`;
such blocks are not written, so neighbours stay intact. Other
compressed-block contents (Huffman, FSE, literals sections) are still
trusted by the kernels: T11's fuzz (`tests/run_gpu_corrupt.py`, slow) finds
11 GPU faults in 2008 corrupt inputs [T11b, deferred]. A fault is a CUDA
illegal-address error (719): the call fails and the process can no longer
use the GPU (primary context). The three committed reproducers each fault on
an out-of-bounds read (memcheck, 2026-11-28); no write was seen, but the
kernel stops at the first fault, so that is not a proof. Do not decode
untrusted data. Valid frames are memcheck-clean since T11a (literal slack).

Out of scope for v1: dictionaries (rejected, UNSUPPORTED), GPU checksum verification
(checksums are skipped). Compression is planned (T17-T19: LZ4 first, then
zstd, the subset pac-zstd emits). Vulkan backend later (behind
`src/sbz_backend.h`).
