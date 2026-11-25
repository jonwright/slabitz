# slabitz

GPU decoders for bitshuffle-zstd and bitshuffle-LZ4 HDF5 chunks (filter 32008)
from X-ray detectors. The zstd kernels are Microsoft's DirectStorage GPU zstd
shaders, compiled from HLSL to CUDA with the Slang compiler; the LZ4 path is a
Slang port of the nvCOMP decoder used by bslz4_to_sparse. A small C host
library (`src/`, `build/libslabitz.so`) runs the stage chain through the CUDA
driver API.

**Status: work in progress.** The C chain decodes zstd frames,
bitshuffle-zstd chunks and bitshuffle-LZ4 chunks, and matches libzstd /
hdf5plugin in the tests, on Volta and newer GPUs, on a Pascal GTX 1050 Ti
(sm_61), and on a Kepler GTX TITAN Black (sm_35). Older cards use the same
kernels: a small header changes how warp-wide operations pick their lanes on
pre-Volta GPUs and does nothing on newer ones. See `AGENTS.md`.

## Authors

@jonwright (ESRF) directed the AI coding agents Claude (Anthropic) and
DeepSeek V4 (served at the ESRF) to get zstd decoding on a GPU for the
bitshuffle-zstd format. This repository is the result: a port of Microsoft's
DirectStorage GPU zstd decoder (MIT) into the Slang language and then CUDA,
with work in progress to make it run on older cards too.

## Quick start

### At ESRF and for development agents: the shared root

Development (Claude, the DeepSeek worker, reviewers) happens in one shared root
folder, `$SLABITZ_DEV_ROOT`, set in `~/.bashrc`. It holds the environment, the Slang
compiler, the real and synthetic test data, built kernel sets and one git worktree per
task, so nobody sets up pixi, downloads data or rebuilds kernels. Layout and rules:
`docs/dev/T-1_agent_root.md`; the first file an agent reads is
`$SLABITZ_DEV_ROOT/AGENT_START.md`.

```bash
cd $SLABITZ_DEV_ROOT/wt/<task> && . $SLABITZ_DEV_ROOT/env.sh
sbz build && sbz test            # kernels linked from the shared cache, library built here
gpu-run sbz test                 # the same on an idle Slurm L40S
```

The maintainer creates the root once with `tools/dev_root.sh init` (see its header) and
makes task worktrees with `dev-root new <task>`.

### Anywhere else: pixi

The checkout is git-only. Pixi, its package cache, build temp files and the
environment live in a separate workspace. The usual layout is a gitignored
`workspace/` symlink next to the checkout:

```
$HOME/git/slabitz/            this checkout (no build, no .pixi)
$HOME/git/slabitz/workspace   -> pixi workspace (large disk; gitignored)
```

Create the workspace on a large filesystem:

```bash
WS=/path/to/workspace            # any folder; e.g. on a large data disk
mkdir -p $WS && git clone <this repo> $WS/slabitz
$WS/slabitz/tools/setup_workspace.sh --workspace $WS [--examples DIR]
# in an existing checkout, link it in:
ln -s $WS /path/to/checkout/workspace    # gitignored

. $WS/env.sh && cd $WS/slabitz
CUDA_ARCH=sm_50 pixi run test    # fetch slangc, build kernels + lib, run tests
```

`--examples DIR` creates `WS/examples` as a read-only link to the example HDF5
data and exports `SLABITZ_EXAMPLES`; the tests that need real files skip with a
notice when it is absent. Never write in the example data.

Set `CUDA_ARCH` to your card: `sm_50` for a GTX 960M, `sm_61` for a GTX 1050
Ti, `sm_89` for an L40S (the default). Cubins are built for that one
architecture. The workspace folder cannot be moved: on a new machine or path,
clone again and rerun `setup_workspace.sh`.

The pixi environment (`pixi.toml`, `pixi.lock`) provides Python 3.12, numpy,
zstandard, h5py, hdf5plugin, bitshuffle, gcc 13 and CUDA 12.6 (nvcc, NVRTC)
from conda-forge. The host must provide only the NVIDIA driver
(`libcuda.so.1`) for CUDA 12.x.

Tasks: `pixi run fetch-slangc`, `pixi run build`, `pixi run test`. Each calls
`tools/sbz.sh`, the same runner the shared root uses as `sbz`.

## Kernels for any card

`pixi run build` compiles one cubin for your local card (fast, for development).
To make a single kernel set that runs on every card from Kepler to Hopper, build
the portable kernels:

```bash
pixi run -e kepler build-sm35    # one set of cubins for the sm_35 floor (NVRTC 11.8)
pixi run build-fatbins           # one .fatbin per kernel: sm_35..sm_90 + compute_90 PTX
```

The result is `build/kernels/`: one fatbin per kernel, each carrying SASS for
sm_35, 50, 52, 60, 61, 70, 75, 80, 86, 89, 90 and PTX for compute_90 (JIT-compiled
on still-newer cards). Point the library at it with

```bash
SLABITZ_KERNEL_DIR=$PWD/build/kernels
```

`build/` (the per-arch cubins) and `build/kernels/` (the fatbins) both load;
`src/sbz_cuda.c` tries `<sub>/<stem>.fatbin` first, then `<sub>/<stem>.cubin`.
`build-fatbins` does not rebuild the Slang kernels or the library.

## Without pixi

Needs: x86_64 Linux with glibc >= 2.28, CUDA 12.x `nvcc`, a C compiler,
`make`, `curl`, Python 3 with numpy, zstandard, h5py, hdf5plugin, bitshuffle.

```bash
tools/fetch_slangc.sh                  # slangc 2026.22 -> third_party/slang/ (SHA-256 checked)
CUDA_ARCH=sm_89 python3 tools/build_all.py   # kernels -> build/, src/sbz_kernels_gen.h
make                                   # build/libslabitz.so
python3 tests/run_chain_stage2.py      # tests run from any directory
```

Tool overrides: `SLANGC`, `NVCC`, `CUDA_ARCH`, `BUILD_DIR`;
`SLABITZ_DS_DIR` builds against an upstream DirectStorage checkout instead of
the vendored headers. At run time `CUDA_DRIVER` selects the driver library,
`SLABITZ_KERNEL_DIR` the cubin directory, and `SLABITZ_DEBUG_SYNC=1`
synchronises after each stage.

## Layout

| Path | Contents |
|---|---|
| `kernels/*.slang` | Slang entry points for the DirectStorage kernels |
| `kernels/cpu_ref_*.py` | Python reference implementations of each stage |
| `src/` | C host library: host frame walk, CUDA backend, stage chain |
| `slabitz/` | Python ctypes loader used by the tests |
| `tools/` | build scripts, slangc fetch, DS header transform, test oracle |
| `tests/` | per-stage and whole-chain tests against libzstd / the oracle |
| `data/` | reference bitshuffle-zstd chunks (synthetic) for `check_bshuf.py` |
| `bench/` | benchmarks against nvCOMP (separate environment; method and results in `docs/history.md`) |
| `third_party/` | vendored DirectStorage headers and zstd educational decoder |

## License

MIT, Copyright (c) 2026 European Synchrotron Radiation Facility (ESRF) and Jon Wright; see
`LICENSE`. Third-party code and its licenses are listed in
`THIRD_PARTY_NOTICES.md` (DirectStorage: MIT; zstd educational decoder:
BSD-3-Clause, used only by the test oracle).
