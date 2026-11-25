# T-1: Shared agent root: one environment, shared data, one worktree per task

Status: todo
Executor: Claude (infrastructure every later task depends on; must be right)
Depends on: -
Branch: `t-1-agent-root`

## Goal

Every agent, Claude or the DeepSeek worker, starts in one root folder and
works on a task with no setup of its own. Today agents lose time and make a
mess rebuilding pixi environments, fetching slangc and hunting for test
data.

After this task:

- **Prepared once in the root:** the Python and CUDA environment, the slangc
  compiler, real example data, synthetic data (T00), external toolchains, and
  built kernel sets keyed by their sources.
- **Per agent:** a git worktree and a local build of the library.
- **Agents never run pixi, never download, and never read or write outside
  the root.**

## Inputs (settled with the user, 2026-11-25)

- **Root path.** The root is given by the environment variable
  `SLABITZ_DEV_ROOT`, set in each user's `~/.bashrc`. For the maintainer:
  `export SLABITZ_DEV_ROOT=/data/id11/jon/hdftest/slabitz`.
  - It is on GPFS (`/gpfs/gd`, 82 TB free).
  - It is visible from the login node and crunch; check the Slurm low-gpu
    nodes in acceptance 5.
  - `$HOME` is 98% full and must not hold builds, environments, data or
    worktrees. The checkout in `~/git/slabitz` stays as the user's own
    copy and is not used by agents.
- **The current workspace** (to be replaced, then deleted by the user once
  this task is accepted): `/data/id11/jon/hdftest/slabitz-workspace`. It
  contains `pixi/` (pixi v0.84.1 and a 5.2 GB package cache),
  `slabitz/.pixi`, and `examples ->
  /data/id11/jon/hdftest/bslz4_to_sparse_examples`.
- **Real data today:**
  - **`/data/id11/jon/hdftest/bslz4_to_sparse_examples/`**: Eiger2 4M CdTe,
    2162 x 2068, with sha256 sums in its README.md.
    - `silicon_u16_zstd2_256k.h5`: sparse, 250 frames.
    - `ceo2_u32_zstd2_256k.h5`: dense powder, 72 frames.
    - `mask.h5`.
    - `silicon_lz4.h5`, `silicon_lz4mask.h5`, `ceo2_lz4.h5` and
      `ceo2_lz4mask.h5`: bitshuffle-LZ4, 8 kB blocks; the mask versions
      have masked pixels set to 2^n-1.
  - **`/data/id11/jon/hdftest/zenodo_2412270_dls_i04_eiger2_16m_thaumatin/`**:
    Eiger2 16M, MX. `thaumatin_9_1_00000{1,2,3}.h5`, `_master.h5` and
    `_mask.h5`, with md5sums.txt. `bench/pac_matrix.py` needs
    `thaumatin_9_1_000001.h5`, which the current `examples` link does not
    contain, so the thaumatin cases have been silently skipped.
  - **Other candidates in `/data/id11/jon/hdftest/`:** `eiger4m_u32.h5`,
    `big_eiger_0000.h5`, `kevlar.h5`, `hplc.h5`, `Calibration_Al2O3.h5`.
    Inventory them (step 4) and include those that are bitshuffle chunks
    or useful frames.

## Layout (settled)

```
$SLABITZ_DEV_ROOT/
  AGENT_START.md        first file every agent reads (generated from this task's template)
  env.sh                source it: environment, PATH (sbz, gpu-run), data variables
  bin/                  sbz, gpu-run, dev-root (thin wrappers; the logic lives in the repo's tools/)
  admin/pixi/           pixi binary + package cache (admin only; NOT on the agents' PATH)
  env/                  worktree pinned to the commit whose pixi.lock the shared env was built from
    .pixi/envs/default  the shared environment (Python, numpy, h5py, hdf5plugin, zstandard, CUDA 12.6 nvcc/nvrtc, ...)
    .pixi/envs/kepler   NVRTC 11.8 for sm_35 builds
    activate.sh         static activation script (from `pixi shell-hook`), sourced by env.sh
    activate-kepler.sh
  shared/slang/         slangc 2026.22, fetched once (sha256-checked); worktrees link third_party/slang here
  kernels/<srckey>/     built kernel sets, write-once, keyed by a hash of the kernel sources:
                          sm_89/ sm_35/ kernels/ (fatbins), each with its sbz_kernels.id
  data/real/            read-only symlinks to the real files above + MANIFEST.sha256 + INVENTORY.md
  data/synth/vN/        synthetic data (T00), read-only once generated
  data/compat/          zstd version corpus (T07)
  toolchains/           zstd at several versions (T07), bitshuffle 0.4.0 venv (T07)
  ext/                  clones of related repos: pac-zstd (+ build/libpac.so), bslz4_to_sparse
  repo.git              bare clone of slabitz (origin = GitHub); all worktrees hang off it
  main/                 worktree of origin/main, kept current by the reviewer; read-only for agents
  wt/<task>/            one worktree per task, branch <task>; the agent writes here
  tmp/<task>/           TMPDIR of that task
  results/              benchmark outputs (bench/gate.py), per GPU and commit
```

## Settled design

- **Environment.** `env.sh`:
  - sources `env/activate.sh`;
  - sets `TMPDIR=$SLABITZ_DEV_ROOT/tmp/<task>` (the task is taken from the
    current worktree's directory name, `misc` elsewhere);
  - sets `SLABITZ_EXAMPLES=$SLABITZ_DEV_ROOT/data/real`,
    `SLABITZ_SYNTH=$SLABITZ_DEV_ROOT/data/synth/current`,
    `PAC_LIB=$SLABITZ_DEV_ROOT/ext/pac-zstd/build/libpac.so` and
    `SLABITZ_SLANG_DIR=$SLABITZ_DEV_ROOT/shared/slang`;
  - puts `$SLABITZ_DEV_ROOT/bin` on PATH;
  - does not put pixi on PATH.

  It works in a login shell, in a Slurm job and on crunch. It finds the
  root from its own location when `SLABITZ_DEV_ROOT` is unset.
- **One task runner, `tools/sbz.sh`, in the repo.** `bin/sbz` finds the
  current worktree with `git rev-parse --show-toplevel` and runs that
  worktree's `tools/sbz.sh`. pixi.toml tasks are changed to call
  `tools/sbz.sh` too, so there is one definition of each task.
  Subcommands:

  | Subcommand | What it does |
  |---|---|
  | `setup` | Links `third_party/slang` to `$SLABITZ_SLANG_DIR`. Idempotent. Run automatically by the others. |
  | `build [--arch sm_XY]` | Computes the source key (below). If `kernels/<srckey>/<arch>` exists, links `build/` kernel dirs to it; otherwise runs `tools/build_all.py` into `build/` and publishes the result to `kernels/<srckey>/` atomically (build in `tmp/`, then `mv`). Always runs `make` for the library, which takes seconds. |
  | `build-sm35`, `build-fatbins` | The same, with the kepler activation for sm_35. |
  | `test`, `test-kernels`, `pytest`, `gate` | The pixi tasks of the same name, run in the activated env. |
  | `where` | Prints root, worktree, source key, kernel dirs in use and data paths. |
- **Source key.** It is computed without slangc: a SHA-256 over the same
  file set as `ID_SOURCES` in `tools/gen_kernel_table.py`, plus
  `tools/gen_kernel_table.py`, `tools/build_all.py` and the build scripts.
  Factor the list into one place that both use. The real `sbz_kernels.id`
  inside each published set still guards loading. If the two ever
  disagree, `sbz` refuses and says so.
- **Admin script: `tools/dev_root.sh`.** It is wrapped by `bin/dev-root`,
  and is used by the user or by Claude as reviewer, not by workers.

  | Subcommand | What it does |
  |---|---|
  | `init` | Creates the layout. Seeds `admin/pixi/cache` from the old workspace cache with `cp -al` (same filesystem, no download), then installs `env/` (default and kepler), writes the activation scripts, fetches slangc into `shared/slang`, links `data/real/` and writes its MANIFEST and INVENTORY, clones `repo.git`, `main/` and `ext/`, builds libpac, builds the kernels for main, and writes `env.sh`, `bin/` and `AGENT_START.md`. Idempotent. |
  | `update-main` | Fetches, moves `main/` to origin/main, and runs `sbz build` there (publishing kernels for the new source key). |
  | `update-env` | Run when `pixi.lock` changed on main: reinstalls `env/` at that commit and rewrites the activation scripts. |
  | `new <task> [base]` | Creates `wt/<task>` on a new branch `<task>` from `base` (default origin/main), creates `tmp/<task>`, and runs `sbz setup` and `sbz build` there. Prints the path. |
  | `rm <task>` | Removes the worktree and tmp, but only when the branch is merged or the user passes `--force`. |
  | `check` | Health check (see Acceptance 1). |
- **GPU jobs: `bin/gpu-run <cmd...>`.** It runs the command on an idle L40S
  through `srun --partition=low-gpu --gres=gpu:nvidia_l40s:1`, inside the
  same env and worktree, with a timeout (default 1 h, `--time`). This is
  the only way timing numbers for reports and baselines are made. On crunch
  the command runs as it is.
- **`AGENT_START.md`** (template in the repo at `tools/AGENT_START.md.in`)
  holds:
  - the layout table;
  - "source env.sh, then `dev-root new <task>` is done for you by the
    reviewer; cd to `wt/<task>`";
  - the rules:
    - read only below the root;
    - write only in your worktree and `tmp/<task>` (plus `sbz`'s atomic
      publish into `kernels/`);
    - never write in `data/`, `main/`, `env/`, `shared/`, `ext/` or
      `toolchains/`;
    - never run pixi, pip or conda;
    - never download;
    - if a dependency is missing, write QUESTIONS.txt;
  - the commands (`sbz ...`, `gpu-run ...`);
  - where the task file and the report go.
- **Workers (opencode).** A worker runs with its working directory at
  `$SLABITZ_DEV_ROOT`, so its out-of-directory guard allows exactly the
  root. Its brief names `wt/<task>` as the only place to edit.
- **Environment changes.** A task that needs a new Python or conda package
  does not change the shared env. It writes QUESTIONS.txt; the reviewer
  adds the dependency on main and runs `dev-root update-env`. This task
  adds now what the planned tasks are known to need: `pytest` (T02) and
  `lz4` (python, T18).
- **Old instructions.** `tools/setup_workspace.sh` stays for people
  outside ESRF. `README.md` and `AGENTS.md` describe the root first, and
  the old single-checkout workspace second.

## Files that may change

`tools/sbz.sh`, `tools/dev_root.sh`, `tools/AGENT_START.md.in`,
`tools/gen_kernel_table.py` (factor out the source list only; the ID must
not change, checked in acceptance 3), `tools/fetch_slangc.sh` (honour
`SLABITZ_SLANG_DIR`), `pixi.toml` (tasks call `tools/sbz.sh`; add pytest
and lz4), `pixi.lock`, `README.md`, `AGENTS.md` ("Build and test"
section), `docs/dev/README.md`, `docs/dev/reports/T-1.md`. Everything else
is created under `$SLABITZ_DEV_ROOT`, outside the repo.

## Acceptance

1. **`dev-root check` exits 0** after `dev-root init`, and prints one OK
   line per item:
   - the env activates, and `python3 -c "import numpy, h5py, hdf5plugin,
     zstandard, pytest, lz4"` works;
   - `nvcc --version` reports 12.6, and the kepler NVRTC 11.8 is found;
   - slangc reports 2026.22;
   - every file in `data/real/MANIFEST.sha256` verifies;
   - `main/` is at origin/main;
   - `kernels/<srckey of main>/{sm_89,sm_35,kernels}` exist and each
     `sbz_kernels.id` equals `SBZ_KERNELS_ID` in main;
   - `PAC_LIB` loads with ctypes;
   - pixi is not on the agents' PATH.
2. **Fresh agent, cold path.** In a new login shell with only
   `SLABITZ_DEV_ROOT` set, run `. $SLABITZ_DEV_ROOT/env.sh && dev-root new
   t99-smoke && cd $SLABITZ_DEV_ROOT/wt/t99-smoke && sbz build && sbz test`.
   - It passes.
   - It takes at most 10 minutes in total, and the kernels are linked, not
     rebuilt. The log shows "kernels: linked <srckey>".
   - No network is used: the run neither calls pixi nor writes to
     `admin/pixi/cache`. Compare its mtime before and after.
3. **Kernel cache.** In `t99-smoke`, touch a comment in
   `kernels/ExecuteSequences.slang` and run `sbz build`. It builds locally
   (new source key), publishes `kernels/<newkey>/`, and `sbz test` passes.
   A second worktree with the same change links instead of building.
   `SBZ_KERNELS_ID` of main is unchanged by the refactor of
   `gen_kernel_table.py` (compare with the value before the change).
4. **Data inventory.** `data/real/INVENTORY.md` lists, for every linked
   file: dataset path, shape, dtype, chunking, filter id and options
   (block size, codec), frame count and sha256. The thaumatin files are
   included. `SLABITZ_EXAMPLES` points at `data/real`, and
   `bench/pac_matrix.py --only thaumatin` no longer skips.
5. **GPU access.** `gpu-run sbz test` passes on a Slurm L40S node. On
   crunch, `. env.sh && cd wt/t99-smoke && SLABITZ_KERNEL_DIR=$(sbz where
   --kernels sm_35) python3 tests/run_chain_stage2.py` passes. If crunch
   or Slurm is unavailable, the report says so.
6. **Worker smoke test.** An opencode run (DeepSeek) with its working
   directory at `$SLABITZ_DEV_ROOT` and a brief "in wt/t99-smoke, add one
   line to README.md, run `sbz test`, commit" completes with no
   out-of-directory rejection in its log, and no pixi or pip calls.
7. **Clean up.** `dev-root rm t99-smoke --force` removes the worktree,
   branch and tmp.

## Reviewer checks

- Rename `env/activate.sh` temporarily: `dev-root check` must fail on the
  env line, and `sbz test` must fail with a clear message, not a pixi
  attempt.
- Change one byte of a symlinked data file's manifest entry: `check` must
  fail on that file.
- `grep -rn "pixi " tools/sbz.sh` shows no invocation of pixi.

## Out of scope

Synthetic data (T00). The zstd version toolchains (T07 builds them into
`toolchains/`). Deleting the old workspace (the user does it).
