# Plan: one fatbin per kernel, for every supported card (done)

Outcome (2026-11-20, worker-drafted ebc99bc, review ebac966): 21 fatbins,
4.8 MB, built in 3.4 s; after `BshufWalkChunks` was added (P4 2c) the set is
22 fatbins, 5.1 MB. The suite passes from `build/kernels` on the L40S
(within 2% of the local cubins: the driver picks the sm_89 SASS) and on the
TITAN Black under driver 340, so the CUDA 12.6 fatbin container loads there
and no Kepler-only directory is needed. The plan's PTX test was wrong: a
compute_90 PTX cannot run on the sm_89 L40S (PTX runs on its own
architecture or newer). The shipped PTX stays compute_90, for cards after
sm_90; the worker checked the JIT path with a compute_89 set (`--ptx 89`,
first call about 1 s). The review fixed the output path (`<sub>/`, not
`<stem>/`) and made the suite script take `PYTHON` instead of sourcing a
file that only existed in the worker's clone.

The packaging item in `AGENTS.md`. Today a kernel directory holds
cubins for one architecture (`build/` for the local card, `build/sm_35/` for
Kepler, made by separate commands), so a user must build for their own card.
After this job, `build/kernels/` holds one fatbin per kernel that runs on any
card from sm_35 to sm_90, and on newer cards through PTX.

## Read first

1. `AGENTS.md`, sections "Rules that keep it correct" and "Pre-Volta".
2. `tools/build_all.py` (`KERNEL_MAP`, how each Slang `.cu` is compiled,
   including the pre-included `tools/slang_cuda_prevolta.h`).
3. `tools/build_kubins.py` (the sm_35 build with NVRTC 11.8) and the
   `kepler` environment in `pixi.toml`.
4. `src/sbz_cuda.c`, `load_kernel()`; `src/sbz_kernels_gen.h` (the `cubin`
   path per kernel); `tools/gen_kernel_table.py`.
5. `tools/build_cuda_fatbin.py` only as a warning: it merges all kernels
   into one module, which is silently wrong (AGENTS.md). Do not use or
   extend it.

## Design (settled; do not redesign)

- Still one CUDA module per kernel. A fatbin here is one kernel's `.cu`
  compiled for many architectures, never several kernels in one file.

## Future cleanup (not now)

The 22 files are intentional. If a single distribution file is wanted later,
add a host-side bundle (an index plus the module images) and load each kernel's
slice by offset with `cuModuleLoadData`; never merge the kernels into one CUDA
module. The wheel planned in `docs/plan_p4_api.md` can already carry the 22
fatbins as package data, so a single artifact does not require a single file
inside it.
- New `tools/build_fatbins.py`. For each `KERNEL_MAP` entry
  `(ker, sub, cubin)`, from the existing `build/<sub>/<ker>.cu` (written by
  `tools/build_all.py`):
  - cubins for sm_50, 52, 60, 61, 70, 75, 80, 86, 89 and 90 with the
    default environment's nvcc 12.6, with the same flags and pre-included
    shim as `tools/build_all.py` (the shim is required on sm_50 to sm_61 and
    a no-op from sm_70);
  - PTX for compute_90 (forward compatibility, JIT on newer cards);
  - the sm_35 cubin from `build/sm_35/<sub>/<cubin>` (made by
    `pixi run -e kepler build-sm35`; NVRTC 11.8 is the last toolkit with
    sm_35). If it is missing, stop with a message that says to run that
    command, unless `--no-sm35` is given;
  - combined with the default environment's `fatbinary` (`--create`,
    `--image3=kind=elf,sm=NN,file=...`, `--image3=kind=ptx,sm=90,file=...`;
    check `fatbinary --help`) into `build/kernels/<sub>/<stem>.fatbin`,
    where `<stem>` is the cubin name without `.cubin`.
  - Compile the architectures in parallel (a process pool); report the
    build time and the total size of `build/kernels/`.
- `--archs` overrides the list; `--out` the output directory.
- Loader: `load_kernel()` in `src/sbz_cuda.c` first tries
  `<kernel_dir>/<sub>/<stem>.fatbin`, then `<kernel_dir>/<sub>/<stem>.cubin`
  (both from the table's `cubin` path), so every existing kernel directory
  keeps working. The error message names both paths when neither opens.
  `cuModuleLoadData` takes a fatbin image as it is; no other host change.
- `pixi.toml`: new default-environment task `build-fatbins` =
  `python3 tools/build_fatbins.py`. `pixi run build` is unchanged (local
  cubins, fast, for development).
- `README.md`: a short "Kernels for any card" section: the two commands
  (`pixi run -e kepler build-sm35`, then `pixi run build-fatbins`) and
  `SLABITZ_KERNEL_DIR=<repo>/build/kernels`.

## Verify (in the scratch clone)

- `bash ../build_test.sh` first (normal build and full suite).
- sm_35 cubins: `. ../activate_kepler.sh && python3 tools/build_kubins.py
  --arch sm_35 --out build/sm_35 --nvrtc $CONDA_PREFIX/lib/libnvrtc.so`, then
  back in the default env (`. ../activate.sh`) run `python3
  tools/build_fatbins.py`.
- `cuobjdump --list-elf --list-ptx build/kernels/exec/exec.fatbin` lists
  sm_35 ... sm_90 and compute_90; put the listing for one kernel in the
  report.
- The full suite with the fatbins: every chain test with
  `SLABITZ_KERNEL_DIR=$PWD/build/kernels` (the per-kernel tests
  `run_unshuffle_slow.py` / `run_unshuffle_warp.py` take the file as
  argv[1]: pass the `.fatbin`). All must pass with the same compared
  counts. Write a small script for this in `tests/` (e.g.
  `tests/run_suite_kernel_dir.sh DIR`) so the architect can rerun it on the
  Kepler machine; it must not call `make` or rebuild anything.
- Same speed as the local cubins: `SLABITZ_PROFILE=1` per-kernel table for
  `silicon_lz4.h5` and `silicon_u16_zstd2_256k.h5` from the read-only example
  data (`SLABITZ_EXAMPLES`), 16 frames per
  call, `build/` vs `build/kernels`. The driver must pick the sm_89 SASS
  (times within 2%).
- PTX path: build a PTX-only set (`--archs none` plus PTX, into
  `build/kernels_ptx`), run `tests/run_bshuf_chain.py` with
  `CUDA_CACHE_DISABLE=1 SLABITZ_KERNEL_DIR=$PWD/build/kernels_ptx`. It must
  pass (the driver JIT-compiles compute_90 for the L40S). Report the extra
  time of the first call.
- The Kepler machine (driver 340) is checked by the architect. You cannot
  reach it; do not try.

## Rules

- ASCII only.
- Change only: new `tools/build_fatbins.py`, `src/sbz_cuda.c`
  (`load_kernel`), `pixi.toml` (the new task), `README.md` (the new
  section), the new test script. Do not edit `AGENTS.md` or `docs/`, the
  Slang kernels, the generated `.cu`, `third_party/`, or `Makefile`.
- Do not commit anything under `build/`.

## Deliverables

1. One commit in the scratch clone: `kernels: one fatbin per kernel,
   sm_35..sm_90 + PTX (drafted by <agent>)`, with the test results, the
   per-kernel timing comparison and the size of `build/kernels/` in the
   message, and a `Co-Authored-By:` trailer naming you.
2. `REPORT_fatbin.md` in the clone root (not committed): commands run, the
   `cuobjdump` listing, test results, timings, the PTX JIT result, every
   decision you made and why.

## Acceptance (checked by the architect)

- Full suite passes from `build/kernels` on the L40S, with timings equal to
  the local cubins, and from `build/kernels_ptx` (JIT).
- The same `build/kernels` passes the suite on the TITAN Black (driver 340).
  If driver 340 cannot load a fatbin made by CUDA 12.6's `fatbinary`, the
  fallback (decided then) is a Kepler-only cubin directory.
- Existing kernel directories (`build/`, `build/sm_35/`) still load.
