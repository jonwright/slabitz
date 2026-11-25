# T16: Wheel

Status: todo
Executor: worker (DeepSeek), reviewed by Claude
Depends on: T05 (the kernel sets verified on old cards), T14
Branch: `t16-wheel`

## Goal

`pip install slabitz-*.whl` gives a working decoder on any card from sm_35
to sm_90, and on newer cards through PTX. Nothing has to be built on the
user's side. This is P4 step 5 of `docs/archive/plan_p4_api.md`.

## Background

- **The fatbins.** `build/kernels/` holds 22 fatbins, one per kernel
  (5.1 MB), covering sm_35, sm_50..sm_90 and compute_90 PTX. They are
  built by `sbz build-sm35 && sbz build-fatbins`.
- **One module per kernel** is intentional (`AGENTS.md` rules). If one
  distribution file is wanted later, it must be a host-side bundle loaded
  by slice, never a merged module.
- **The library must have no RPATH** (`AGENTS.md` rules). Build only
  through `sbz` (the shared env) or the Makefile with PATCHELF.
- **The kernel directory** defaults to `kernels/` next to the loaded
  library (T13's `dladdr` logic). `SLABITZ_KERNEL_DIR` overrides it.

## Settled design

- **The `sbz wheel` task:**
  - builds the sm_35 set (kepler environment), the fatbins and the
    library;
  - assembles `slabitz/` with `libslabitz.so` and
    `kernels/<sub>/<stem>.fatbin` plus `sbz_kernels.id` as package data;
  - builds a pure-ctypes wheel tagged `py3-none-manylinux_2_28_x86_64`.
- **Tooling.** Use a pyproject build backend that needs no compiled
  extension (hatchling or setuptools). The `.so` is package data.

## Files that may change

`pyproject.toml` (new or extended), `pixi.toml` (wheel task),
`tools/build_wheel.py` (if needed), `slabitz/_lib.py` (kernel dir default),
`README.md` (install), `docs/dev/reports/T16.md`.

## Acceptance

1. `sbz wheel` produces `dist/slabitz-<ver>-py3-none-manylinux_2_28_x86_64.whl`.
2. `unzip -l` lists `libslabitz.so`, 22 fatbins and `sbz_kernels.id`.
3. `readelf -d libslabitz.so | grep -i -E "rpath|runpath"` prints nothing.
   `ldd` shows no conda paths.
4. **Fresh-venv test.** Make a venv in
   `$SLABITZ_DEV_ROOT/tmp/t16-wheel/venv` (not /tmp), with numpy and h5py,
   and `pip install` the wheel. This test venv is the one allowed use of
   pip; never install into the shared env. With `SLABITZ_KERNEL_DIR`
   unset, `pytest tests/test_python.py` passes on the L40S.
5. **Old cards.** The same fresh-venv test on crunch (TITAN Black) and on
   the sm_61 machine when available, or the report states which cards
   were not available.

## Reviewer checks

- Install the wheel yourself in a new venv and decode one pac fixture.
- Corrupt `sbz_kernels.id` inside the installed package: the library must
  refuse to load the kernels, with a clear message.
