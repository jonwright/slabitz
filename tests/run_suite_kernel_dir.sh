#!/bin/bash
# Run the regression suite against one kernel directory (the per-kernel fatbins).
#
#   tests/run_suite_kernel_dir.sh DIR
#
# DIR is what SLABITZ_KERNEL_DIR points at (e.g. build/kernels or a PTX-only set).
# The chain tests read SLABITZ_KERNEL_DIR; the per-kernel tests take the named
# .fatbin as their file argument. This script rebuilds NOTHING: build/libslabitz.so
# and the kernel directory must already exist (the chain tests' internal `make -s`
# is a no-op once the library is up to date).
set -e
DIR="${1:?usage: run_suite_kernel_dir.sh DIR}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
# Run from the project's pixi environment (`pixi shell`, or set PYTHON to its
# python, e.g. PYTHON=$ROOT/.pixi/envs/default/bin/python3 on a machine without
# pixi): zstandard/numpy versions matter for the test vectors.
PY="${PYTHON:-python3}"
echo "kernel dir: $DIR"
echo "lib:        $ROOT/build/libslabitz.so"
echo "python:     $(command -v "$PY")"
export SLABITZ_KERNEL_DIR="$DIR"
fail=0
run() {  # run NAME CMD...
  local name="$1"; shift
  printf '%-20s' "$name"
  if timeout 600 "$@" >"$ROOT/build/suite_$name.log" 2>&1; then
    echo " PASS"
  else
    echo " FAIL (see build/suite_$name.log)"; fail=1
  fi
}
run chain_stage01  "$PY" "$ROOT/tests/run_chain_stage01.py"
run chain_stage2   "$PY" "$ROOT/tests/run_chain_stage2.py"
run chain_frames   "$PY" "$ROOT/tests/run_chain_frames.py"
run bshuf_chain    "$PY" "$ROOT/tests/run_bshuf_chain.py"
run bslz4_chain    "$PY" "$ROOT/tests/run_bslz4_chain.py"
run zstd_async     "$PY" "$ROOT/tests/run_zstd_async.py"
run bslz4_device   "$PY" "$ROOT/tests/run_bslz4_device.py"
run unshuffle_slow "$PY" "$ROOT/tests/run_unshuffle_slow.py" "$DIR/unshuf/unshuf.fatbin"
run unshuffle_warp "$PY" "$ROOT/tests/run_unshuffle_warp.py" "$DIR/unshufw/unshufw.fatbin"
exit $fail
