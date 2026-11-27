#!/usr/bin/env bash
# T11: run compute-sanitizer (memcheck and racecheck) over a small case of each
# chain test and over the GPU corrupt-block fuzz.
#
# compute-sanitizer is not in the pixi environment; it is in the cuda/12.9.1
# module.  We find it the same way on the GPU node (a login shell runs the
# module).  ncu does not work on the L40S node (RmProfilingAdminOnly: 1); the
# sanitizer may hit the same limit, so we first check memcheck runs at all and,
# if it refuses, record the exact error and stop (that is a result, not a
# failure).
#
# Tools: memcheck over one zstd frame, one bitshuffle-zstd chunk, one
# bitshuffle-LZ4 chunk fused and one unfused, and over the corrupt fuzz; then
# racecheck over the same.  Writes everything to build/sanitizer/ and exits 1
# on any sanitizer error.
set -uo pipefail
ST="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ST"
. "$ST/../../env.sh" 2>/dev/null || . ../../env.sh

ROOT="$PWD"
LIB="$ROOT/build/libslabitz.so"
KDIR="${SLABITZ_KERNEL_DIR:-$ROOT/build}"
OUT="$ROOT/build/sanitizer"
LOG="$OUT/sanitizer.log"
mkdir -p "$OUT"
: > "$LOG"
printf 'sanitizer run  (worktree %s)\n' "$ST" | tee -a "$LOG"

# --- find compute-sanitizer (cuda/12.9.1 module, login shell) ---
SAN="$(bash -lc 'module load cuda/12.9.1 2>/dev/null && command -v compute-sanitizer' 2>/dev/null | tail -1)"
if [ -z "$SAN" ]; then
    printf 'FAIL: compute-sanitizer not found (tried module load cuda/12.9.1)\n' | tee -a "$LOG"
    exit 1
fi
printf 'compute-sanitizer: %s\n' "$SAN" | tee -a "$LOG"
if ! timeout 60 "$SAN" --version >>"$LOG" 2>&1; then
    printf 'FAIL: compute-sanitizer --version failed on this node\n' | tee -a "$LOG"
    exit 1
fi

SAN_OPTS="--check-api-memory-access yes"
RESULT=0

# A small single-case decoder, used under the sanitizer.  argv:
#   kind (zstd|bshuf|bslz4)  path  elem  fused(""|fused|unfused)
cat > "$OUT/decode_case.py" <<'PY'
import ctypes as ct, os, sys
LIB = os.environ["SLABITZ_LIB"]
KDIR = os.environ["SLABITZ_KDIR"]
kind, path, elem, fused = sys.argv[1], sys.argv[2], int(sys.argv[3] or 0), sys.argv[4]
if fused == "fused":
    os.environ["SLABITZ_LZ4_FUSED"] = "1"
elif fused == "unfused":
    os.environ["SLABITZ_LZ4_FUSED"] = "0"
class Span(ct.Structure):
    _fields_ = [("offs", ct.c_uint64), ("size", ct.c_uint64)]
lib = ct.CDLL(LIB)
lib.sbz_dctx_create_cuda.restype = ct.c_void_p
lib.sbz_dctx_create_cuda.argtypes = [ct.c_int, ct.c_char_p, ct.c_char_p, ct.c_size_t]
lib.sbz_dctx_destroy.argtypes = [ct.c_void_p]
lib.sbz_dctx_error.restype = ct.c_char_p
lib.sbz_dctx_error.argtypes = [ct.c_void_p]
lib.sbz_dctx_decompress.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_size_t]
lib.sbz_dctx_bshuf_decompress.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_size_t,
                                          ct.POINTER(Span), ct.c_uint32, ct.c_uint32]
lib.sbz_dctx_bslz4_decompress.argtypes = lib.sbz_dctx_bshuf_decompress.argtypes
lib.sbz_dctx_read_buffer.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_void_p, ct.POINTER(ct.c_size_t)]
def read_buf(dc, name):
    n = ct.c_size_t(0)
    assert lib.sbz_dctx_read_buffer(dc, name.encode(), None, ct.byref(n)) == 0
    b = ct.create_string_buffer(max(n.value, 1))
    assert lib.sbz_dctx_read_buffer(dc, name.encode(), b, ct.byref(n)) == 0
    return b.raw[:n.value]
err = ct.create_string_buffer(512)
dc = lib.sbz_dctx_create_cuda(0, KDIR.encode(), err, len(err))
assert dc, err.value.decode()
src = open(path, "rb").read()
if kind == "zstd":
    rc = lib.sbz_dctx_decompress(dc, src, len(src))
    out = read_buf(dc, "UnCompressedFramesData") if rc == 0 else b""
else:
    fn = "sbz_dctx_bshuf_decompress" if kind == "bshuf" else "sbz_dctx_bslz4_decompress"
    arr = (Span * 1)(Span(0, len(src)))
    rc = getattr(lib, fn)(dc, src, len(src), arr, 1, elem)
    out = read_buf(dc, "BshufOutput") if rc == 0 else b""
lib.sbz_dctx_destroy(dc)
print("decode %s %s: rc=%d out=%d bytes" % (kind, path, rc, len(out)))
sys.exit(0 if rc == 0 else 2)
PY
chmod +x "$OUT/decode_case.py"

decode_case() { # $1=tool $2=label $3=kind $4=path $5=elem $6=fused
    local tool="$1" label="$2" kind="$3" path="$4" elem="$5" fused="$6"
    printf '\n=== %s / %s ===\n' "$tool" "$label" | tee -a "$LOG"
    local rc
    SLABITZ_LIB="$LIB" SLABITZ_KDIR="$KDIR" \
        timeout 900 "$SAN" --tool "$tool" $SAN_OPTS \
        python3 "$OUT/decode_case.py" "$kind" "$path" "$elem" "$fused" >>"$LOG" 2>&1
    rc=$?
    if [ "$rc" -ne 0 ]; then
        printf '  %s: FAIL (exit %d; see %s)\n' "$label" "$rc" "$LOG" | tee -a "$LOG"
        RESULT=1
    else
        printf '  %s: ok (exit 0)\n' "$label" | tee -a "$LOG"
    fi
}

# --- first: check memcheck is functional on this L40S at all.  A working
# memcheck on a decode that actually reads out of bounds returns a nonzero
# exit (the kernel launch fails after the flagged access); that is a real
# FINDING, not a "cannot run".  Only a run that never produces the
# compute-sanitizer banner (the tool itself refusing, e.g. RmProfilingAdminOnly)
# counts as "cannot run on this node". ---
printf '\n=== check memcheck runs on this node (RmProfilingAdminOnly probe) ===\n' | tee -a "$LOG"
SLABITZ_LIB="$LIB" SLABITZ_KDIR="$KDIR" \
    timeout 300 "$SAN" --tool memcheck --check-api-memory-access yes \
    python3 "$OUT/decode_case.py" zstd tests/data/compat/v1.5.7--level3_text.zst 0 "" >"$OUT/probe.log" 2>&1
if ! grep -q "COMPUTE-SANITIZER" "$OUT/probe.log"; then
    printf 'compute-sanitizer memcheck did not run on this node; recording the error and skipping.\n' | tee -a "$LOG"
    tail -n 20 "$OUT/probe.log" | tee -a "$LOG"
    printf 'SKIP\n'
    exit 0
fi
printf 'memcheck runs on this node (it reports any out-of-bounds access it finds).\n' | tee -a "$LOG"

# --- the chain cases, memcheck then racecheck ---
for TOOL in memcheck racecheck; do
    decode_case "$TOOL" "zstd level3 text frame"        zstd   tests/data/compat/v1.5.7--level3_text.zst   0 ""
    decode_case "$TOOL" "bitshuffle-zstd ceo2 chunk"    bshuf  data/pac/ceo2_u32_zstd_maskmax.chunk.bin    4 ""
    decode_case "$TOOL" "bitshuffle-LZ4 fused"          bslz4  data/pac/silicon_u16_lz4.chunk.bin        2 fused
    decode_case "$TOOL" "bitshuffle-LZ4 unfused"        bslz4  data/pac/silicon_u16_lz4.chunk.bin        2 unfused
done

# --- the corrupt-block fuzz, in-process (single process so the sanitizer
# instruments the decode), 200 inputs as the brief says ---
for TOOL in memcheck racecheck; do
    printf '\n=== %s / run_gpu_corrupt.py --sanitize 200 ===\n' "$TOOL" | tee -a "$LOG"
    SLABITZ_LIB="$LIB" SLABITZ_KDIR="$KDIR" \
        timeout 1800 "$SAN" --tool "$TOOL" $SAN_OPTS \
        python3 tests/run_gpu_corrupt.py --sanitize 200 --jobs 1 --seed 20261127 \
        >>"$LOG" 2>&1
    rc=$?
    if [ "$rc" -ne 0 ]; then
        printf '  run_gpu_corrupt.py sanitize: exit %d (a fault tripped; see %s)\n' "$rc" "$LOG" | tee -a "$LOG"
        RESULT=1
    else
        printf '  run_gpu_corrupt.py sanitize: executed\n' | tee -a "$LOG"
    fi
done

printf '\n--- sanitizer result ---\n' | tee -a "$LOG"
# a compute-sanitizer memory/race error is reported as an "Invalid __global__"
# access or a nonzero "ERROR SUMMARY".
if grep -E "ERROR SUMMARY: [1-9]|Invalid __global__|Invalid __shared__|Race discovered|ERROR SUMMARY" "$LOG" >/dev/null 2>&1; then
    RESULT=1
fi
if [ "$RESULT" -ne 0 ]; then
    printf 'FAIL: sanitizer reported errors (see %s)\n' "$LOG" | tee -a "$LOG"
else
    printf 'PASS: no sanitizer errors\n' | tee -a "$LOG"
fi
printf 'log: %s\n' "$LOG" | tee -a "$LOG"
exit "$RESULT"
