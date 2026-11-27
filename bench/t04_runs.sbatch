#!/bin/bash
# T04 measurements (docs/dev/T04_perf_open_questions.md, docs/dev/reports/T04.md): the gate
# harness of this checkout run against other commits' libraries and kernels (SLABITZ_LIB), two
# pac-zstd versions (PAC_LIB) and SLABITZ_LZ4_FUSED, rounds interleaved so that slow drift on
# the GPU hits every configuration alike. Submit from the worktree, on the baseline's node:
#   sbatch -w $(SLABITZ_GATE_GPU=l40s python3 bench/gate.py --baseline-node) \
#          bench/t04_runs.sbatch q1q2|q3q4|bisect [ROUNDS]
# Needs: wt/t04-at-{439e2f2,b840caa,a0338bf} built (tools/build_all.py + make), pac-zstd 0.1
# (76cb246) built at $T04_PAC01. Results: $SLABITZ_DEV_ROOT/results/t04/<job>/<tag>-r<n>.json
#SBATCH --partition=low-gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --time=03:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=slabitz-t04
#SBATCH --output=../../results/t04-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
: "${SLABITZ_DEV_ROOT:?set SLABITZ_DEV_ROOT (source env.sh before sbatch)}"
. "$SLABITZ_DEV_ROOT/env.sh"
WHAT=${1:?q1q2 or q3q4}
ROUNDS=${2:-3}
WT=$SLABITZ_DEV_ROOT/wt
PAC02=$PAC_LIB
PAC01=${T04_PAC01:-$SLABITZ_DEV_ROOT/tmp/t04-perf-questions/pac01/x86_64-conda-linux-gnu/libpac.so}
OUT=$SLABITZ_DEV_ROOT/results/t04/${SLURM_JOB_ID:-local}
mkdir -p "$OUT"
echo "node $(hostname) job ${SLURM_JOB_ID:-} harness $(git rev-parse --short HEAD) $(date -Is)"
nvidia-smi --query-gpu=name,pci.bus_id,utilization.gpu,memory.used --format=csv,noheader
sbz build

lib() {  # lib <commit|main|var-x>
    case $1 in
    main) echo "$PWD/build/libslabitz.so" ;;
    var-*) echo "$WT/t04-$1/build/libslabitz.so" ;;
    *) echo "$WT/t04-at-$1/build/libslabitz.so" ;;
    esac
}
gate() {  # gate <tag> <round> [env...] -- gate args
    local tag=$1 r=$2; shift 2
    local envs=()
    while [ "$1" != -- ]; do envs+=("$1"); shift; done; shift
    echo "=== $tag round $r $(date +%T)"
    env "${envs[@]}" timeout 1800 python3 bench/gate.py -o "$OUT/$tag-r$r.json" "$@" || echo "FAILED $tag r$r rc=$?"
}

case $WHAT in
q1q2)
    for r in $(seq 1 "$ROUNDS"); do
        # Q1: kernel x pac frame version, the pac-zstd cells
        for k in b840caa a0338bf main; do
            for p in 01 02; do
                pl=$PAC02; [ $p = 01 ] && pl=$PAC01
                gate q1-$k-pac$p "$r" SLABITZ_LIB="$(lib $k)" PAC_LIB="$pl" -- --only pac-zstd/
            done
        done
        # Q2: stock zstd (zstd3, zstd19, twins = zstd2 256k files) at four commits
        for k in 439e2f2 b840caa a0338bf main; do
            gate q2-$k "$r" SLABITZ_LIB="$(lib $k)" -- --only zstd3/ --only zstd19/ --only _zstd2_ --only _zst/
        done
    done ;;
q3q4)
    for r in $(seq 1 "$ROUNDS"); do
        # Q3: CeO2 at 1, 4, 16, 64 frames per call (input built once; the gate's harness)
        for f in 1 4 16 64; do
            gate q3-f$f "$r" SLABITZ_LIB="$(lib main)" -- --only /ceo2_ --frames "$f"
        done
        # Q4: fused vs unfused LZ4 over the LZ4 cells
        for fu in 1 0; do
            gate q4-fused$fu "$r" SLABITZ_LIB="$(lib main)" SLABITZ_LZ4_FUSED=$fu -- --only lz4
        done
    done ;;
bisect)
    # Q2 cause: the current kernel with MatchCopy branches switched off (wt/t04-var-*:
    # a = no aligned uint32 copy, b = no aligned uint4 copy, c = no offset-1 vector fill,
    # d = none of the three), against 439e2f2
    for r in $(seq 1 "$ROUNDS"); do
        for k in ${BISECT_LIBS:-439e2f2 main var-a var-b var-c var-d}; do
            gate bis-$k "$r" SLABITZ_LIB="$(lib $k)" -- --only pac-zstd/ --only zstd3/ \
                --only twin/silicon_zstd2 --only short_offsets
        done
    done ;;
*) echo "unknown: $WHAT" >&2; exit 2 ;;
esac
echo "done $(date -Is)"
