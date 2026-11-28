#!/bin/bash
# Benchmark gate on an idle L40S (docs/dev/T03_bench_gate.md). Submit from the worktree:
#   sbatch bench/gate.sbatch                      measure, write the baseline, then --noise
#   sbatch bench/gate.sbatch --compare            any gate.py arguments instead; --compare
#                                                 runs on any L40S (another node than the
#                                                 baseline's: wall limit +20%, not +5%); add
#     -w $(SLABITZ_GATE_GPU=l40s python3 bench/gate.py --baseline-node) for the baseline's node
# The log goes to $SLABITZ_DEV_ROOT/results/gate-<job>.out (results/ when run outside the root
# needs SLABITZ_DEV_ROOT set; sbatch reads the --output line before this script runs, so the
# log path is relative to the submit directory: ../../results from wt/<task>).
#SBATCH --partition=low-gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=slabitz-gate
#SBATCH --output=../../results/gate-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
: "${SLABITZ_DEV_ROOT:?set SLABITZ_DEV_ROOT (source env.sh before sbatch)}"
. "$SLABITZ_DEV_ROOT/env.sh"
echo "node $(hostname) job ${SLURM_JOB_ID:-} commit $(git rev-parse --short HEAD) $(date -Is)"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader
sbz build
if [ $# -gt 0 ]; then
    timeout 6000 python3 bench/gate.py "$@"
else
    rc=0
    timeout 3000 python3 bench/gate.py --write-baseline || rc=$?
    timeout 3000 python3 bench/gate.py --noise || rc=$?
    exit $rc
fi
