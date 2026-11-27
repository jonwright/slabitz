#!/usr/bin/env bash
# Run every per-kernel script (T01): the scripts that launch one kernel directly, or
# check one stage's CPU reference, and are not in `sbz test`. One line per script:
#   PASS     exit 0 and a "compared N ..." line with N > 0
#   FAIL     anything else (exit code, timeout, or nothing compared)
#   RETIRED  module docstring starts with "RETIRED (T01):" (not run)
# Exits nonzero on any FAIL. Each script gets the cubin of its kernel (from
# tests/kparams.py) under ${SLABITZ_KERNEL_DIR:-build}, and runs under `timeout 300`.
# Logs: ${SBZ_KS_LOG:-build/kernel_scripts}/<script>.log.
#
#   bash tests/run_kernel_scripts.sh        (or: sbz test-kernels)
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 2
KDIR="${SLABITZ_KERNEL_DIR:-build}"
LOG="${SBZ_KS_LOG:-build/kernel_scripts}"
mkdir -p "$LOG"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

# script:kernel[:ENV=VAL,...]  kernel = whose cubin the script takes ("-": CPU reference
# only, no kernel); the optional third field runs the script again in another mode
# (DSQ_MULTI: a multi-sequence payload, 7671 sequences; DSQ_SEQMS: the multi-stream kernel)
SCRIPTS="
compare_initfsetable:InitFseTable
compare_parse_blocks:ParseCompressedBlocks
compare_parse_blocks_cases:ParseCompressedBlocks
run_bo_chain:ComputeDestBlockOffsets
run_cps_chain:ComputePrefixSum
run_decodehw_chain:DecodeHuffmanWeights
run_dhw_chain:DecompressHuffmanWeights
run_dsq_chain:-
run_e2e_chain:-
run_exec_gpu_chain:ExecuteSequences
run_exec_multiblock:ExecuteSequences
run_fso_chain:-
run_fso_gpu_chain:FinaliseSequenceOffsets
run_frame_chain:ExecuteSequences
run_initresources_chain:InitResources
run_lit_chain:-
run_lit_gpu_chain:InitHuffmanTableAndDecompressLiterals
run_memsetmemcpy_chain:MemsetMemcpy
run_parse_blocks:ParseCompressedBlocks
run_prefixsum_chain:PrefixSum
run_propagatefse_chain:PropagateFseIndex
run_pso_chain:PrefixSequenceOffsets
run_seq_gpu_chain:DecompressSequences
smoke_initfsetable:InitFseTable
run_dsq_chain:-:DSQ_MULTI=1
run_e2e_chain:-:DSQ_MULTI=1
run_exec_gpu_chain:ExecuteSequences:DSQ_MULTI=1
run_fso_chain:-:DSQ_MULTI=1
run_fso_gpu_chain:FinaliseSequenceOffsets:DSQ_MULTI=1
run_seq_gpu_chain:DecompressSequences:DSQ_MULTI=1
run_seq_gpu_chain:DecompressSequencesMultiStream:DSQ_SEQMS=1
run_seq_gpu_chain:DecompressSequencesMultiStream:DSQ_SEQMS=1,DSQ_MULTI=1
"
# kernel launchers that `sbz test` runs (tools/sbz.sh cmd_test), not this script
IN_SBZ_TEST="run_unshuffle_slow run_unshuffle_warp"

npass=0 nfail=0 nret=0
# every kernel launcher in tests/ must be listed here or run by `sbz test`
for f in $(grep -l "fn(grid=" tests/*.py); do
    b="$(basename "$f" .py)"
    case " $IN_SBZ_TEST " in *" $b "*) continue ;; esac
    if ! printf '%s\n' $SCRIPTS | grep -q "^$b:"; then
        echo "FAIL     $b (launches a kernel but is not listed in tests/run_kernel_scripts.sh)"
        nfail=$((nfail + 1))
    fi
done

for entry in $SCRIPTS; do
    s="${entry%%:*}" rest="${entry#*:}"
    k="${rest%%:*}" envs=""
    [ "$rest" != "$k" ] && envs="${rest#*:}"
    name="$s${envs:+ [$envs]}"
    logf="$LOG/$s${envs:+.$(echo "$envs" | tr ',=' '__')}.log"
    if head -5 "tests/$s.py" | grep -q '^"""RETIRED (T01):'; then
        echo "RETIRED  $name"
        nret=$((nret + 1))
        continue
    fi
    args=()
    if [ "$k" != "-" ]; then
        c="$KDIR/$(python3 tests/kparams.py --cubin "$k")" || { echo "FAIL     $name (no cubin path for $k)"; nfail=$((nfail + 1)); continue; }
        args=("$c")
    fi
    t0=$SECONDS
    env $(echo "$envs" | tr ',' ' ') timeout 300 python3 "tests/$s.py" "${args[@]}" > "$logf" 2>&1
    rc=$?
    cmp="$(grep -E '^compared [0-9]+ ' "$logf" | tail -1)"
    n="$(echo "$cmp" | awk '{print $2}')"
    if [ $rc -eq 0 ] && [ -n "$n" ] && [ "$n" -gt 0 ]; then
        echo "PASS     $name ($cmp, $((SECONDS - t0)) s)"
        npass=$((npass + 1))
    else
        why="exit $rc"
        [ $rc -eq 124 ] && why="timeout 300 s"
        [ $rc -eq 0 ] && why="no nonzero 'compared N' line"
        echo "FAIL     $name ($why; log $logf)"
        nfail=$((nfail + 1))
    fi
done
echo "run_kernel_scripts: $npass PASS, $nfail FAIL, $nret RETIRED"
[ $nfail -eq 0 ]
