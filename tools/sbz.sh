#!/usr/bin/env bash
# sbz: the one task runner for slabitz (replaces the pixi tasks; pixi.toml
# calls this script, so each task has one definition).
#
#   sbz setup                    link third_party/slang to the shared compiler (or fetch it)
#   sbz build [--arch sm_XY]     kernels (linked from the shared cache when the
#                                sources match, else built and published) + build/libslabitz.so
#            [--local]           never link: build the kernels in this worktree
#   sbz build-sm35               sm_35 cubins (NVRTC 11.8) into build/sm_35
#   sbz build-fatbins            one fatbin per kernel into build/kernels
#   sbz test                     the main regression tests (each under timeout)
#   sbz test-kernels | pytest | gate    (added by tasks T01, T02, T03)
#   sbz where [--kernels DIR]    root, worktree, source key, kernel dirs, data paths
#   sbz key                      the source key (hash of the kernel sources)
#
# Root mode: SLABITZ_DEV_ROOT is set and the shared environment is active
# (. $SLABITZ_DEV_ROOT/env.sh). Kernel sets are cached in
# $SLABITZ_DEV_ROOT/kernels/<srckey>/{<arch>,sm_35,kernels}, write-once.
# Local mode (no SLABITZ_DEV_ROOT, e.g. a pixi shell or a plain clone): no
# cache, everything is built in this checkout's build/.
# See docs/dev/T-1_agent_root.md.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="${REPO#/gpfs/gd}"   # crunch sees /data, not /gpfs/gd
cd "$REPO"

ROOT="${SLABITZ_DEV_ROOT:-}"
ROOT_MODE=0
if [ -n "$ROOT" ] && [ -d "$ROOT/kernels" ]; then ROOT_MODE=1; fi

die() { echo "sbz: $*" >&2; exit 2; }

require_env() {
    if [ "$ROOT_MODE" = 1 ]; then
        [ "${SLABITZ_ENV_ACTIVE:-}" = 1 ] || die "the shared environment is not active: run  . $ROOT/env.sh"
        case "$(readlink -f "$REPO")" in
            "$(readlink -f "$ROOT")"/*) ;;
            *) [ "${SBZ_ALLOW_OUTSIDE:-}" = 1 ] ||
                die "$REPO is outside $ROOT (set SBZ_ALLOW_OUTSIDE=1 only if you are the maintainer)" ;;
        esac
    fi
    command -v python3 >/dev/null || die "no python3 on PATH"
}

key() { python3 "$REPO/tools/kernel_sources.py"; }

# "<sub>/<cubin>" for every kernel, from tools/build_all.py
kernel_files() {
    python3 - <<'EOF'
import sys
sys.path.insert(0, "tools")
from build_all import KERNEL_MAP
for _k, sub, cubin in KERNEL_MAP:
    print("%s/%s" % (sub, cubin))
EOF
}

header_id() { sed -n 's/^#define SBZ_KERNELS_ID "\(.*\)"/\1/p' "$REPO/src/sbz_kernels_gen.h"; }

in_cache() {  # is path $1 (resolved) inside the shared kernel cache?
    [ "$ROOT_MODE" = 1 ] || return 1
    case "$(readlink -f "$1" 2>/dev/null || echo "$1")" in "$(readlink -f "$ROOT")"/kernels/*) return 0 ;; esac
    return 1
}

drop_cache_links() {  # remove build/ entries that are links into the cache (never write through them)
    [ -d build ] || return 0
    local e
    for e in build/* build/sm_35 build/kernels; do
        if [ -L "$e" ] && in_cache "$e"; then rm -f "$e"; fi
    done
}

cmd_setup() {
    if [ "$ROOT_MODE" = 1 ]; then
        local want="${SLABITZ_SLANG_DIR:-$ROOT/shared/slang}"
        [ -x "$want/bin/slangc" ] || die "no slangc in $want (ask the maintainer: dev-root init)"
        mkdir -p third_party
        if [ -L third_party/slang ]; then
            [ "$(readlink -f third_party/slang)" = "$(readlink -f "$want")" ] || ln -sfn "$want" third_party/slang
        elif [ -e third_party/slang ]; then
            echo "sbz: third_party/slang is a real directory here; leaving it" >&2
        else
            ln -sfn "$want" third_party/slang
        fi
    else
        bash tools/fetch_slangc.sh
    fi
}

publish() {  # publish <staging dir> <key> <name>: atomic, write-once; returns 0 only if the set is in the cache
    local stage="$1" k="$2" name="$3" dest="$ROOT/kernels/$2/$3"
    mkdir -p "$ROOT/kernels/$k"
    if [ -e "$dest" ]; then rm -rf "$stage"; return 0; fi   # somebody published it meanwhile
    # move first, then make read-only (moving a directory needs write permission on it)
    if mv -T "$stage" "$dest"; then
        chmod -R a-w "$dest"
        return 0
    fi
    if [ -e "$dest" ]; then rm -rf "$stage"; return 0; fi
    echo "sbz: could not publish $dest" >&2
    return 1
}

stage_dir() { mktemp -d "$ROOT/tmp/publish.XXXXXX"; }

link_set() {  # link_set <cache set dir> <build dir>: link its entries into the build dir
    local set="$1" bdir="$2" e b
    mkdir -p "$bdir"
    for e in "$set"/*; do
        b="$(basename "$e")"
        [ "$b" = sbz_kernels_gen.h ] && continue
        rm -rf "${bdir:?}/$b"
        ln -s "$e" "$bdir/$b"
    done
}

sync_header() {  # the table in src/ must be the one the cached kernels were generated with
    local set="$1"
    [ -f "$set/sbz_kernels_gen.h" ] || return 0
    cmp -s "$set/sbz_kernels_gen.h" src/sbz_kernels_gen.h || cp "$set/sbz_kernels_gen.h" src/sbz_kernels_gen.h
}

build_local() {  # build_local <arch>: slangc + nvcc for every kernel into build/
    local arch="$1"
    drop_cache_links
    echo "kernels: building for $arch (source key $(key))"
    CUDA_ARCH="$arch" python3 tools/build_all.py
    key > build/sbz_source.key   # which sources build/ was generated from (have_local_cu)
}

publish_arch() {  # publish build/<sub>/<cubin> as <key>/<arch>
    local arch="$1" k stage f
    [ "$ROOT_MODE" = 1 ] || return 0
    k="$(key)"
    [ -e "$ROOT/kernels/$k/$arch" ] && return 0
    stage="$(stage_dir)"
    while read -r f; do
        mkdir -p "$stage/$(dirname "$f")"
        cp "build/$f" "$stage/$f"
    done < <(kernel_files)
    cp build/sbz_kernels.id "$stage/sbz_kernels.id"
    cp src/sbz_kernels_gen.h "$stage/sbz_kernels_gen.h"
    publish "$stage" "$k" "$arch" || die "publish failed"
    echo "kernels: published $k/$arch"
}

cmd_build() {
    local arch="${CUDA_ARCH:-sm_89}" local_only=0
    while [ $# -gt 0 ]; do
        case "$1" in
            --arch) arch="$2"; shift 2 ;;
            --local) local_only=1; shift ;;
            *) die "build: unknown argument $1" ;;
        esac
    done
    require_env
    cmd_setup
    mkdir -p build
    local k set
    k="$(key)"
    set="$ROOT/kernels/$k/$arch"
    if [ "$ROOT_MODE" = 1 ] && [ "$local_only" = 0 ] && [ -d "$set" ]; then
        link_set "$set" build
        sync_header "$set"
        [ "$(cat build/sbz_kernels.id)" = "$(header_id)" ] ||
            die "cached set $set has id $(cat build/sbz_kernels.id), src/sbz_kernels_gen.h has $(header_id)"
        echo "kernels: linked $k ($arch)"
    else
        build_local "$arch"
        publish_arch "$arch"
        echo "kernels: built $k ($arch)"
    fi
    make
}

# Generated .cu in build/ from the current sources? Old .cu must never be compiled
# and published under a new key (cache sets are write-once).
stamp_current() { [ -f "$1/sbz_source.key" ] && [ "$(cat "$1/sbz_source.key")" = "$(key)" ]; }
have_local_cu() { [ -f build/exec/ExecuteSequences.cu ] && [ -f build/sbz_kernels.id ] && stamp_current build; }

cmd_build_sm35() {
    require_env
    cmd_setup
    local k set stage nvrtc f
    k="$(key)"; set="$ROOT/kernels/$k/sm_35"
    if [ "$ROOT_MODE" = 1 ] && [ -d "$set" ]; then
        mkdir -p build
        rm -rf build/sm_35
        ln -s "$set" build/sm_35
        echo "kernels: linked $k (sm_35)"
        return 0
    fi
    have_local_cu || build_local "${CUDA_ARCH:-sm_89}"
    nvrtc="${SLABITZ_KEPLER_NVRTC:-${CONDA_PREFIX:-}/lib/libnvrtc.so}"
    [ -f "$nvrtc" ] || die "no NVRTC 11.8 (set SLABITZ_KEPLER_NVRTC)"
    if [ -L build/sm_35 ]; then rm -f build/sm_35; fi
    # libnvrtc 11.8 finds its libnvrtc-builtins next to itself only through the loader path
    LD_LIBRARY_PATH="$(dirname "$nvrtc")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        python3 tools/build_kubins.py --arch sm_35 --out build/sm_35 --nvrtc "$nvrtc"
    cp build/sbz_source.key build/sm_35/sbz_source.key
    if [ "$ROOT_MODE" = 1 ] && [ ! -e "$set" ]; then
        stage="$(stage_dir)"
        while read -r f; do
            mkdir -p "$stage/$(dirname "$f")"
            cp "build/sm_35/$f" "$stage/$f"
        done < <(kernel_files)
        cp build/sm_35/sbz_kernels.id "$stage/sbz_kernels.id"
        publish "$stage" "$k" sm_35 || die "publish failed"
        echo "kernels: published $k/sm_35"
    fi
}

cmd_build_fatbins() {
    require_env
    cmd_setup
    local k set stage
    k="$(key)"; set="$ROOT/kernels/$k/kernels"
    if [ "$ROOT_MODE" = 1 ] && [ -d "$set" ]; then
        mkdir -p build
        rm -rf build/kernels
        ln -s "$set" build/kernels
        echo "kernels: linked $k (fatbins)"
        return 0
    fi
    have_local_cu || build_local "${CUDA_ARCH:-sm_89}"
    # reuse build/sm_35 only if built from the current sources; otherwise
    # cmd_build_sm35 links the cached set for this key or builds it
    { [ -f build/sm_35/sbz_kernels.id ] && stamp_current build/sm_35; } || cmd_build_sm35
    if [ -L build/kernels ]; then rm -f build/kernels; fi
    python3 tools/build_fatbins.py
    if [ "$ROOT_MODE" = 1 ] && [ ! -e "$set" ]; then
        stage="$(stage_dir)"
        cp -r build/kernels/. "$stage/"
        publish "$stage" "$k" kernels || die "publish failed"
        echo "kernels: published $k/kernels"
    fi
}

run_t() {  # run_t <script> [args]: one test under timeout, named in the output
    echo "== $*"
    timeout "${SBZ_TEST_TIMEOUT:-900}" python3 "$@"
}

cmd_test() {
    cmd_build "$@"
    run_t tests/run_kernel_id.py
    run_t tests/run_host_walk.py
    run_t tests/run_chain_stage01.py
    run_t tests/run_chain_stage2.py
    run_t tests/run_chain_frames.py
    run_t tests/run_bshuf_chain.py
    run_t tests/run_bslz4_chain.py
    run_t tests/run_zstd_async.py
    run_t tests/run_bslz4_device.py
    run_t tests/run_unshuffle_slow.py
    run_t tests/run_unshuffle_warp.py
    run_t tests/run_synth_decode.py
    echo "sbz test: all passed"
}

cmd_where() {
    require_env
    local k a
    k="$(key)"
    if [ "${1:-}" = "--kernels" ]; then
        a="${2:?where --kernels needs sm_XY or kernels}"
        echo "$ROOT/kernels/$k/$a"
        return 0
    fi
    echo "root:      ${ROOT:-<none: local mode>}"
    echo "worktree:  $REPO"
    echo "key:       $k"
    for a in sm_89 sm_35 kernels; do
        if [ "$ROOT_MODE" = 1 ] && [ -d "$ROOT/kernels/$k/$a" ]; then echo "cached:    $a"; else echo "cached:    $a (no)"; fi
    done
    echo "examples:  ${SLABITZ_EXAMPLES:-<unset>}"
    echo "synth:     ${SLABITZ_SYNTH:-<unset>}"
    echo "pac lib:   ${PAC_LIB:-<unset>}"
    echo "slangc:    $(readlink -f third_party/slang 2>/dev/null || echo '<not set up>')"
}

optional() {  # optional <task> <script> [args]: tasks added later
    local task="$1" script="$2"; shift 2
    [ -f "$script" ] || die "'$task' is not available yet: $script does not exist (see docs/dev/README.md)"
    cmd_build
    timeout "${SBZ_TEST_TIMEOUT:-3600}" python3 "$script" "$@"
}

sub="${1:-}"
[ $# -gt 0 ] && shift
case "$sub" in
    setup) require_env; cmd_setup ;;
    build) cmd_build "$@" ;;
    build-sm35) cmd_build_sm35 ;;
    build-fatbins) cmd_build_fatbins ;;
    test) cmd_test "$@" ;;
    test-kernels) optional test-kernels tests/run_kernel_args.py "$@" ;;
    pytest) require_env; cmd_build; timeout "${SBZ_TEST_TIMEOUT:-3600}" python3 -m pytest tests "$@" ;;
    gate) optional gate bench/gate.py "$@" ;;
    where) cmd_where "$@" ;;
    key) require_env; key ;;
    ""|-h|--help) sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//' ;;
    *) die "unknown subcommand '$sub' (try sbz --help)" ;;
esac
