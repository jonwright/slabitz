#!/usr/bin/env bash
# dev-root: administer the shared agent root $SLABITZ_DEV_ROOT (for the user or Claude
# as reviewer; workers do not run it). Design: docs/dev/T-1_agent_root.md.
#
#   dev-root init [--from REPO] [--main-ref REF] [--old-ws DIR] [--ext-src DIR]
#                                create or complete the root (idempotent)
#   dev-root update-main [REF]   fetch, move main/ to origin/main, publish its kernels
#   dev-root update-env          reinstall env/ from the pixi.lock at the main ref
#   dev-root new <task> [base]   worktree wt/<task> on branch <task>, tmp/<task>, built
#   dev-root rm <task> [--force] remove the worktree, branch and tmp (merged only, or --force)
#   dev-root check               health check: one OK/FAIL line per item
#
# init options (also environment variables):
#   --from REPO      where repo.git is cloned from: a path or URL (default: the GitHub origin).
#                    Its origin URL is set back to GitHub, and its branches become origin/*.
#   --main-ref REF   what main/ and env/ are checked out at (default origin/main). Before
#                    this task is merged, use origin/t-1-agent-root.
#   --old-ws DIR     the old single workspace: its pixi binary, package cache and slangc
#                    tarball are reused, so nothing is downloaded again.
#   --ext-src DIR    directory holding local clones of pac-zstd and bslz4_to_sparse-dev,
#                    cloned into ext/ (their origin URLs are kept).
# pixi is used here and only here (admin/pixi), never on the agents' PATH.
set -euo pipefail

ROOT="${SLABITZ_DEV_ROOT:?set SLABITZ_DEV_ROOT (e.g. in ~/.bashrc)}"
GITHUB_URL="https://github.com/jonwright/slabitz.git"
MAIN_REF="${DEV_ROOT_MAIN_REF:-$(cat "$ROOT/.main-ref" 2>/dev/null || echo origin/main)}"
OLD_WS="${DEV_ROOT_OLD_WS:-}"
EXT_SRC="${DEV_ROOT_EXT_SRC:-}"
FROM="${DEV_ROOT_FROM:-$GITHUB_URL}"
PIXI_VERSION="v0.84.1"
PIXI_ASSET="pixi-x86_64-unknown-linux-musl.tar.gz"
PIXI_SHA256="c81a16595e9e925fac3b12fe50cedbf415619a421ad2b72c1bba2a6ba6c2454a"
SLANG_ASSET="slang-2026.22-linux-x86_64-glibc-2.28.tar.gz"

die() { echo "dev-root: $*" >&2; exit 2; }
say() { echo "== $*"; }
repo() { git -C "$ROOT/repo.git" "$@"; }

# Real data: name in data/real -> source file. Sources that are missing are skipped with a note.
EX=/data/id11/jon/hdftest
REAL_FILES="
ceo2_lz4.h5=$EX/bslz4_to_sparse_examples/ceo2_lz4.h5
ceo2_lz4mask.h5=$EX/bslz4_to_sparse_examples/ceo2_lz4mask.h5
ceo2_u32_zstd2_256k.h5=$EX/bslz4_to_sparse_examples/ceo2_u32_zstd2_256k.h5
silicon_lz4.h5=$EX/bslz4_to_sparse_examples/silicon_lz4.h5
silicon_lz4mask.h5=$EX/bslz4_to_sparse_examples/silicon_lz4mask.h5
silicon_u16_zstd2_256k.h5=$EX/bslz4_to_sparse_examples/silicon_u16_zstd2_256k.h5
mask.h5=$EX/bslz4_to_sparse_examples/mask.h5
geometry.par=$EX/bslz4_to_sparse_examples/geometry.par
examples_README.md=$EX/bslz4_to_sparse_examples/README.md
thaumatin_9_1_000001.h5=$EX/zenodo_2412270_dls_i04_eiger2_16m_thaumatin/thaumatin_9_1_000001.h5
thaumatin_9_1_000002.h5=$EX/zenodo_2412270_dls_i04_eiger2_16m_thaumatin/thaumatin_9_1_000002.h5
thaumatin_9_1_000003.h5=$EX/zenodo_2412270_dls_i04_eiger2_16m_thaumatin/thaumatin_9_1_000003.h5
thaumatin_9_1_master.h5=$EX/zenodo_2412270_dls_i04_eiger2_16m_thaumatin/thaumatin_9_1_master.h5
thaumatin_9_1_mask.h5=$EX/zenodo_2412270_dls_i04_eiger2_16m_thaumatin/thaumatin_9_1_mask.h5
thaumatin_md5sums.txt=$EX/zenodo_2412270_dls_i04_eiger2_16m_thaumatin/md5sums.txt
eiger4m_u32.h5=$EX/eiger4m_u32.h5
big_eiger_0000.h5=$EX/big_eiger_0000.h5
kevlar.h5=$EX/kevlar.h5
hplc.h5=$EX/hplc.h5
Calibration_Al2O3.h5=$EX/Calibration_Al2O3.h5
"

pixi_env() {
    export PIXI_HOME="$ROOT/admin/pixi" PIXI_CACHE_DIR="$ROOT/admin/pixi/cache" \
           RATTLER_CACHE_DIR="$ROOT/admin/pixi/cache" TMPDIR="$ROOT/tmp/admin"
    mkdir -p "$TMPDIR"
    export PATH="$ROOT/admin/pixi/bin:$PATH"
}

# A hook for the shared env: only the env's own PATH entries (never the generating user's
# PATH, never pixi), no PIXI_* variables and no pixi() function.
write_hook() {  # write_hook <env name> <out file>
    ( pixi_env; cd "$ROOT/env"; pixi shell-hook -e "$1" ) | python3 -c '
import re, sys
root = sys.argv[1]
out, skip = [], False
for line in sys.stdin.read().split("\n"):
    if skip:
        if line == "}":
            skip = False
        continue
    if line.startswith("pixi()") or line.startswith("pixi ()"):
        skip = True
        continue
    if line.startswith("export PIXI_") or "PIXI_EXE" in line:
        continue
    if line.startswith("export PATH="):
        keep = [x for x in line[len("export PATH="):].strip().strip(chr(34)).split(":") if x.startswith(root)]
        line = "export PATH=\"%s:$PATH\"" % ":".join(keep)
    out.append(line)
print("\n".join(out))
' "$(readlink -f "$ROOT/env")" > "$2"
}

in_env() {  # run a command with the shared environment, from directory $1
    local d="$1"; shift
    ( cd "$d" && export SLABITZ_DEV_ROOT="$ROOT" && . "$ROOT/env.sh" && "$@" )
}

# ---------------------------------------------------------------- init steps

init_layout() {
    say "layout"
    mkdir -p "$ROOT"/{bin,admin/pixi,shared,kernels,data/real,data/synth,data/compat,toolchains,ext,wt,tmp/admin,tmp/misc,results}
}

init_pixi() {
    say "pixi (admin only)"
    if [ ! -x "$ROOT/admin/pixi/bin/pixi" ]; then
        mkdir -p "$ROOT/admin/pixi/bin" "$ROOT/admin/pixi/download"
        if [ -n "$OLD_WS" ] && [ -x "$OLD_WS/pixi/bin/pixi" ] &&
           [ "$("$OLD_WS/pixi/bin/pixi" --version)" = "pixi ${PIXI_VERSION#v}" ]; then
            cp "$OLD_WS/pixi/bin/pixi" "$ROOT/admin/pixi/bin/pixi"
        else
            local t="$ROOT/admin/pixi/download/pixi-$PIXI_VERSION.tar.gz"
            curl -fL --retry 3 -o "$t.part" "https://github.com/prefix-dev/pixi/releases/download/$PIXI_VERSION/$PIXI_ASSET"
            mv "$t.part" "$t"
            echo "$PIXI_SHA256  $t" | sha256sum -c -
            tar -xzf "$t" -C "$ROOT/admin/pixi/bin"
        fi
    fi
    if [ ! -d "$ROOT/admin/pixi/cache/pkgs" ] && [ -n "$OLD_WS" ] && [ -d "$OLD_WS/pixi/cache" ]; then
        echo "seeding the package cache from $OLD_WS/pixi/cache (hard links, no download)"
        mkdir -p "$ROOT/admin/pixi/cache"
        cp -al "$OLD_WS/pixi/cache/." "$ROOT/admin/pixi/cache/" 2>/dev/null ||
            cp -a "$OLD_WS/pixi/cache/." "$ROOT/admin/pixi/cache/"
    fi
    ( pixi_env; pixi --version )
}

init_repo() {
    say "repo.git, main/, env/"
    if [ ! -d "$ROOT/repo.git" ]; then
        git clone --bare --quiet "$FROM" "$ROOT/repo.git"
        # branches of the source become origin/*, and origin is GitHub
        local b
        for b in $(repo for-each-ref --format='%(refname:short)' refs/heads); do
            repo update-ref "refs/remotes/origin/$b" "refs/heads/$b"
            repo update-ref -d "refs/heads/$b"
        done
        repo symbolic-ref HEAD refs/heads/unborn
        repo remote remove origin 2>/dev/null || true
        repo remote add origin "$GITHUB_URL"
        repo config remote.origin.fetch '+refs/heads/*:refs/remotes/origin/*'
    fi
    repo rev-parse --verify --quiet "$MAIN_REF^{commit}" >/dev/null || die "ref $MAIN_REF not in repo.git (use --main-ref)"
    local d
    for d in main env; do
        if [ ! -d "$ROOT/$d" ]; then
            repo worktree add --detach "$ROOT/$d" "$MAIN_REF" >/dev/null
        fi
    done
}

init_env() {
    say "shared environment (env/): default and kepler"
    ( pixi_env; cd "$ROOT/env"
      pixi install --locked
      pixi install --locked -e kepler
      true )
    write_hook default "$ROOT/env/activate.sh"
    write_hook kepler "$ROOT/env/activate-kepler.sh"
}

init_slang() {
    say "slangc"
    if [ ! -x "$ROOT/shared/slang/bin/slangc" ]; then
        local tb=""
        [ -n "$OLD_WS" ] && tb="$OLD_WS/slabitz/third_party/slang/download/$SLANG_ASSET"
        ( cd "$ROOT/env"
          if [ -f "$tb" ]; then SLANG_TARBALL="$tb" bash tools/fetch_slangc.sh; else bash tools/fetch_slangc.sh; fi )
        mv "$ROOT/env/third_party/slang" "$ROOT/shared/slang"
        ln -s "$ROOT/shared/slang" "$ROOT/env/third_party/slang"
    fi
    "$ROOT/shared/slang/bin/slangc" -version
}

init_scripts() {
    say "env.sh, bin/, AGENT_START.md"
    cat > "$ROOT/env.sh" <<'EOF'
# slabitz agent environment. Source it:  . $SLABITZ_DEV_ROOT/env.sh
# Works in a login shell, a Slurm job and on crunch. Does not put pixi on PATH.
_sbz_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${SLABITZ_DEV_ROOT:-}" ] && [ "$SLABITZ_DEV_ROOT" != "$_sbz_here" ]; then
    echo "env.sh: SLABITZ_DEV_ROOT=$SLABITZ_DEV_ROOT but this file is in $_sbz_here; using $_sbz_here" >&2
fi
export SLABITZ_DEV_ROOT="$_sbz_here"
unset _sbz_here
# the conda activation scripts are not safe under `set -u`
case "$-" in *u*) _sbz_u=1; set +u ;; *) _sbz_u=0 ;; esac
# shellcheck disable=SC1091
. "$SLABITZ_DEV_ROOT/env/activate.sh"
[ "$_sbz_u" = 1 ] && set -u
unset _sbz_u
export SLABITZ_ENV_ACTIVE=1
export SLABITZ_EXAMPLES="$SLABITZ_DEV_ROOT/data/real"
export SLABITZ_SYNTH="$SLABITZ_DEV_ROOT/data/synth/current"
export PAC_LIB="$SLABITZ_DEV_ROOT/ext/pac-zstd/build/libpac.so"
export SLABITZ_SLANG_DIR="$SLABITZ_DEV_ROOT/shared/slang"
export SLABITZ_KEPLER_NVRTC="$SLABITZ_DEV_ROOT/env/.pixi/envs/kepler/lib/libnvrtc.so"
export PIP_NO_INDEX=1 PIP_REQUIRE_VIRTUALENV=1
case ":$PATH:" in *":$SLABITZ_DEV_ROOT/bin:"*) ;; *) export PATH="$SLABITZ_DEV_ROOT/bin:$PATH" ;; esac
# TMPDIR of the worktree we are in (tmp/misc elsewhere); the wrappers in bin/ redo this per call
_sbz_top="$(git rev-parse --show-toplevel 2>/dev/null || true)"
case "$_sbz_top" in "$SLABITZ_DEV_ROOT"/wt/*) _sbz_task="$(basename "$_sbz_top")" ;; *) _sbz_task=misc ;; esac
export TMPDIR="$SLABITZ_DEV_ROOT/tmp/$_sbz_task" XDG_CACHE_HOME="$SLABITZ_DEV_ROOT/tmp/$_sbz_task/cache"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
unset _sbz_top _sbz_task
EOF
    cat > "$ROOT/bin/_task_env" <<'EOF'
# sourced by the wrappers: TMPDIR for the worktree we are in
_top="$(git rev-parse --show-toplevel 2>/dev/null || true)"
case "$_top" in "$SLABITZ_DEV_ROOT"/wt/*) _task="$(basename "$_top")" ;; *) _task=misc ;; esac
export TMPDIR="$SLABITZ_DEV_ROOT/tmp/$_task" XDG_CACHE_HOME="$SLABITZ_DEV_ROOT/tmp/$_task/cache"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
EOF
    cat > "$ROOT/bin/sbz" <<'EOF'
#!/usr/bin/env bash
# sbz: runs tools/sbz.sh of the worktree you are in
[ -n "${SLABITZ_DEV_ROOT:-}" ] || { echo "sbz: run  . <root>/env.sh  first" >&2; exit 2; }
top="$(git rev-parse --show-toplevel 2>/dev/null)" ||
    { echo "sbz: not inside a git worktree (cd $SLABITZ_DEV_ROOT/wt/<task>)" >&2; exit 2; }
# shellcheck disable=SC1091
. "$SLABITZ_DEV_ROOT/bin/_task_env"
exec bash "$top/tools/sbz.sh" "$@"
EOF
    cat > "$ROOT/bin/gpu-run" <<'EOF'
#!/usr/bin/env bash
# gpu-run [--time HH:MM:SS] cmd...: run a command on an idle L40S through Slurm (low-gpu),
# in the same worktree and environment; on crunch, run it as it is. Timing numbers for
# reports and baselines are made only this way.
set -euo pipefail
T=01:00:00
if [ "${1:-}" = "--time" ]; then T="$2"; shift 2; fi
[ $# -gt 0 ] || { echo "usage: gpu-run [--time HH:MM:SS] cmd..." >&2; exit 2; }
case "$(hostname)" in crunch*) exec "$@" ;; esac
exec srun --partition=low-gpu --gres=gpu:nvidia_l40s:1 --time="$T" --chdir="$PWD" \
     --export=ALL --job-name=slabitz-gpu "$@"
EOF
    cat > "$ROOT/bin/dev-root" <<'EOF'
#!/usr/bin/env bash
# dev-root: administration of the shared root (see docs/dev/T-1_agent_root.md)
[ -n "${SLABITZ_DEV_ROOT:-}" ] || { echo "dev-root: run  . <root>/env.sh  first" >&2; exit 2; }
exec bash "$SLABITZ_DEV_ROOT/main/tools/dev_root.sh" "$@"
EOF
    chmod +x "$ROOT/bin/sbz" "$ROOT/bin/gpu-run" "$ROOT/bin/dev-root"
    sed "s|@ROOT@|$ROOT|g" "$ROOT/main/tools/AGENT_START.md.in" > "$ROOT/AGENT_START.md"
}

init_data() {
    say "real data links, manifest, inventory"
    local d="$ROOT/data/real" line name src
    for line in $REAL_FILES; do
        name="${line%%=*}"; src="${line#*=}"
        if [ -e "$src" ]; then ln -sfn "$src" "$d/$name"; else echo "note: missing $src (skipped)"; fi
    done
    if [ ! -s "$d/MANIFEST.sha256" ]; then
        ( cd "$d"
          ls | grep -v -E '^(MANIFEST.sha256|INVENTORY.md)$' |
              xargs -P 4 -I{} sha256sum -- {} | sort -k2 > MANIFEST.sha256 )
    fi
    in_env "$ROOT/main" python3 tools/data_inventory.py "$d"
}

init_ext() {
    say "ext/: pac-zstd, bslz4_to_sparse"
    local name src
    for pair in "pac-zstd:pac-zstd" "bslz4_to_sparse:bslz4_to_sparse-dev"; do
        name="${pair%%:*}"; src="${pair#*:}"
        if [ ! -d "$ROOT/ext/$name" ]; then
            if [ -n "$EXT_SRC" ] && [ -d "$EXT_SRC/$src/.git" ]; then
                git clone --quiet "$EXT_SRC/$src" "$ROOT/ext/$name"
                git -C "$ROOT/ext/$name" remote set-url origin "$(git -C "$EXT_SRC/$src" remote get-url origin)"
            else
                echo "note: no source for ext/$name (pass --ext-src)"; continue
            fi
        fi
    done
    if [ -d "$ROOT/ext/pac-zstd" ] && [ ! -f "$ROOT/ext/pac-zstd/build/libpac.so" ]; then
        # submodules (zstd, lz4, kcb) from GitHub: admin step, network. BUILD=build because
        # the conda compiler activation exports its own BUILD variable.
        git -C "$ROOT/ext/pac-zstd" submodule update --init --quiet
        in_env "$ROOT/ext/pac-zstd" make BUILD=build
    fi
}

init_kernels() {
    say "kernels for main (sm_89, sm_35, fatbins)"
    in_env "$ROOT/main" bash tools/sbz.sh build
    in_env "$ROOT/main" bash tools/sbz.sh build-sm35
    in_env "$ROOT/main" bash tools/sbz.sh build-fatbins
}

cmd_init() {
    while [ $# -gt 0 ]; do
        case "$1" in
            --from) FROM="$2"; shift 2 ;;
            --main-ref) MAIN_REF="$2"; shift 2 ;;
            --old-ws) OLD_WS="$2"; shift 2 ;;
            --ext-src) EXT_SRC="$2"; shift 2 ;;
            *) die "init: unknown argument $1" ;;
        esac
    done
    init_layout
    echo "$MAIN_REF" > "$ROOT/.main-ref"
    init_pixi
    init_repo
    init_env
    init_slang
    init_scripts
    init_data
    init_ext
    init_kernels
    say "init done; run: dev-root check"
}

# ---------------------------------------------------------------- other commands

cmd_update_main() {
    local ref="${1:-$MAIN_REF}"
    repo fetch --quiet origin || echo "note: fetch failed; using what repo.git has"
    git -C "$ROOT/main" checkout --quiet --detach "$ref"
    in_env "$ROOT/main" bash tools/sbz.sh build
    in_env "$ROOT/main" bash tools/sbz.sh build-sm35
    in_env "$ROOT/main" bash tools/sbz.sh build-fatbins
}

cmd_update_env() {
    git -C "$ROOT/env" checkout --quiet --detach "$MAIN_REF"
    ( pixi_env; cd "$ROOT/env"
      pixi install --locked
      pixi install --locked -e kepler )
    write_hook default "$ROOT/env/activate.sh"
    write_hook kepler "$ROOT/env/activate-kepler.sh"
}

cmd_new() {
    local task="${1:?usage: dev-root new <task> [base]}" base="${2:-origin/main}"
    case "$task" in *[!A-Za-z0-9._-]*|"") die "bad task name $task" ;; esac
    [ ! -e "$ROOT/wt/$task" ] || die "wt/$task exists"
    repo worktree add -b "$task" "$ROOT/wt/$task" "$base" >/dev/null
    mkdir -p "$ROOT/tmp/$task"
    in_env "$ROOT/wt/$task" bash tools/sbz.sh setup
    in_env "$ROOT/wt/$task" bash tools/sbz.sh build
    echo "$ROOT/wt/$task"
}

cmd_rm() {
    local task="${1:?usage: dev-root rm <task> [--force]}" force="${2:-}"
    [ -d "$ROOT/wt/$task" ] || die "no wt/$task"
    if [ "$force" != "--force" ]; then
        repo branch --merged origin/main --format='%(refname:short)' | grep -qx "$task" ||
            die "branch $task is not merged into origin/main (use --force)"
    fi
    repo worktree remove --force "$ROOT/wt/$task"
    repo branch -D "$task" >/dev/null 2>&1 || true
    rm -rf "$ROOT/tmp/$task"
    echo "removed wt/$task, branch $task, tmp/$task"
}

FAILS=0
ok() { echo "OK    $*"; }
bad() { echo "FAIL  $*"; FAILS=$((FAILS + 1)); }

cmd_check() {
    local out k kid a
    # 1. env activates, packages import, pixi not on PATH
    if out="$(bash -c ". '$ROOT/env.sh' && python3 -c 'import numpy, h5py, hdf5plugin, zstandard, pytest, lz4; print(\"imports ok\")' && ! command -v pixi" 2>&1)"; then
        ok "environment activates; numpy h5py hdf5plugin zstandard pytest lz4 import; pixi not on PATH"
    else bad "environment: $out"; fi
    # 2. nvcc 12.6, kepler NVRTC 11.8
    if out="$(bash -c ". '$ROOT/env.sh' && nvcc --version" 2>&1)" && echo "$out" | grep -q "release 12.6"; then
        ok "nvcc 12.6"; else bad "nvcc is not 12.6: $out"; fi
    if out="$(bash -c ". '$ROOT/env.sh' && python3 - <<'PY'
import ctypes, os
l = ctypes.CDLL(os.environ['SLABITZ_KEPLER_NVRTC'])
a, b = ctypes.c_int(), ctypes.c_int()
l.nvrtcVersion(ctypes.byref(a), ctypes.byref(b))
print('%d.%d' % (a.value, b.value))
PY" 2>&1)" && [ "$out" = "11.8" ]; then ok "kepler NVRTC 11.8"; else bad "kepler NVRTC: $out"; fi
    # 3. slangc
    out="$("$ROOT/shared/slang/bin/slangc" -version 2>&1 || true)"
    if [ "$out" = "2026.22" ]; then ok "slangc $out"; else bad "slangc is '$out', expected 2026.22"; fi
    # 4. data manifest
    if out="$(cd "$ROOT/data/real" && sha256sum -c --quiet MANIFEST.sha256 2>&1)"; then
        ok "data/real: $(wc -l < "$ROOT/data/real/MANIFEST.sha256") files verify against MANIFEST.sha256"
    else bad "data/real manifest: $out"; fi
    # 5. main/ at the ref
    if [ "$(git -C "$ROOT/main" rev-parse HEAD)" = "$(repo rev-parse "$MAIN_REF^{commit}")" ]; then
        ok "main/ is at $MAIN_REF ($(git -C "$ROOT/main" rev-parse --short HEAD))"
    else bad "main/ is not at $MAIN_REF"; fi
    # 6. kernel sets of main
    k="$(in_env "$ROOT/main" python3 tools/kernel_sources.py)"
    kid="$(sed -n 's/^#define SBZ_KERNELS_ID "\(.*\)"/\1/p' "$ROOT/main/src/sbz_kernels_gen.h")"
    for a in sm_89 sm_35 kernels; do
        if [ -f "$ROOT/kernels/$k/$a/sbz_kernels.id" ] && [ "$(cat "$ROOT/kernels/$k/$a/sbz_kernels.id")" = "$kid" ]; then
            ok "kernels/$k/$a has id $kid"
        else bad "kernels/$k/$a missing or its id differs from SBZ_KERNELS_ID $kid"; fi
    done
    # 7. libpac
    if out="$(bash -c ". '$ROOT/env.sh' && python3 -c 'import ctypes, os; ctypes.CDLL(os.environ[\"PAC_LIB\"]); print(1)'" 2>&1)"; then
        ok "PAC_LIB loads"; else bad "PAC_LIB: $out"; fi
    [ "$FAILS" = 0 ] && echo "dev-root check: all OK" || { echo "dev-root check: $FAILS failed"; return 1; }
}

sub="${1:-}"
[ $# -gt 0 ] && shift
case "$sub" in
    init) cmd_init "$@" ;;
    update-main) cmd_update_main "$@" ;;
    update-env) cmd_update_env ;;
    new) cmd_new "$@" ;;
    rm) cmd_rm "$@" ;;
    check) cmd_check ;;
    ""|-h|--help) sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//' ;;
    *) die "unknown subcommand '$sub'" ;;
esac
