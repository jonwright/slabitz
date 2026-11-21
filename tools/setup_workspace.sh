#!/usr/bin/env bash
# Set up a self-contained slabitz workspace around a checkout.
#
#   WS=/path/to/workspace
#   mkdir -p "$WS" && git clone <slabitz repo> "$WS/slabitz"
#   "$WS/slabitz/tools/setup_workspace.sh" --workspace "$WS" [--examples DIR]
#   . "$WS/env.sh" && cd "$WS/slabitz" && pixi run test
#
# Creates, inside the workspace only:
#   WS/pixi/bin/pixi    pinned pixi release (SHA-256 checked)
#   WS/pixi/cache/      conda package cache
#   WS/tmp/             TMPDIR for builds and tools
#   WS/env.sh           source it to put pixi on PATH and point caches/TMPDIR at WS
#   WS/examples         optional symlink to read-only example HDF5 data
# and installs the pixi environment into the checkout's .pixi/.
#
# Nothing is written to $HOME or /tmp. Idempotent; safe to rerun.
# A workspace is not relocatable: on a move, clone again and rerun this script.
#
# The default workspace is the parent of the checkout, so a public clone at
# $HOME/git/slabitz uses $HOME/git as the workspace. To keep pixi, its cache,
# build temp files and .pixi on a large disk, pass --workspace (and optionally
# --examples) and link the external workspace into the checkout:
#
#   ln -s "$WS" "$REPO/workspace"          # gitignored by the repository
set -euo pipefail

usage() {
    sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
}

PIXI_VERSION="v0.84.1"
PIXI_ASSET="pixi-x86_64-unknown-linux-musl.tar.gz"
PIXI_SHA256="c81a16595e9e925fac3b12fe50cedbf415619a421ad2b72c1bba2a6ba6c2454a"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="${SLABITZ_WS:-$(cd "$REPO/.." && pwd)}"
EXAMPLES=""
WS_GIVEN=0
if [ -n "${SLABITZ_WS:-}" ]; then WS_GIVEN=1; fi

while [ $# -gt 0 ]; do
    case "$1" in
        --workspace)
            [ $# -ge 2 ] || { echo "--workspace needs a directory" >&2; exit 2; }
            WS="$2"; WS_GIVEN=1; shift 2 ;;
        --examples)
            [ $# -ge 2 ] || { echo "--examples needs a directory" >&2; exit 2; }
            EXAMPLES="$2"; shift 2 ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "unknown argument: $1 (try --help)" >&2; exit 2 ;;
    esac
done

mkdir -p "$WS"
WS="$(cd "$WS" && pwd)"
echo "workspace: $WS"
echo "checkout:  $REPO"

if [ "$WS_GIVEN" -eq 0 ]; then
    case "$REPO" in
        "$HOME"/*)
            echo "warning: the checkout is under \$HOME; pass --workspace to keep" \
                 "pixi, its cache and build temp files on a larger disk" >&2 ;;
    esac
fi

mkdir -p "$WS/pixi/bin" "$WS/pixi/cache" "$WS/pixi/download" "$WS/tmp"

if [ ! -x "$WS/pixi/bin/pixi" ]; then
    tarball="$WS/pixi/download/pixi-$PIXI_VERSION.tar.gz"
    if [ ! -f "$tarball" ]; then
        url="https://github.com/prefix-dev/pixi/releases/download/$PIXI_VERSION/$PIXI_ASSET"
        echo "downloading $url"
        curl -fL --retry 3 -o "$tarball.part" "$url"
        mv "$tarball.part" "$tarball"
    fi
    echo "$PIXI_SHA256  $tarball" | sha256sum -c -
    tar -xzf "$tarball" -C "$WS/pixi/bin"
fi
echo "pixi: $("$WS/pixi/bin/pixi" --version)"

if [ -n "$EXAMPLES" ]; then
    EXAMPLES="$(cd "$EXAMPLES" && pwd)"
    link="$WS/examples"
    if [ -e "$link" ] && [ ! -L "$link" ]; then
        echo "$link exists and is not a symlink; remove it or choose another workspace" >&2
        exit 1
    fi
    ln -sfn "$EXAMPLES" "$link"
    echo "examples:  $EXAMPLES"
fi

cat > "$WS/env.sh" <<'EOF'
# Source this to work on slabitz:  . /path/to/workspace/env.sh
# Written by slabitz/tools/setup_workspace.sh. Keeps pixi, its package cache
# and temp files inside this folder.
SLABITZ_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SLABITZ_WS
export PIXI_HOME="$SLABITZ_WS/pixi"
export PIXI_CACHE_DIR="$SLABITZ_WS/pixi/cache"
export RATTLER_CACHE_DIR="$SLABITZ_WS/pixi/cache"
export TMPDIR="$SLABITZ_WS/tmp"
export SLABITZ_EXAMPLES="$SLABITZ_WS/examples"
mkdir -p "$TMPDIR"
case ":$PATH:" in *":$PIXI_HOME/bin:"*) ;; *) export PATH="$PIXI_HOME/bin:$PATH" ;; esac
EOF
echo "wrote $WS/env.sh"

# shellcheck disable=SC1091
. "$WS/env.sh"
cd "$REPO"
pixi install
echo
echo "ready. Next:"
echo "  . $WS/env.sh && cd $REPO"
echo "  CUDA_ARCH=sm_XY pixi run test     # e.g. sm_50 for a GTX 960M, sm_89 for an L40S"
