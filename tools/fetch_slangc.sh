#!/usr/bin/env bash
# Fetch the Slang compiler (slangc v2026.22) into third_party/slang/.
#
# slangc is a large external binary, so it is not committed (third_party/slang/
# is gitignored). This script downloads the official release tarball from
# GitHub, checks its SHA-256 and unpacks it. Everything, including the
# download, stays inside third_party/slang/; nothing is written elsewhere.
#
# The kernels are only tested with this exact version: other slangc versions
# may produce different CUDA. The glibc-2.28 build runs on any x86_64 Linux
# with glibc >= 2.28.
#
# Usage:
#   tools/fetch_slangc.sh                         # download + verify + unpack
#   SLANG_TARBALL=/path/slang.tar.gz tools/fetch_slangc.sh   # offline: use a local copy
#                                                  # (same checksum is required)
# Idempotent: does nothing if third_party/slang/bin/slangc already exists.
# To replace an existing copy: rm -rf third_party/slang, then rerun.
set -euo pipefail

VERSION="2026.22"
ASSET="slang-${VERSION}-linux-x86_64-glibc-2.28.tar.gz"
URL="https://github.com/shader-slang/slang/releases/download/v${VERSION}/${ASSET}"
SHA256="35c765d20b69f6217b02848cdb82a5216bf03990044a8991ceb0b89a198a80fb"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$(cd "$HERE/.." && pwd)/third_party/slang"

if [ -x "$DEST/bin/slangc" ]; then
    echo "slangc already present: $DEST/bin/slangc ($("$DEST/bin/slangc" -version 2>&1 || true))"
    exit 0
fi

mkdir -p "$DEST/download"
TARBALL="${SLANG_TARBALL:-$DEST/download/$ASSET}"

if [ ! -f "$TARBALL" ]; then
    echo "downloading $URL"
    curl -fL --retry 3 -o "$TARBALL.part" "$URL"
    mv "$TARBALL.part" "$TARBALL"
fi

echo "checking sha256 of $TARBALL"
echo "$SHA256  $TARBALL" | sha256sum -c -

echo "unpacking into $DEST"
tar -xzf "$TARBALL" -C "$DEST"

if [ ! -x "$DEST/bin/slangc" ]; then
    echo "error: bin/slangc missing after unpacking $TARBALL" >&2
    exit 1
fi
got="$("$DEST/bin/slangc" -version 2>&1 || true)"
if [ "$got" != "$VERSION" ]; then
    echo "error: expected slangc $VERSION, got '$got'" >&2
    exit 1
fi
echo "slangc $VERSION ready: $DEST/bin/slangc"
