#!/usr/bin/env python3
"""The one list of files that shape the compiled kernels.

Used by tools/gen_kernel_table.py (SBZ_KERNELS_ID) and by tools/sbz.sh (the
source key that names a published kernel set in $SLABITZ_DEV_ROOT/kernels/).

  SBZ_KERNELS_ID  = hash of ID_SOURCES + the generated launch table. The
                    library refuses a kernel directory with another ID.
  source key      = hash of ID_SOURCES + KEY_EXTRA, computable WITHOUT slangc
                    (so a worktree can find an existing build before building).
                    Different tools or scripts give a different key even when
                    the kernel text is the same.

Content only (sorted relative paths and bytes): the same in every clone.

Usage: python3 tools/kernel_sources.py [--key | --list]
"""
import hashlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Everything that shapes the compiled kernels (keeps the historical order and
# content: SBZ_KERNELS_ID must not change when this list is only moved).
ID_SOURCES = ["kernels/*.slang", "kernels/lib/*", "third_party/directstorage_zstdgpu/**/*",
              "tools/transform_ds_for_slang.py", "tools/slang_compat.h",
              "tools/slang_cuda_prevolta.h", "tools/fetch_slangc.sh"]

# Added for the source key only: the build scripts and the table generator
# decide what the build produces, so a change must give a new key.
KEY_EXTRA = ["tools/gen_kernel_table.py", "tools/build_all.py", "tools/build_kubins.py",
             "tools/build_fatbins.py", "tools/kernel_sources.py"]


def files(patterns):
    return sorted({p for pat in patterns for p in ROOT.glob(pat) if p.is_file()})


def hash_files(patterns, h=None):
    h = h or hashlib.sha256()
    fl = files(patterns)
    if not fl:
        raise SystemExit("kernel_sources: no files for %r under %s" % (patterns, ROOT))
    for p in fl:
        h.update(str(p.relative_to(ROOT)).encode() + b"\0")
        h.update(p.read_bytes() + b"\0")
    return h


def source_key():
    """16 hex digits naming a kernel set by its sources."""
    return hash_files(ID_SOURCES + KEY_EXTRA).hexdigest()[:16]


if __name__ == "__main__":
    if "--list" in sys.argv:
        for p in files(ID_SOURCES + KEY_EXTRA):
            print(p.relative_to(ROOT))
    else:
        print(source_key())
