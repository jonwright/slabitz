#!/usr/bin/env python3
"""T22: compile every Slang kernel entry for the non-CUDA targets.

For each entry in src/sbz_kernels_gen.h (via tools/build_all.KERNEL_MAP, which
is the same order) compile kernels/<Name>.slang with the pinned slangc for
-target spirv, metal, hlsl and cpp (the CPU target). The flags are the CUDA
build's flags (build_all.py) with only -target changed, so a failure is about
the target and not about the flags: -profile sm_6_0, -I tools, -I <transformed
DirectStorage tree>, -entry main, -stage compute.

Per entry and target it records "ok" or the FIRST error line (with its error
code), truncated to 160 characters. The full compiler output is written only to
a temporary directory under build/ (gitignored) and removed afterwards.

Writes docs/dev/reports/T22_targets.md as one table, rows = entries in
src/sbz_kernels_gen.h order, columns = the 4 targets, plus a totals line per
target. Also prints a summary to stdout. Exits 0 on success.

Usage:
  python3 tools/survey_targets.py
"""

import pathlib
import re
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
REPORTS = ROOT / "docs" / "dev" / "reports"
KERNELS = ROOT / "kernels"
SLANGC = ROOT / "third_party" / "slang" / "bin" / "slangc"
TMPWORK = ROOT / "build" / "survey_work"

# Matches build_all.py KERNEL_MAP (same order as src/sbz_kernels_gen.h).
ENTRIES = [
    "MemsetMemcpy",
    "ComputeDestBlockOffsets",
    "InitResources",
    "PropagateFseIndex",
    "DecodeHuffmanWeights",
    "PrefixSum",
    "ComputePrefixSum",
    "PrefixSequenceOffsets",
    "ParseFrames",
    "ParseCompressedBlocks",
    "InitFseTable",
    "DecompressHuffmanWeights",
    "InitHuffmanTableAndDecompressLiterals",
    "DecompressSequences",
    "DecompressSequencesMultiStream",
    "FinaliseSequenceOffsets",
    "CheckSequenceOffsets",
    "ExecuteSequences",
    "BitUnshuffleSlow",
    "BitUnshuffleWarp",
    "Lz4DecodeBlocks",
    "Lz4DecodeUnshuffle",
    "BshufWalkChunks",
]

TARGETS = ["spirv", "metal", "hlsl", "cpp"]

PROFILE = "sm_6_0"

# First error line: 'error[E36107]: ...' or 'error 30019: ...' or 'error: ...'.
# Capture the whole line so the message survives.
ERROR_RE = re.compile(r"^.*?\berror(\[[^\]]*\])?\s*:.*$", re.M)
# Failing feature name from the diagnostic trail: note: see using of '<name>'.
USING_RE = re.compile(r"note: see using of '([^']+)'")


def git_head():
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
        return out
    except subprocess.CalledProcessError:
        return "<unknown>"


def slangc_version():
    out = subprocess.run([str(SLANGC), "-v"], capture_output=True, text=True)
    return out.stdout.strip() or out.stderr.strip() or "<unknown>"


def transform_ds(tmp):
    """Build the transformed DirectStorage tree at tmp/ds. Read-only inputs."""
    subprocess.run(
        [sys.executable, str(HERE / "transform_ds_for_slang.py"),
         "--out", str(tmp / "ds")],
        check=True, capture_output=True)


def compile_entry(tmp, entry, target):
    """Return (status, features). status is 'ok' or the first error line."""
    src = KERNELS / ("%s.slang" % entry)
    out = tmp / ("%s_%s.out" % (entry, target))
    cmd = [str(SLANGC), str(src), "-target", target,
           "-profile", PROFILE,
           "-I", str(HERE), "-I", str(tmp / "ds"),
           "-entry", "main", "-stage", "compute",
           "-o", str(out)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    combined = p.stdout + p.stderr
    m = ERROR_RE.search(combined)
    first = m.group(0).strip() if m else None
    features = sorted(set(USING_RE.findall(combined)))
    if p.returncode == 0:
        return ("ok", features)
    if first:
        return (first[:160], features)
    return ("<error, null rc=%d>" % p.returncode, features)


def main():
    if not SLANGC.exists():
        print("error: %s not found (run sbz setup)" % SLANGC)
        return 2

    version = slangc_version()
    commit = git_head()

    if TMPWORK.exists():
        shutil.rmtree(TMPWORK)
    TMPWORK.mkdir(parents=True)

    try:
        transform_ds(TMPWORK)
        rows = []
        for entry in ENTRIES:
            row = {"entry": entry}
            for tgt in TARGETS:
                status, feats = compile_entry(TMPWORK, entry, tgt)
                row[tgt] = status
                if tgt == "cpp" and status != "ok":
                    row["cpp_feats"] = feats
            rows.append(row)

        results = {}
        for tgt in TARGETS:
            ok = sum(1 for r in rows if r[tgt] == "ok")
            results[tgt] = (ok, len(ENTRIES))

        REPORTS.mkdir(parents=True, exist_ok=True)
        out_path = REPORTS / "T22_targets.md"
        lines = []
        lines.append("# T22 Slang target survey (compile only)")
        lines.append("")
        lines.append("- slangc: %s" % version)
        lines.append("- worktree HEAD: `%s`" % commit)
        lines.append("- targets: %s" % ", ".join(TARGETS))
        lines.append("- flags: `-profile %s -I tools -I <transformed DS tree> "
                     "-entry main -stage compute` (the CUDA build's flags, "
                     "only `-target` changed)" % PROFILE)
        lines.append("- rows are the entries of `src/sbz_kernels_gen.h` in "
                     "order; cell is `ok` or the first error line "
                     "(truncated to 160 chars).")
        lines.append("")
        header = "| entry | " + " | ".join(TARGETS) + " |"
        lines.append(header)
        lines.append("|---|" + "---|" * len(TARGETS))
        for r in rows:
            cells = [r["entry"]]
            for tgt in TARGETS:
                cells.append(r[tgt])
            lines.append("| " + " | ".join(cells) + " |")
        tot = ["| **totals (ok / %d)**" % len(ENTRIES)]
        for tgt in TARGETS:
            o, n = results[tgt]
            tot.append("%d/%d" % (o, n))
        lines.append(" | ".join(tot) + " |")
        lines.append("")
        out_path.write_text("\n".join(lines) + "\n")

        print("survey: %d entries x %d targets" % (len(ENTRIES), len(TARGETS)))
        for tgt in TARGETS:
            o, n = results[tgt]
            print("  %-6s %d/%d ok" % (tgt, o, n))
        print("table: %s" % out_path)
    finally:
        shutil.rmtree(TMPWORK, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
