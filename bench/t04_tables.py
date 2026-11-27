#!/usr/bin/env python3
"""Markdown tables from the T04 runs (bench/t04_runs.sbatch): configurations as columns.

  python3 bench/t04_tables.py DIR --tags q2-439e2f2,q2-main [--cells zstd3/] [--metric wall]
                              [--metric kernel:ExecuteSequences] [--per-frame] [--ratio]

DIR holds <tag>-r<round>.json (gate results). Each entry is the median over rounds of the
cell's per-round median, then "iqr" = the median within-round IQR and "spread" = max - min of
the round medians (run-to-run, separate processes, same GPU). --ratio adds the last column over
the first, with the run-to-run spread of both added in quadrature as its uncertainty.
"""
import argparse
import glob
import json
import os
import re
import statistics as st


def load(d, tag):
    rounds = []
    for p in sorted(glob.glob(os.path.join(d, "%s-r*.json" % tag))):
        if re.search(r"-r\d+\.json$", p):
            rounds.append(json.load(open(p)))
    return rounds


def value(cell, metric, per_frame):
    if cell.get("status") != "ok":
        return None
    if metric == "wall":
        v, q = cell["wall_ms"], cell["iqr_ms"]
        unit = 1.0
    else:
        v = cell.get("kernels_us", {}).get(metric.split(":", 1)[1])
        if v is None:
            return None
        q, unit = 0.0, 1.0
    if per_frame:
        v, q = v / cell["frames"], q / cell["frames"]
    return v * unit, q * unit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--tags", required=True)
    ap.add_argument("--labels", default="")
    ap.add_argument("--cells", action="append", default=[])
    ap.add_argument("--metric", default="wall")
    ap.add_argument("--per-frame", action="store_true")
    ap.add_argument("--ratio", action="store_true")
    a = ap.parse_args()
    tags = a.tags.split(",")
    labels = a.labels.split(",") if a.labels else tags
    data = {t: load(a.dir, t) for t in tags}
    names = []
    for t in tags:
        for r in data[t]:
            for n in r["cells"]:
                if n not in names and (not a.cells or any(c in n for c in a.cells)):
                    names.append(n)
    unit = "ms" if a.metric == "wall" else "us"
    what = "%s%s, %s" % (a.metric, " per frame" if a.per_frame else " per call", unit)
    hdr = "| cell (%s) | " % what + " | ".join(labels) + (" | last/first |" if a.ratio else " |")
    print(hdr)
    print("|---|" + "---:|" * (len(tags) + (1 if a.ratio else 0)))
    for n in names:
        row, meds, spreads = [], [], []
        for t in tags:
            vs = [value(r["cells"][n], a.metric, a.per_frame) for r in data[t] if n in r["cells"]]
            vs = [v for v in vs if v is not None]
            if not vs:
                row.append("-")
                meds.append(None)
                spreads.append(None)
                continue
            m = st.median(v[0] for v in vs)
            q = st.median(v[1] for v in vs)
            sp = max(v[0] for v in vs) - min(v[0] for v in vs)
            meds.append(m)
            spreads.append(sp)
            fmt = "%.3f" if unit == "ms" else "%.1f"
            s = fmt % m
            if a.metric == "wall":
                s += " (iqr %.3f, spread %.3f, n=%d)" % (q, sp, len(vs))
            else:
                s += " (spread %.1f, n=%d)" % (sp, len(vs))
            row.append(s)
        if a.ratio:
            if meds[0] and meds[-1]:
                r = meds[-1] / meds[0]
                u = r * ((spreads[0] / meds[0]) ** 2 + (spreads[-1] / meds[-1]) ** 2) ** 0.5
                row.append("%.3f +- %.3f" % (r, u))
            else:
                row.append("-")
        print("| %s | %s |" % (n, " | ".join(row)))
    for t in tags:
        if data[t]:
            h = data[t][0]
            print("\n%s: commit %s, kernels %s, pac %s, node %s, job %s, %d rounds, %d frames/call"
                  % (t, h["commit"][:10], h["kernels_id"], h.get("pac", ""), h["slurm_node"],
                     h["slurm_job"], len(data[t]), h["frames_per_call"]))


if __name__ == "__main__":
    main()
