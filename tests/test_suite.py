#!/usr/bin/env python3
"""The pytest entry point over every slabitz test script (T02).

One parametrized test per script in `sbz test` (the `run_t tests/*.py` lines of
cmd_test in tools/sbz.sh) and per non-retired script in `sbz test-kernels` (the
SCRIPTS table of tests/run_kernel_scripts.sh, skipping RETIRED ones).  The lists
are parsed from those two files at collection time, so they cannot drift from
what the underlying runners actually execute.

Each test runs the script as a subprocess under a timeout and fails on a nonzero
exit, printing the script's output.  A script is marked `gpu` when it needs a
CUDA device (every `sbz test` script, and the kernel scripts that take a cubin);
a script marked `slow` took more than 60 s when measured (the set is hard-coded
here with the measured time in a comment).  When SLABITZ_EXAMPLES is set the
child set SLABITZ_REQUIRE_DATA=1, so missing example data is a failure there.

Usage:  sbz pytest            (== python3 -m pytest tests)
        sbz pytest -m "not slow"
"""

import os
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"

# Scripts that take more than 60 s, measured once on the login node's L40S
# (shared GPU); listed so `sbz pytest -m "not slow"` can skip them.
#   run_chain_stage01  196 s
#   run_chain_stage2   242 s
#   run_roundtrip      34 s (T06 soak; maintained here so `-m "not slow"` skips it)
SLOW = frozenset({"run_chain_stage01", "run_chain_stage2", "run_roundtrip"})

# T06 scripts that are not part of `sbz test` (so sbz.sh has no run_t line for
# them) but must still be exercised by the pytest entry point.  run_features is
# in sbz test (parsed from sbz.sh); run_roundtrip is the slow soak.
EXTRA = [("run_roundtrip", ())]

# CPU-only kernel scripts (no cubin, no CUDA context): the ones T01's report
# calls CPU-only.  They all appear in run_kernel_scripts.sh with kernel "-".
CPU_ONLY = frozenset({"run_dsq_chain", "run_e2e_chain", "run_fso_chain", "run_lit_chain"})

# Per-test timeouts (seconds); the slow scripts are excluded from the default
# `sbz pytest` run, but keep a generous timeout for when they do run.
SBZ_TEST_TIMEOUT = 900
KERNEL_TIMEOUT = 300


class Spec:
    """One parametrized test: a script plus how to run it."""

    def __init__(self, script, gpu, slow, data_script, envs=(), argv=(), timeout=900,
                 kernel=None):
        self.script = script
        self.gpu = gpu
        self.slow = slow
        self.data_script = data_script  # script launches a kernel (test-kernels) vs not
        self.envs = tuple(envs)
        self.argv = tuple(argv)
        self.timeout = timeout
        self.kernel = kernel
        self.id = script + ("[%s]" % ",".join(self.envs) if self.envs else "")

    def run(self):
        env = dict(os.environ)
        if os.environ.get("SLABITZ_EXAMPLES"):
            env["SLABITZ_REQUIRE_DATA"] = "1"
        for kv in self.envs:
            k, v = kv.split("=", 1)
            env[k] = v
        cmd = [sys.executable, str(TESTS / (self.script + ".py"))] + list(self.argv)
        p = subprocess.run(cmd, env=env, capture_output=True, text=True,
                           timeout=self.timeout)
        if p.returncode != 0:
            raise AssertionError(
                "%s failed rc=%d\n--- STDOUT ---\n%s\n--- STDERR ---\n%s"
                % (self.id, p.returncode, p.stdout, p.stderr))
        return p


def sbz_test_scripts():
    """The `run_t tests/*.py` lines of cmd_test in tools/sbz.sh."""
    text = (ROOT / "tools" / "sbz.sh").read_text()
    names = re.findall(r"^\s*run_t tests/([A-Za-z0-9_]+)\.py", text, re.M)
    if not names:
        raise RuntimeError("sbz.sh: no run_t tests/ lines found")
    return names


def kernel_scripts():
    """(script, kernel, envs) tuples from the SCRIPTS table of the kernel runner."""
    text = (TESTS / "run_kernel_scripts.sh").read_text()
    m = re.search(r'SCRIPTS="\n(.*?)\n"', text, re.S)
    if not m:
        raise RuntimeError("run_kernel_scripts.sh: no SCRIPTS table found")
    out = []
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(":")
        script = parts[0]
        kernel = parts[1] if len(parts) > 1 else "-"
        envs = parts[2] if len(parts) > 2 else ""
        out.append((script, kernel, envs))
    return out


def is_retired(script):
    try:
        head = (TESTS / (script + ".py")).read_text().splitlines()[:5]
    except OSError:
        return False
    return any(l.startswith('"""RETIRED (T01):') for l in head)


def cubin_path(kernel):
    """The kernel's relative cubin, joined to the active kernel directory."""
    sys.path.insert(0, str(TESTS))
    import kparams
    rel = kparams.info(kernel)["cubin"]
    kdir = os.environ.get("SLABITZ_KERNEL_DIR", str(ROOT / "build"))
    return str(pathlib.Path(kdir) / rel)


def build_specs():
    specs = []
    seen = set()

    for name in sbz_test_scripts():
        if name in seen:
            continue
        seen.add(name)
        specs.append(Spec(name, gpu=True, slow=name in SLOW, data_script=False,
                          timeout=SBZ_TEST_TIMEOUT))

    for name, argv in EXTRA:
        if name in seen:
            continue
        seen.add(name)
        specs.append(Spec(name, gpu=True, slow=name in SLOW, data_script=False,
                          argv=argv, timeout=SBZ_TEST_TIMEOUT))

    for script, kernel, envs in kernel_scripts():
        if is_retired(script):
            continue
        key = (script, envs)
        if key in seen:
            continue
        seen.add(key)
        gpu = kernel != "-"
        argv = (cubin_path(kernel),) if gpu else ()
        specs.append(Spec(script, gpu=gpu, slow=script in SLOW, data_script=True,
                          envs=envs.split(",") if envs else (), argv=argv,
                          timeout=KERNEL_TIMEOUT, kernel=kernel))
    return specs


SPECS = build_specs()


@pytest.mark.parametrize(
    "spec",
    [pytest.param(s, id=s.id, marks=tuple(
        ([pytest.mark.gpu] if s.gpu else []) + ([pytest.mark.slow] if s.slow else [])))
        for s in SPECS],
)
def test_script(spec):
    spec.run()
