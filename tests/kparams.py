"""Kernel parameter lists from the generated table (T01).

src/sbz_kernels_gen.h (tools/gen_kernel_table.py, from slangc reflection) is what
the C library checks every launch against. The per-kernel Python scripts build
their cuLaunchKernel lists by hand, and a short list is not an error at launch:
the kernel reads garbage and may still "pass" (2026-11-25: six scripts were
behind for weeks). So every script calls check_args(kernel, args) right before
each launch.

    params(kernel)            [(name, kind, elem_size, writable), ...]; kind 0 structured
                              buffer, 1 byte-address buffer, 2 scalar (SBZ_KP_* in sbz_kernels.h)
    info(kernel)              {"cubin", "entry", "block_x", "nparams"} (cubin relative to a kernel dir)
    check_args(kernel, args)  AssertionError naming kernel and position if the count differs,
                              a buffer position (kind 0 or 1) is not a {u64 data, u64 count}
                              ctypes structure (for kind 1, count is the size in bytes), or a scalar position is not a ctypes integer of
                              elem_size bytes. Buffer counts are not checked.

    python3 tests/kparams.py --self-test
    python3 tests/kparams.py --cubin ExecuteSequences    (prints exec/exec.cubin)

Parses the header with a regex; needs no compiler. Importable from any cwd.
"""
import ctypes as ct
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEADER = os.path.join(ROOT, "src", "sbz_kernels_gen.h")
KIND_SBUF, KIND_BAB, KIND_SCALAR = 0, 1, 2  # SBZ_KP_* in src/sbz_kernels.h

_PARAMS_RE = re.compile(r"static const sbz_kparam sbz_kparams_(\w+)\[\] = \{(.*?)\n\};", re.S)
_PARAM_RE = re.compile(r'\{\s*"(\w+)",\s*(\d+),\s*(\d+),\s*(\d+)\s*\}')
_TABLE_RE = re.compile(r"sbz_kernel_table\[SBZK_COUNT\] = \{(.*?)\n\};", re.S)
_ROW_RE = re.compile(r'\{\s*"(\w+)",\s*"([^"]+)",\s*"(\w+)",\s*(\d+),\s*(\d+),\s*sbz_kparams_(\w+)\s*\}')
_ENUM_RE = re.compile(r"enum sbz_kernel_id \{(.*?)\};", re.S)

_cache = None


def _load():
    global _cache
    if _cache is None:
        with open(HEADER) as f:
            text = f.read()
        params = {}
        for name, body in _PARAMS_RE.findall(text):
            params[name] = [(n, int(k), int(e), int(w)) for n, k, e, w in _PARAM_RE.findall(body)]
        table = {}
        m = _TABLE_RE.search(text)
        assert m, "%s: sbz_kernel_table not found" % HEADER
        for name, cubin, entry, bx, npar, pname in _ROW_RE.findall(m.group(1)):
            assert pname == name, "%s: row %s points at sbz_kparams_%s" % (HEADER, name, pname)
            table[name] = {"cubin": cubin, "entry": entry, "block_x": int(bx), "nparams": int(npar)}
        m = _ENUM_RE.search(text)
        assert m, "%s: enum sbz_kernel_id not found" % HEADER
        enum = [s.strip()[len("SBZK_"):] for s in m.group(1).split(",")
                if s.strip() and s.strip() != "SBZK_COUNT"]
        _cache = (params, table, enum)
    return _cache


def kernels():
    """Kernel names in table order."""
    return list(_load()[2])


def params(kernel):
    p = _load()[0]
    assert kernel in p, "unknown kernel %r (known: %s)" % (kernel, ", ".join(sorted(p)))
    return list(p[kernel])


def info(kernel):
    t = _load()[1]
    assert kernel in t, "unknown kernel %r (known: %s)" % (kernel, ", ".join(sorted(t)))
    return dict(t[kernel])


def _is_buf(a):
    if not isinstance(a, ct.Structure):
        return False
    f = getattr(type(a), "_fields_", None)
    return (ct.sizeof(a) == 16 and f is not None and len(f) == 2 and
            all(ct.sizeof(t) == 8 for _, t in f))


_INT_TYPES = (ct.c_int8, ct.c_uint8, ct.c_int16, ct.c_uint16, ct.c_int32, ct.c_uint32,
              ct.c_int64, ct.c_uint64)


def check_args(kernel, args):
    """Raise AssertionError unless `args` matches the kernel's table entry (see module doc)."""
    p = params(kernel)
    args = list(args)
    assert len(args) == len(p), "%s: %d args given, the kernel takes %d (%s)" % (
        kernel, len(args), len(p), ", ".join(n for n, _, _, _ in p))
    for i, ((name, kind, esize, _), a) in enumerate(zip(p, args)):
        if kind != KIND_SCALAR:
            assert _is_buf(a), "%s: arg %d (%s) is a buffer; got %r, not a {u64 data, u64 count} struct" % (
                kernel, i, name, type(a).__name__)
        else:
            assert isinstance(a, _INT_TYPES) and ct.sizeof(a) == esize, (
                "%s: arg %d (%s) is a %d-byte scalar; got %r" % (kernel, i, name, esize, type(a).__name__))
    return True


def _self_test():
    params_, table, enum = _load()
    assert len(enum) == len(table) == len(params_), "enum %d, table %d, param lists %d" % (
        len(enum), len(table), len(params_))
    for k in enum:
        n = len(params(k))
        assert n == info(k)["nparams"], "%s: %d params parsed, table says %d" % (k, n, info(k)["nparams"])
        assert n > 0, k
        for name, kind, esize, _ in params(k):
            assert kind in (KIND_SBUF, KIND_BAB, KIND_SCALAR), "%s.%s: kind %d" % (k, name, kind)
            assert kind != KIND_SCALAR or esize == 4, "%s.%s: scalar of %d bytes" % (k, name, esize)

    class Buf(ct.Structure):
        _fields_ = [("data", ct.c_uint64), ("count", ct.c_uint64)]

    def full(kernel):
        return [ct.c_uint32(0) if kind == KIND_SCALAR else Buf(0, 0) for _, kind, _, _ in params(kernel)]

    good = full("ExecuteSequences")
    check_args("ExecuteSequences", good)
    for bad, what in ((good[:-1], "short list"),
                      (good + [ct.c_uint32(0)], "long list"),
                      ([ct.c_uint32(0)] + good[1:], "scalar in a buffer slot"),
                      (good[:-1] + [Buf(0, 0)], "buffer in a scalar slot"),
                      (good[:-1] + [ct.c_uint64(0)], "8-byte scalar in a 4-byte slot")):
        try:
            check_args("ExecuteSequences", bad)
        except AssertionError:
            continue
        raise SystemExit("self-test FAIL: check_args accepted a %s" % what)
    print("%d kernels parsed, counts match" % len(enum))
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        sys.exit(_self_test())
    if len(sys.argv) == 3 and sys.argv[1] == "--cubin":  # for tests/run_kernel_scripts.sh
        print(info(sys.argv[2])["cubin"])
        sys.exit(0)
    for k in kernels():
        i = info(k)
        print("%-40s %-22s %2d params" % (k, i["cubin"], i["nparams"]))
