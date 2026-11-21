"""Minimal CUDA-driver loader for Slang-compiled zstd kernels.

The Slang CUDA target emits kernels with plain, raw arguments, e.g.
``extern "C" __global__ void k(ulonglong src, ulonglong dst, uint n)``.
Those map cleanly onto the CUDA Driver API used by ``bslz4_to_sparse``'s
``_backend_drv.py``: load a cubin with ``cuModuleLoadData``, fetch the function
by name, and launch with ``cuLaunchKernel`` passing raw device pointers.

This mirrors the ctypes pattern in bslz4_to_sparse's ``src/cuda/_backend_drv.py``
but is self-contained here so it can be tested before integration.

IMPORTANT: the argument-value ctypes objects (e.g. a ``c_uint64`` holding a
device pointer) MUST stay alive until after ``cuLaunchKernel`` returns.  Build
them, keep references, and pass them through :class:`_Kernel`.
"""

import ctypes as _ct
import os

__all__ = ["Driver", "DeviceMemory", "load"]

_cu_cuda = None


def _driver():
    global _cu_cuda
    if _cu_cuda is None:
        path = os.environ.get("CUDA_DRIVER", "libcuda.so.1")
        _cu_cuda = _ct.CDLL(path)
    return _cu_cuda


class _CUDARuntimeError(RuntimeError):
    """Raised when a CUDA Driver API call returns a nonzero status."""


def _sym(name, *args, restype=None):
    f = getattr(_driver(), name)
    f.restype = restype or _ct.c_int
    f.argtypes = args
    return f


class Driver:
    """One-initiation CUDA Driver API session bound to the primary context."""

    def __init__(self, device=0):
        cu = _driver()
        self._cuInit = _sym("cuInit", _ct.c_uint)
        self._cuDeviceGet = _sym("cuDeviceGet", _ct.POINTER(_ct.c_int), _ct.c_int)
        self._cuCtxCreate = _sym("cuCtxCreate", _ct.POINTER(_ct.c_void_p), _ct.c_uint, _ct.c_int)
        self._cuCtxDestroy = _sym("cuCtxDestroy", _ct.c_void_p)
        self._cuCtxSet = _sym("cuCtxSetCurrent", _ct.c_void_p)
        self._cuSynchronize = _sym("cuCtxSynchronize")
        self._cuGetErrorString = _sym("cuGetErrorString", _ct.c_int, _ct.POINTER(_ct.c_char_p))

        self._check(self._cuInit(0))
        dev = _ct.c_int()
        self._check(self._cuDeviceGet(_ct.byref(dev), device))
        self.device = dev.value
        # Create an explicit driver context: the primary-context path returns
        # invalid-GPU-memory (CUDA_ERROR_INVALID_CONTEXT / 201) on this host even
        # though cuCtxSynchronize succeeds, so we make our own context. Pure CUDA.
        self._ctx = _ct.c_void_p()
        self._check(self._cuCtxCreate(_ct.byref(self._ctx), 0, self.device))
        self._check(self._cuCtxSet(self._ctx))

    def _check(self, rc):
        if rc != 0:
            msg = _ct.c_char_p()
            self._cuGetErrorString(rc, _ct.byref(msg))
            raise _CUDARuntimeError(rc, msg.value.decode("utf-8", "replace"))

    def load(self, cubin, **kw):
        """Load cubin bytes, returning a :class:`_Module`."""
        return _Module(self, cubin, **kw)

    def synchronize(self):
        self._check(self._cuSynchronize())

    def destroy(self):
        self._check(self._cuCtxDestroy(self._ctx))

    # ---- Pure-CUDA device memory (Driver API) --------------------------------
    # The kernels only ever consume raw device pointers, so memory is managed
    # here with cuMemAlloc/cuMemcpy*/cuMemFree (no torch/cudart dependency). This
    # keeps slabitz usable from any CUDA library that hands us pointers.

    def malloc(self, size):
        s = _ct.c_ulonglong()
        self._check(_sym("cuMemAlloc", _ct.POINTER(_ct.c_ulonglong), _ct.c_ulonglong)(
            _ct.byref(s), size))
        return int(s.value)

    def free(self, ptr):
        self._check(_sym("cuMemFree", _ct.c_ulonglong)(ptr))

    def dtoh(self, host_buf, dev_ptr, size):
        """Copy ``size`` bytes device->host into an existing ctypes buffer."""
        self._check(_sym("cuMemcpyDtoH", _ct.c_void_p, _ct.c_ulonglong, _ct.c_ulonglong)(
            host_buf, dev_ptr, size))

    def htod(self, dev_ptr, host_buf, size):
        self._check(_sym("cuMemcpyHtoD", _ct.c_ulonglong, _ct.c_void_p, _ct.c_ulonglong)(
            dev_ptr, host_buf, size))

    def d2d(self, dst, src, size):
        self._check(_sym("cuMemcpyDtoD", _ct.c_ulonglong, _ct.c_ulonglong, _ct.c_ulonglong)(
            dst, src, size))


class DeviceMemory:
    """A cuMemAlloc'd device buffer; auto-frees on GC or context teardown."""

    def __init__(self, drv, size):
        self.drv = drv
        self.size = size
        self.ptr = drv.malloc(size)

    def to_host(self, size=None):
        """Copy `size` (default: the whole buffer) back to host as raw bytes."""
        size = self.size if size is None else size
        buf = _ct.create_string_buffer(size)
        self.drv.dtoh(buf, self.ptr, size)
        return buf.raw

    def from_host(self, data):
        self.drv.htod(self.ptr, data, self.size)

    def __del__(self):
        try:
            self.drv.free(self.ptr)
        except Exception:
            pass


class _Module:
    def __init__(self, drv, cubin):
        self._drv = drv
        self._cuLoad = sym_("cuModuleLoadData", _ct.POINTER(_ct.c_void_p), _ct.c_void_p)
        self._cuGetFn = sym_("cuModuleGetFunction", _ct.POINTER(_ct.c_void_p), _ct.c_void_p, _ct.c_char_p)
        self._mod = _ct.c_void_p()
        drv._check(self._cuLoad(_ct.byref(self._mod), bytes(cubin)))
        self._kernels = {}

    def function(self, name):
        if name not in self._kernels:
            fn = _ct.c_void_p()
            self._drv._check(self._cuGetFn(_ct.byref(fn), self._mod, name.encode()))
            self._kernels[name] = fn
        return _Kernel(self._drv, self._kernels[name])


class _Kernel:
    def __init__(self, drv, fn):
        self._drv = drv
        self._fn = fn
        self._cuLaunch = sym_(
            "cuLaunchKernel", _ct.c_void_p,
            *([_ct.c_uint] * 6), _ct.c_uint, _ct.c_void_p,
            _ct.POINTER(_ct.c_void_p), _ct.c_void_p,
        )

    def __call__(self, grid, block, args, shared=0, stream=None):
        """Launch with ``args`` a list/tuple of ctypes values (kept alive)."""
        args = list(args)
        n = len(args)
        params = (_ct.c_void_p * n)(
            *(_ct.addressof(a) for a in args)
        ) if n else None
        g = tuple(grid) + (1,) * (3 - len(grid))
        b = tuple(block) + (1,) * (3 - len(block))
        # Keep args alive until after the launch so cuLaunchKernel reads valid memory.
        self._drv._check(self._cuLaunch(
            self._fn, *[c for c in g], *[c for c in b], shared, stream, params, None
        ))
        return args


def sym_(name, *args, restype=None):
    return _sym(name, *args, restype=restype)
