/* CUDA driver API backend for slabitz (see sbz_backend.h).
 *
 * libcuda is dlopen()ed at run time (no link-time CUDA dependency) and only
 * entry points that exist in CUDA 6.5 / driver 340 are used, so the same
 * library runs on the Kepler sm_35 floor and on current cards.
 *
 * Kernels are loaded lazily, one CUDA module per kernel cubin (never one
 * combined module: see tools/build_all.py for why).
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sbz_backend.h"
#include "sbz_kernels.h"

/* ---- minimal driver API declarations (ABI-stable since CUDA 3.2 _v2) ---- */
typedef int CUresult;
typedef int CUdevice;
typedef struct CUctx_st *CUcontext;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
typedef struct CUstream_st *CUstream;
typedef struct CUevent_st *CUevent;
typedef unsigned long long CUdeviceptr;

#define CU_STREAM_NON_BLOCKING 0x1

typedef struct cu_api {
    CUresult (*Init)(unsigned int);
    CUresult (*DeviceGet)(CUdevice *, int);
    CUresult (*CtxGetCurrent)(CUcontext *);
    CUresult (*CtxSetCurrent)(CUcontext);
    CUresult (*CtxCreate)(CUcontext *, unsigned int, CUdevice);
    CUresult (*CtxDestroy)(CUcontext);
    CUresult (*ModuleLoadData)(CUmodule *, const void *);
    CUresult (*ModuleUnload)(CUmodule);
    CUresult (*ModuleGetFunction)(CUfunction *, CUmodule, const char *);
    CUresult (*MemAlloc)(CUdeviceptr *, size_t);
    CUresult (*MemFree)(CUdeviceptr);
    CUresult (*EventCreate)(CUevent *, unsigned int);
    CUresult (*EventRecord)(CUevent, CUstream);
    CUresult (*EventSynchronize)(CUevent);
    CUresult (*EventElapsedTime)(float *, CUevent, CUevent);
    CUresult (*EventDestroy)(CUevent);
    CUresult (*MemHostAlloc)(void **, size_t, unsigned int);
    CUresult (*MemFreeHost)(void *);
    CUresult (*MemcpyHtoDAsync)(CUdeviceptr, const void *, size_t, CUstream);
    CUresult (*MemcpyDtoHAsync)(void *, CUdeviceptr, size_t, CUstream);
    CUresult (*MemsetD32Async)(CUdeviceptr, unsigned int, size_t, CUstream);
    CUresult (*LaunchKernel)(CUfunction, unsigned int, unsigned int, unsigned int,
                             unsigned int, unsigned int, unsigned int, unsigned int,
                             CUstream, void **, void **);
    CUresult (*StreamCreate)(CUstream *, unsigned int);
    CUresult (*StreamDestroy)(CUstream);
    CUresult (*StreamSynchronize)(CUstream);
    CUresult (*GetErrorString)(CUresult, const char **); /* optional (6.0+) */
    CUresult (*PrimaryCtxRetain)(CUcontext *, CUdevice);  /* optional (7.0+) */
    CUresult (*PrimaryCtxRelease)(CUdevice);              /* optional (7.0+) */
} cu_api;

typedef struct sbz_cuda {
    sbz_backend base; /* must be first */
    void *lib;
    cu_api cu;
    CUcontext ctx;
    int own_ctx;  /* 1: we created ctx (driver without the primary-context API) */
    int prim;     /* 1: ctx is the device's primary context, retained by us */
    CUdevice dev;
    CUstream stream;
    char kernel_dir[1024];
    CUmodule mod[SBZK_COUNT];
    CUfunction fn[SBZK_COUNT];
    /* SLABITZ_PROFILE=1: every operation is followed by a stream sync and its
     * wall time is accumulated per kernel / op kind; the table is printed to
     * stderr at destroy. Serialises everything: for finding hot spots only. */
    int prof;
    double prof_t[SBZK_COUNT + 6];
    unsigned long prof_n[SBZK_COUNT + 6];
    /* SLABITZ_TRACE=1: a CUDA event before and after every upload, download,
     * fill and launch on the stream, no syncs; the device timeline is printed
     * to stderr at destroy, one "gztrace" line per op, in microseconds on a
     * clock shared by every context of the process (each context's origin event
     * is aligned to the host clock at create, so lines of several contexts can be
     * merged to within some microseconds). For finding what overlaps. */
    int trace;
    CUevent trace_origin;
    double trace_host0;
    struct sbz_trace_ev { int slot; CUevent a, b; } *tr;
    size_t trn, trcap;
} sbz_cuda;

static double g_trace_zero; /* host time of the first traced context's origin */

enum { P_UPLOAD = SBZK_COUNT, P_DOWNLOAD, P_FILL, P_SYNC, P_ALLOC, P_FREE };
static const char *const kProfOps[] = { "upload", "download", "fill32", "sync", "alloc", "free" };

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* trace: an event recorded on the stream now (NULL if tracing is off or fails) */
static CUevent trace_mark(sbz_cuda *c)
{
    CUevent e = NULL;
    if (!c->trace || c->cu.EventCreate(&e, 0) != 0)
        return NULL;
    if (c->cu.EventRecord(e, c->stream) != 0) {
        c->cu.EventDestroy(e);
        return NULL;
    }
    return e;
}

/* trace: close the op started by trace_mark (stream ops only) */
static void trace_add(sbz_cuda *c, int slot, CUevent a)
{
    CUevent b;
    if (!a)
        return;
    if (slot == P_SYNC || slot == P_ALLOC || slot == P_FREE || !(b = trace_mark(c))) {
        c->cu.EventDestroy(a);
        return;
    }
    if (c->trn == c->trcap) {
        size_t n = c->trcap ? 2 * c->trcap : 1024;
        struct sbz_trace_ev *t = (struct sbz_trace_ev *)realloc(c->tr, n * sizeof(*t));
        if (!t) {
            c->cu.EventDestroy(a);
            c->cu.EventDestroy(b);
            return;
        }
        c->tr = t;
        c->trcap = n;
    }
    c->tr[c->trn].slot = slot;
    c->tr[c->trn].a = a;
    c->tr[c->trn].b = b;
    ++c->trn;
}

/* profile bracket: PROF_BEGIN before the op, PROF_END(slot) after it */
#define PROF_BEGIN                                                              \
    double prof_t0 = c->prof ? now_s() : 0.0;                                  \
    CUevent trace_a = trace_mark(c)
#define PROF_END(slot)                                                          \
    do {                                                                        \
        trace_add(c, (slot), trace_a);                                          \
        if (c->prof) {                                                          \
            c->cu.StreamSynchronize(c->stream);                                 \
            c->prof_t[slot] += now_s() - prof_t0;                               \
            c->prof_n[slot] += 1;                                               \
        }                                                                       \
    } while (0)

static void set_err(sbz_backend *be, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(be->err, sizeof(be->err), fmt, ap);
    va_end(ap);
}

/* Records a failing driver call in be->err; returns SBZ_OK or SBZ_ERR_BACKEND. */
static int cu_check(sbz_cuda *c, CUresult r, const char *what)
{
    const char *s = NULL;
    if (r == 0)
        return SBZ_OK;
    if (c->cu.GetErrorString)
        c->cu.GetErrorString(r, &s);
    set_err(&c->base, "%s failed: CUresult %d (%s)", what, (int)r, s ? s : "?");
    return SBZ_ERR_BACKEND;
}

#define CU(call, what) cu_check(c, (call), (what))

static int load_api(sbz_cuda *c, char *errbuf, size_t errlen)
{
    const char *names[] = { getenv("CUDA_DRIVER"), "libcuda.so.1", "libcuda.so" };
    size_t i;
    for (i = 0; i < sizeof(names) / sizeof(names[0]) && !c->lib; ++i)
        if (names[i] && names[i][0])
            c->lib = dlopen(names[i], RTLD_NOW | RTLD_LOCAL);
    if (!c->lib) {
        snprintf(errbuf, errlen, "cannot dlopen libcuda: %s", dlerror());
        return SBZ_ERR_BACKEND;
    }
#define SYM(field, name)                                                        \
    do {                                                                        \
        *(void **)(&c->cu.field) = dlsym(c->lib, name);                         \
        if (!c->cu.field) {                                                     \
            snprintf(errbuf, errlen, "libcuda lacks %s", name);                 \
            return SBZ_ERR_BACKEND;                                              \
        }                                                                       \
    } while (0)
    *(void **)(&c->cu.PrimaryCtxRetain) = dlsym(c->lib, "cuDevicePrimaryCtxRetain");
    *(void **)(&c->cu.PrimaryCtxRelease) = dlsym(c->lib, "cuDevicePrimaryCtxRelease_v2");
    if (!c->cu.PrimaryCtxRelease)
        *(void **)(&c->cu.PrimaryCtxRelease) = dlsym(c->lib, "cuDevicePrimaryCtxRelease");
    if (!c->cu.PrimaryCtxRelease)
        c->cu.PrimaryCtxRetain = NULL;
    SYM(Init, "cuInit");
    SYM(DeviceGet, "cuDeviceGet");
    SYM(CtxGetCurrent, "cuCtxGetCurrent");
    SYM(CtxSetCurrent, "cuCtxSetCurrent");
    SYM(CtxCreate, "cuCtxCreate_v2");
    SYM(CtxDestroy, "cuCtxDestroy_v2");
    SYM(ModuleLoadData, "cuModuleLoadData");
    SYM(ModuleUnload, "cuModuleUnload");
    SYM(ModuleGetFunction, "cuModuleGetFunction");
    SYM(MemAlloc, "cuMemAlloc_v2");
    SYM(MemFree, "cuMemFree_v2");
    SYM(EventCreate, "cuEventCreate");
    SYM(EventRecord, "cuEventRecord");
    SYM(EventSynchronize, "cuEventSynchronize");
    SYM(EventElapsedTime, "cuEventElapsedTime");
    SYM(EventDestroy, "cuEventDestroy_v2");
    SYM(MemHostAlloc, "cuMemHostAlloc");
    SYM(MemFreeHost, "cuMemFreeHost");
    SYM(MemcpyHtoDAsync, "cuMemcpyHtoDAsync_v2");
    SYM(MemcpyDtoHAsync, "cuMemcpyDtoHAsync_v2");
    SYM(MemsetD32Async, "cuMemsetD32Async");
    SYM(LaunchKernel, "cuLaunchKernel");
    SYM(StreamCreate, "cuStreamCreate");
    SYM(StreamDestroy, "cuStreamDestroy_v2");
    SYM(StreamSynchronize, "cuStreamSynchronize");
#undef SYM
    *(void **)(&c->cu.GetErrorString) = dlsym(c->lib, "cuGetErrorString");
    return SBZ_OK;
}

/* Contexts this library created itself (drivers without the primary-context API,
 * e.g. Kepler's 340): such a context stays current on the thread after its
 * sbz_dctx's calls, and must never be borrowed by another sbz_dctx, which would
 * crash once the creator destroys it. Guarded by a spin lock (creates are rare). */
#define SBZ_MAX_OWN_CTX 256
static CUcontext g_own_ctx[SBZ_MAX_OWN_CTX];
static int g_own_n;
static volatile int g_own_lock;

static void own_ctx_lock(void)
{
    while (__sync_lock_test_and_set(&g_own_lock, 1))
        ;
}

static void own_ctx_unlock(void) { __sync_lock_release(&g_own_lock); }

static int own_ctx_known(CUcontext x)
{
    int i, found = 0;
    own_ctx_lock();
    for (i = 0; i < g_own_n && !found; ++i)
        found = g_own_ctx[i] == x;
    own_ctx_unlock();
    return found;
}

static void own_ctx_add(CUcontext x)
{
    own_ctx_lock();
    if (g_own_n < SBZ_MAX_OWN_CTX)
        g_own_ctx[g_own_n++] = x;
    own_ctx_unlock();
}

static void own_ctx_forget(CUcontext x)
{
    int i;
    own_ctx_lock();
    for (i = 0; i < g_own_n; ++i)
        if (g_own_ctx[i] == x) {
            g_own_ctx[i] = g_own_ctx[--g_own_n];
            break;
        }
    own_ctx_unlock();
}

static void cuda_destroy(sbz_backend *be)
{
    sbz_cuda *c = (sbz_cuda *)be;
    int k;
    if (!c)
        return;
    if (c->prof) {
        double tot = 0;
        for (k = 0; k < SBZK_COUNT + 6; ++k)
            tot += c->prof_t[k];
        fprintf(stderr, "slabitz profile (each op synced; ms total, calls, us/call):\n");
        for (k = 0; k < SBZK_COUNT + 6; ++k)
            if (c->prof_n[k])
                fprintf(stderr, "  %-40s %9.3f %7lu %9.1f\n",
                        k < SBZK_COUNT ? sbz_kernel_table[k].name : kProfOps[k - SBZK_COUNT],
                        1e3 * c->prof_t[k], c->prof_n[k], 1e6 * c->prof_t[k] / c->prof_n[k]);
        fprintf(stderr, "  %-40s %9.3f\n", "total", 1e3 * tot);
    }
    if (c->ctx) {
        c->cu.CtxSetCurrent(c->ctx);
        if (c->stream)
            c->cu.StreamSynchronize(c->stream);
        if (c->trace_origin) {
            size_t i;
            const double off = 1e6 * (c->trace_host0 - g_trace_zero);
            for (i = 0; i < c->trn; ++i) {
                float ta = 0.f, tb = 0.f;
                const int s = c->tr[i].slot;
                c->cu.EventElapsedTime(&ta, c->trace_origin, c->tr[i].a);
                c->cu.EventElapsedTime(&tb, c->trace_origin, c->tr[i].b);
                fprintf(stderr, "gztrace %p %12.1f %12.1f %s\n", (void *)c, off + 1e3 * ta,
                        off + 1e3 * tb, s < SBZK_COUNT ? sbz_kernel_table[s].name : kProfOps[s - SBZK_COUNT]);
                c->cu.EventDestroy(c->tr[i].a);
                c->cu.EventDestroy(c->tr[i].b);
            }
            free(c->tr);
            c->cu.EventDestroy(c->trace_origin);
        }
        for (k = 0; k < SBZK_COUNT; ++k)
            if (c->mod[k])
                c->cu.ModuleUnload(c->mod[k]);
        if (c->stream)
            c->cu.StreamDestroy(c->stream);
        if (c->own_ctx) {
            own_ctx_forget(c->ctx);
            c->cu.CtxDestroy(c->ctx);
        }
        if (c->prim) {
            CUcontext cur = NULL;
            /* leave no released primary context current on this thread */
            if (c->cu.CtxGetCurrent(&cur) == 0 && cur == c->ctx)
                c->cu.CtxSetCurrent(NULL);
            c->cu.PrimaryCtxRelease(c->dev);
        }
    }
    if (c->lib)
        dlclose(c->lib);
    free(c);
}

/* Each entry point makes our context current: the caller may have switched
 * contexts (or threads) since the last call. Cheap when already current. */
static int make_current(sbz_cuda *c)
{
    return CU(c->cu.CtxSetCurrent(c->ctx), "cuCtxSetCurrent");
}

static int cuda_alloc(sbz_backend *be, size_t bytes, sbz_devptr *out)
{
    sbz_cuda *c = (sbz_cuda *)be;
    CUdeviceptr p = 0;
    CUresult r;
    *out = 0;
    if (bytes == 0)
        return SBZ_OK;
    if (make_current(c))
        return SBZ_ERR_BACKEND;
    {
        PROF_BEGIN;
        r = c->cu.MemAlloc(&p, bytes);
        PROF_END(P_ALLOC);
    }
    if (r == 2 /* CUDA_ERROR_OUT_OF_MEMORY */) {
        set_err(be, "cuMemAlloc(%zu) out of memory", bytes);
        return SBZ_ERR_NOMEM;
    }
    if (CU(r, "cuMemAlloc"))
        return SBZ_ERR_BACKEND;
    *out = (sbz_devptr)p;
    return SBZ_OK;
}

static int cuda_host_alloc(sbz_backend *be, size_t bytes, void **out)
{
    sbz_cuda *c = (sbz_cuda *)be;
    *out = NULL;
    if (make_current(c))
        return SBZ_ERR_BACKEND;
    return CU(c->cu.MemHostAlloc(out, bytes, 0), "cuMemHostAlloc");
}

static void cuda_host_free(sbz_backend *be, void *p)
{
    sbz_cuda *c = (sbz_cuda *)be;
    if (p && make_current(c) == SBZ_OK)
        c->cu.MemFreeHost(p);
}

static void cuda_free(sbz_backend *be, sbz_devptr p)
{
    sbz_cuda *c = (sbz_cuda *)be;
    if (!p)
        return;
    if (make_current(c) == SBZ_OK) {
        PROF_BEGIN;
        c->cu.MemFree((CUdeviceptr)p);
        PROF_END(P_FREE);
    }
}

static int cuda_upload(sbz_backend *be, sbz_devptr dst, const void *src, size_t bytes)
{
    sbz_cuda *c = (sbz_cuda *)be;
    if (bytes == 0)
        return SBZ_OK;
    if (make_current(c))
        return SBZ_ERR_BACKEND;
    {
        int rc;
        PROF_BEGIN;
        rc = CU(c->cu.MemcpyHtoDAsync((CUdeviceptr)dst, src, bytes, c->stream), "cuMemcpyHtoDAsync");
        PROF_END(P_UPLOAD);
        return rc;
    }
}

static int cuda_download(sbz_backend *be, void *dst, sbz_devptr src, size_t bytes)
{
    sbz_cuda *c = (sbz_cuda *)be;
    if (bytes == 0)
        return SBZ_OK;
    if (make_current(c))
        return SBZ_ERR_BACKEND;
    {
        int rc;
        PROF_BEGIN;
        rc = CU(c->cu.MemcpyDtoHAsync(dst, (CUdeviceptr)src, bytes, c->stream), "cuMemcpyDtoHAsync");
        PROF_END(P_DOWNLOAD);
        return rc;
    }
}

static int cuda_fill32(sbz_backend *be, sbz_devptr dst, uint32_t value, size_t count)
{
    sbz_cuda *c = (sbz_cuda *)be;
    if (count == 0)
        return SBZ_OK;
    if (make_current(c))
        return SBZ_ERR_BACKEND;
    {
        int rc;
        PROF_BEGIN;
        rc = CU(c->cu.MemsetD32Async((CUdeviceptr)dst, value, count, c->stream), "cuMemsetD32Async");
        PROF_END(P_FILL);
        return rc;
    }
}

static int load_kernel(sbz_cuda *c, int k)
{
    char cubin[1300], fatbin[1300];
    const char *opened;
    FILE *f;
    long n;
    void *img;
    CUresult r;

    snprintf(cubin, sizeof(cubin), "%s/%s", c->kernel_dir, sbz_kernel_table[k].cubin);
    /* A fatbin is one kernel compiled for many architectures: same path, .fatbin
     * in place of .cubin (build/kernels/<sub>/<stem>.fatbin). Prefer it; fall back
     * to the single-arch cubin so every existing kernel directory keeps working. */
    snprintf(fatbin, sizeof(fatbin), "%s", cubin);
    {
        char *dot = strrchr(fatbin, '.');
        if (dot && strcmp(dot, ".cubin") == 0)
            strcpy(dot, ".fatbin");
    }
    f = fopen(fatbin, "rb");
    opened = fatbin;
    if (!f) {
        f = fopen(cubin, "rb");
        opened = cubin;
    }
    if (!f) {
        set_err(&c->base, "cannot open kernel %s or %s", fatbin, cubin);
        return SBZ_ERR_KERNEL;
    }
    fseek(f, 0, SEEK_END);
    n = ftell(f);
    fseek(f, 0, SEEK_SET);
    img = n > 0 ? malloc((size_t)n) : NULL;
    if (!img || fread(img, 1, (size_t)n, f) != (size_t)n) {
        fclose(f);
        free(img);
        set_err(&c->base, "cannot read kernel %s", opened);
        return SBZ_ERR_KERNEL;
    }
    fclose(f);
    r = c->cu.ModuleLoadData(&c->mod[k], img);
    free(img);
    if (r) {
        cu_check(c, r, "cuModuleLoadData");
        c->mod[k] = NULL;
        return SBZ_ERR_KERNEL;
    }
    if (CU(c->cu.ModuleGetFunction(&c->fn[k], c->mod[k], sbz_kernel_table[k].entry),
           "cuModuleGetFunction"))
        return SBZ_ERR_KERNEL;
    return SBZ_OK;
}

static int cuda_launch(sbz_backend *be, int k, uint32_t groups, const sbz_launch_arg *args,
                       uint32_t nargs)
{
    sbz_cuda *c = (sbz_cuda *)be;
    void *params[64];
    uint32_t i;
    int rc;

    if (k < 0 || k >= SBZK_COUNT || nargs > 64 || groups == 0) {
        set_err(be, "bad launch (kernel %d, nargs %u, groups %u)", k, nargs, groups);
        return SBZ_ERR_ARGS;
    }
    if (make_current(c))
        return SBZ_ERR_BACKEND;
    if (!c->fn[k] && (rc = load_kernel(c, k)) != SBZ_OK)
        return rc;
    for (i = 0; i < nargs; ++i)
        params[i] = args[i].is_buf ? (void *)&args[i].v.buf : (void *)&args[i].v.u32;
    /* grid.x up to 2^31-1 on sm_30+; DS's 2-D (groupId.y << 15) split is not needed */
    {
        PROF_BEGIN;
        rc = CU(c->cu.LaunchKernel(c->fn[k], groups, 1, 1, sbz_kernel_table[k].block_x, 1, 1, 0,
                                   c->stream, params, NULL),
                sbz_kernel_table[k].name);
        PROF_END(k);
    }
    return rc;
}

static int cuda_sync(sbz_backend *be)
{
    sbz_cuda *c = (sbz_cuda *)be;
    if (make_current(c))
        return SBZ_ERR_BACKEND;
    {
        int rc;
        PROF_BEGIN;
        rc = CU(c->cu.StreamSynchronize(c->stream), "cuStreamSynchronize");
        PROF_END(P_SYNC);
        return rc;
    }
}

static const sbz_backend_ops cuda_ops = {
    cuda_destroy, cuda_alloc, cuda_free, cuda_upload, cuda_download,
    cuda_fill32, cuda_launch, cuda_sync, cuda_host_alloc, cuda_host_free,
};

/* The kernel directory must hold the kernel set this library was generated
 * with: its sbz_kernels.id (written by tools/gen_kernel_table.py and copied by
 * build_kubins.py / build_fatbins.py) must equal SBZ_KERNELS_ID. A stale set
 * would otherwise load silently, and a changed argument list crashes the
 * launch. Returns 0 if it matches. */
static int check_kernels_id(const char *kernel_dir, char *errbuf, size_t errlen)
{
    char path[1100], id[64];
    size_t n;
    FILE *f;

    snprintf(path, sizeof(path), "%s/sbz_kernels.id", kernel_dir);
    f = fopen(path, "rb");
    if (!f) {
        snprintf(errbuf, errlen,
                 "kernel dir %s has no sbz_kernels.id (built before kernel IDs, or not a "
                 "kernel dir); rebuild the kernels", kernel_dir);
        return 1;
    }
    n = fread(id, 1, sizeof(id) - 1, f);
    fclose(f);
    id[n] = 0;
    while (n > 0 && (id[n - 1] == '\n' || id[n - 1] == '\r' || id[n - 1] == ' '))
        id[--n] = 0;
    if (strcmp(id, SBZ_KERNELS_ID) != 0) {
        snprintf(errbuf, errlen,
                 "kernel dir %s holds kernels id %s, this library expects %s: kernels and "
                 "library come from different sources; rebuild both", kernel_dir, id,
                 SBZ_KERNELS_ID);
        return 1;
    }
    return 0;
}

sbz_backend *sbz_cuda_create(int device, const char *kernel_dir, char *errbuf, size_t errlen)
{
    char dummy[8];
    sbz_cuda *c;
    CUcontext cur = NULL;
    CUdevice dev;

    if (!errbuf) {
        errbuf = dummy;
        errlen = sizeof(dummy);
    }
    errbuf[0] = 0;
    if (!kernel_dir)
        kernel_dir = getenv("SLABITZ_KERNEL_DIR");
    if (!kernel_dir || strlen(kernel_dir) >= sizeof(c->kernel_dir)) {
        snprintf(errbuf, errlen, "no kernel dir (pass one or set SLABITZ_KERNEL_DIR)");
        return NULL;
    }
    c = (sbz_cuda *)calloc(1, sizeof(*c));
    if (!c) {
        snprintf(errbuf, errlen, "out of host memory");
        return NULL;
    }
    c->base.ops = &cuda_ops;
    c->prof = getenv("SLABITZ_PROFILE") && atoi(getenv("SLABITZ_PROFILE")) > 0;
    strcpy(c->kernel_dir, kernel_dir);
    if (check_kernels_id(kernel_dir, errbuf, errlen)) {
        free(c);
        return NULL;
    }
    if (load_api(c, errbuf, errlen))
        goto fail;
    if (CU(c->cu.Init(0), "cuInit"))
        goto fail_msg;
    if (CU(c->cu.CtxGetCurrent(&cur), "cuCtxGetCurrent"))
        goto fail_msg;
    /* Which CUDA context: the caller's current one (borrowed); else the device's
     * primary context, retained (shared, refcounted: what cupy, torch and jax use);
     * else, on drivers without that API, one of our own, which another sbz_dctx
     * never borrows. */
    if (CU(c->cu.DeviceGet(&dev, device), "cuDeviceGet"))
        goto fail_msg;
    c->dev = dev;
    if (c->cu.PrimaryCtxRetain) {
        CUcontext pctx = NULL;
        if (CU(c->cu.PrimaryCtxRetain(&pctx, dev), "cuDevicePrimaryCtxRetain"))
            goto fail_msg;
        if (!cur || cur == pctx) {
            c->ctx = pctx;
            c->prim = 1;
            if (CU(c->cu.CtxSetCurrent(pctx), "cuCtxSetCurrent"))
                goto fail_msg;
        } else {
            c->cu.PrimaryCtxRelease(dev);
            c->ctx = cur;
        }
    } else if (cur && !own_ctx_known(cur)) {
        c->ctx = cur;
    } else {
        if (CU(c->cu.CtxCreate(&c->ctx, 0, dev), "cuCtxCreate"))
            goto fail_msg;
        c->own_ctx = 1;
        own_ctx_add(c->ctx);
    }
    if (CU(c->cu.StreamCreate(&c->stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate"))
        goto fail_msg;
    if (getenv("SLABITZ_TRACE") && atoi(getenv("SLABITZ_TRACE")) > 0 &&
        c->cu.EventCreate(&c->trace_origin, 0) == 0) {
        c->cu.EventRecord(c->trace_origin, c->stream);
        c->cu.EventSynchronize(c->trace_origin);
        c->trace_host0 = now_s();
        if (g_trace_zero == 0.0)
            g_trace_zero = c->trace_host0;
        c->trace = 1;
    }
    return &c->base;

fail_msg:
    snprintf(errbuf, errlen, "%s", c->base.err);
fail:
    cuda_destroy(&c->base);
    return NULL;
}
