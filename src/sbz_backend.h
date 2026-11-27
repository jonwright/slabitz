/* slabitz device backend interface (P2).
 *
 * The stage logic (sbz_chain.c) talks to the GPU only through this small vtable,
 * so a Vulkan backend can be added later without touching the orchestration.
 * The only implementation today is the CUDA driver API backend (sbz_cuda.c),
 * which dlopen()s libcuda and uses only calls that exist in CUDA 6.5 (driver
 * 340, Kepler sm_35 floor).
 *
 * Model: one backend == one context + one in-order queue (CUDA stream).
 * Every operation is enqueued in order; `sync` waits for all of it. `download`
 * results are only valid after `sync`. `upload` may return before the copy is
 * done; the source must stay valid until the next `sync`.
 *
 * Device memory is an opaque 64-bit handle (a CUdeviceptr for CUDA). 0 is the
 * "no buffer" handle and is legal to bind for a buffer of 0 elements.
 */
#ifndef SBZ_BACKEND_H
#define SBZ_BACKEND_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* exported from libslabitz.so (built with -fvisibility=hidden) */
#if defined(__GNUC__)
#define SBZ_API __attribute__((visibility("default")))
#else
#define SBZ_API
#endif

/* status codes shared by the backend and the chain */
enum {
    SBZ_OK = 0,
    SBZ_ERR_BACKEND = -1,  /* driver/library failure: see sbz_backend.err */
    SBZ_ERR_NOMEM = -2,    /* device or host allocation failed */
    SBZ_ERR_ARGS = -3,     /* bad API arguments / binding mismatch (a bug) */
    SBZ_ERR_KERNEL = -4,   /* kernel image missing or failed to load */
    SBZ_ERR_CORRUPT = -5,  /* input is not decodable (see per-frame status) */
    SBZ_ERR_LIMIT = -6,    /* input exceeds a size limit of the DS kernels */
    SBZ_ERR_UNSUPPORTED = -7 /* valid zstd this decoder does not handle (dictionary,
                              * window of 2^29 or more, no content size) */
};

typedef uint64_t sbz_devptr;

/* One kernel launch argument, in kernel parameter order. For a buffer the
 * {ptr,count} pair is laid out exactly like Slang's CUDA (RW)StructuredBuffer /
 * ByteAddressBuffer { T *data; size_t count; } so a backend can pass &arg.v.buf
 * straight through. count is in elements (bytes for a ByteAddressBuffer). */
typedef struct sbz_launch_arg {
    uint32_t is_buf;
    uint32_t pad_;
    union {
        struct { sbz_devptr ptr; uint64_t count; } buf;
        uint32_t u32;
    } v;
} sbz_launch_arg;

typedef struct sbz_backend sbz_backend;

typedef struct sbz_backend_ops {
    void (*destroy)(sbz_backend *be);
    int (*alloc)(sbz_backend *be, size_t bytes, sbz_devptr *out);
    void (*free)(sbz_backend *be, sbz_devptr p);
    int (*upload)(sbz_backend *be, sbz_devptr dst, const void *src, size_t bytes);
    int (*download)(sbz_backend *be, void *dst, sbz_devptr src, size_t bytes);
    int (*fill32)(sbz_backend *be, sbz_devptr dst, uint32_t value, size_t count);
    /* kernel: enum sbz_kernel_id; groups: 1-D grid size (> 0); block size comes
     * from the generated kernel table. */
    int (*launch)(sbz_backend *be, int kernel, uint32_t groups,
                  const sbz_launch_arg *args, uint32_t nargs);
    int (*sync)(sbz_backend *be);
    /* Page-locked host memory for staging small uploads, so an upload queued on
     * the stream does not hold the host until the stream reaches it. */
    int (*host_alloc)(sbz_backend *be, size_t bytes, void **out);
    void (*host_free)(sbz_backend *be, void *p);
} sbz_backend_ops;

struct sbz_backend {
    const sbz_backend_ops *ops;
    char err[256]; /* last error message, NUL-terminated */
};

/* CUDA driver backend.
 *   device:     CUDA device ordinal. If a CUDA context is already current on the
 *               calling thread it is used (and not destroyed); otherwise a new
 *               context is created on `device`.
 *   kernel_dir: directory holding the per-kernel cubins at the paths listed in
 *               sbz_kernels_gen.h (normally the repo `build/`). NULL -> $SLABITZ_KERNEL_DIR.
 * Returns NULL on failure and, if errbuf is non-NULL, writes a message there. */
SBZ_API sbz_backend *sbz_cuda_create(int device, const char *kernel_dir, char *errbuf, size_t errlen);

#ifdef __cplusplus
}
#endif

#endif /* SBZ_BACKEND_H */
