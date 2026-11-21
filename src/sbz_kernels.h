/* Kernel launch table types; the table itself is generated (sbz_kernels_gen.h). */
#ifndef SBZ_KERNELS_H
#define SBZ_KERNELS_H

#include <stdint.h>

/* parameter kinds */
enum { SBZ_KP_SBUF = 0, SBZ_KP_BAB = 1, SBZ_KP_U32 = 2 };

typedef struct sbz_kparam {
    const char *name; /* Slang entry parameter name, e.g. "inoutCounters" */
    uint8_t kind;     /* SBZ_KP_* */
    uint8_t esize;    /* element size in bytes (1 for a ByteAddressBuffer) */
    uint8_t rw;       /* 1 if the kernel may write it */
} sbz_kparam;

typedef struct sbz_kinfo {
    const char *name;  /* kernel name, e.g. "ParseFrames" */
    const char *cubin; /* path relative to the kernel dir */
    const char *entry; /* entry symbol: "main_0" (Slang renames main) or the CUDA name */
    uint32_t block_x;  /* numthreads.x */
    uint32_t nparams;
    const sbz_kparam *params;
} sbz_kinfo;

#include "sbz_kernels_gen.h"

#endif /* SBZ_KERNELS_H */
