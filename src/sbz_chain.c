/* slabitz device-resident decode chain: DS zstdgpu.cpp stage logic in C.
 * See sbz_chain.h for the status and the readback-mode design.
 *
 * Buffers are named exactly like the DS resources (zstdgpu_resources.h) and are
 * bound to kernel parameters BY NAME: a parameter "inoutX"/"inX" binds buffer X
 * unless the launch overrides it (generic kernels such as PrefixSum). Every
 * launch is checked against the reflection-generated table (sbz_kernels_gen.h):
 * all parameters bound, element sizes equal, scalars given. A mismatch is a bug
 * and fails with SBZ_ERR_ARGS before anything reaches the GPU.
 */
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sbz_chain.h"
#include "sbz_kernels.h"
#include "host_walk.h"

/* ---- DS constants (zstdgpu_structs.h, PC build) ---- */
#define DS_WAVE_SIZE_MIN 4u /* kzstdgpu_WaveSize_Min on PC: sizes lookback buffers */
#define DS_MAX_COUNT_FSE_PROBS 64u
#define DS_FSE_RLE_TABLE_COUNT 256u
#define DS_FSE_ELEM_MAX_HUFW (1u << 7)
#define DS_FSE_ELEM_MAX_LLEN (1u << 9)
#define DS_FSE_ELEM_MAX_OFFS (1u << 8)
#define DS_FSE_ELEM_MAX_MLEN (1u << 9)
#define DS_MAX_COUNT_HUFFMAN_WEIGHTS 256u
#define DS_FSE_DEFAULT_PROB_COUNT_LLEN 36u
#define DS_FSE_DEFAULT_PROB_COUNT_OFFS 29u
#define DS_FSE_DEFAULT_PROB_COUNT_MLEN 53u
#define DS_MIN_COUNT_BLOCKS 1u
#define DS_MIN_COUNT_UNCOMPRESSED_LIT_BYTES 16u
#define DS_MIN_COUNT_UNCOMPRESSED_SEQ_ELEMS 4u

/* stage-2 constant kernels (zstdgpu_structs.h): the fused literal kernel processes
 * `kzstdgpu_StreamsPerGroup_DecompressLiterals` streams per threadgroup. On NVIDIA
 * this is the default width = kzstdgpu_TgSizeX_DecompressLiterals = 32. */
#define DS_DECOMPRESS_LITERALS_STREAMS_PER_GROUP 32u

#define TG_COUNT(n, tg) (((n) + (tg) - 1u) / (tg))
static uint32_t lookback_count(uint32_t n) { return TG_COUNT(n, DS_WAVE_SIZE_MIN); }

/* Phase 3 step 2: the fused LZ4 decode+unshuffle kernel's groupshared tile size
 * (kernels/Lz4DecodeUnshuffle.slang); full blocks bigger than this go unfused. */
#define SBZ_LZ4_FUSED_MAX_BLOCK 8192u

/* FSE table data layout (zstdgpu_structs.h). The FseElems array is laid out as:
 * [256 RLE tables][cmp * HufW][cmp * LLen][cmp * Offs][cmp * MLen][LLen][Offs][MLen],
 * with per-table capacity kzstdgpu_FseElemMaxCount_<kind>. InitFseTable writes one
 * table per threadgroup at the base + tableIndex*count; DecompressSequences reads it
 * back with zstdgpu_ComputeFseDataStartFromFseIndex*. */
static uint32_t fse_data_start_huw(uint32_t cmp) { return DS_FSE_RLE_TABLE_COUNT + 0u * cmp; }
static uint32_t fse_data_start_llen(uint32_t cmp) {
    return DS_FSE_RLE_TABLE_COUNT + cmp * DS_FSE_ELEM_MAX_HUFW;
}
static uint32_t fse_data_start_offs(uint32_t cmp) {
    return (DS_FSE_RLE_TABLE_COUNT + DS_FSE_ELEM_MAX_LLEN) +
           cmp * (DS_FSE_ELEM_MAX_HUFW + DS_FSE_ELEM_MAX_LLEN);
}
static uint32_t fse_data_start_mlen(uint32_t cmp) {
    return (DS_FSE_RLE_TABLE_COUNT + DS_FSE_ELEM_MAX_LLEN + DS_FSE_ELEM_MAX_OFFS) +
           cmp * (DS_FSE_ELEM_MAX_HUFW + DS_FSE_ELEM_MAX_LLEN + DS_FSE_ELEM_MAX_OFFS);
}

/* DS kzstdgpuFseProbsDefault (zstdgpu.cpp), widened to int32: our InitResources
 * port takes StructuredBuffer<int32_t> inFseProbsDefault. */
static const int32_t kFseProbsDefault[DS_FSE_DEFAULT_PROB_COUNT_LLEN + DS_FSE_DEFAULT_PROB_COUNT_OFFS +
                                      DS_FSE_DEFAULT_PROB_COUNT_MLEN] = {
    /* SEQ_LITERAL_LENGTH_DEFAULT_DIST */
    4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 3, 2, 1, 1, 1, 1, 1, -1, -1, -1, -1,
    /* SEQ_OFFSET_DEFAULT_DIST */
    1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1,
    /* SEQ_MATCH_LENGTH_DEFAULT_DIST */
    1, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1,
};

/* ---- buffer registry ----
 * X(name, element size in bytes, raw)
 * raw = byte-addressed data that kernels view as u8, u32, uint4 or
 * ByteAddressBuffer; such buffers are padded to 4 bytes and may be bound with element
 * size 1, 4 or 16 (uint4 views: the count is bytes/16, and the kernel only stores
 * 16-byte units at 16-byte aligned offsets inside the byte count).
 * Element sizes are OUR port's types (e.g. FseProbs is int32 here, int16 in DS). */
#define SBZ_BUFFER_LIST(X)                                                          \
    /* stage 0 */                                                                  \
    X(CompressedData, 4, 1)                                                        \
    X(FramesRefs, 8, 0)                                                            \
    X(FseProbsDefault, 4, 0)                                                       \
    X(Counters, 88, 0)                                                             \
    X(PerFrameBlockCountRAW, 4, 0)                                                 \
    X(PerFrameBlockCountRLE, 4, 0)                                                 \
    X(PerFrameBlockCountCMP, 4, 0)                                                 \
    X(PerFrameBlockCountAll, 4, 0)                                                 \
    X(PerFrameSeqStreamMinIdx, 4, 0)                                               \
    X(FrameStatus, 4, 0)                                                           \
    X(DispatchArgs, 4, 0) /* never read: we always pass workItemCount >= 0 */      \
    X(PerFrameBlockCountRAWLookback, 4, 0)                                         \
    X(PerFrameBlockCountRLELookback, 4, 0)                                         \
    X(PerFrameBlockCountCMPLookback, 4, 0)                                         \
    X(PerFrameBlockCountAllLookback, 4, 0)                                         \
    /* stage 1 */                                                                  \
    X(BlockSizePrefix, 4, 0)                                                       \
    X(GlobalBlockIndexPerRawBlock, 4, 0)                                           \
    X(GlobalBlockIndexPerRleBlock, 4, 0)                                           \
    X(GlobalBlockIndexPerCmpBlock, 4, 0)                                           \
    X(PerSeqStreamFinalOffset1, 4, 0)                                              \
    X(PerSeqStreamFinalOffset2, 4, 0)                                              \
    X(PerSeqStreamFinalOffset3, 4, 0)                                              \
    X(PerSeqStreamSeqStart, 4, 0)                                                  \
    X(RawBlockSizePrefix, 4, 0)                                                    \
    X(RleBlockSizePrefix, 4, 0)                                                    \
    X(FseProbs, 4, 0)                                                              \
    X(FseInfos, 4, 0)                                                              \
    X(FseElems, 4, 0)                                                              \
    X(DecompressedHuffmanWeights, 1, 1)                                            \
    X(DecompressedHuffmanWeightCount, 1, 1)                                        \
    X(LitRefs, 16, 0)                                                              \
    X(HufLitIdToHufWId_DBG, 4, 0)                                                  \
    X(HufWIdToHufLitId, 4, 0)                                                      \
    X(HufLitIdToLitStreamId, 4, 0)                                                 \
    X(SeqStreamToRef, 8, 0)                                                        \
    X(SeqStreamToLLenFseId, 4, 0)                                                  \
    X(SeqStreamToOffsFseId, 4, 0)                                                  \
    X(SeqStreamToMLenFseId, 4, 0)                                                  \
    X(SeqStreamToBlockId, 4, 0)                                                    \
    X(BlockSeqCountPrefixLookback, 4, 0)                                           \
    X(SeqCountPrefixLookback, 4, 0)                                                \
    X(LitStreamCountPrefixLookback, 4, 0)                                          \
    X(HufLitCompactionLookback, 4, 0)                                              \
    X(HufRefs, 8, 0)                                                               \
    X(CompressedBlocks, 16, 0)                                                     \
    X(FseIndexLookbackLLen, 4, 0)                                                  \
    X(FseIndexLookbackOffs, 4, 0)                                                  \
    X(FseIndexLookbackMLen, 4, 0)                                                  \
    X(LitGroupEndPerHuffmanTable, 4, 0)                                            \
    X(BlocksRLERefs, 8, 0)                                                         \
    X(BlocksRAWRefs, 8, 0)                                                         \
    X(BlocksCMPRefs, 8, 0)                                                         \
    X(BlockDestOffs, 4, 0)                                                         \
    X(BlockSizePrefixLookback, 4, 0)                                               \
    X(RawBlockSizePrefixLookback, 4, 0)                                            \
    X(RleBlockSizePrefixLookback, 4, 0)                                            \
    X(LitGroupEndPerHuffmanTableLookback, 4, 0)                                    \
    X(PerSeqStreamFinalOffset1Lookback, 4, 0)                                      \
    X(PerSeqStreamFinalOffset2Lookback, 4, 0)                                      \
    X(PerSeqStreamFinalOffset3Lookback, 4, 0)                                      \
    /* stage 2 */                                                                  \
    X(DecompressedLiterals, 1, 1)                                                  \
    X(DecompressedSequenceLLen, 4, 0)                                              \
    X(DecompressedSequenceMLen, 4, 0)                                              \
    X(DecompressedSequenceOffs, 4, 0)                                              \
    X(UnCompressedFramesData, 1, 1)                                                \
    X(UnCompressedFramesRefs, 8, 0)                                                \
    /* P3: bitshuffle output (untransposed elements + raw tail) */                 \
    X(BshufOutput, 1, 1)                                                           \
    /* bitshuffle-LZ4: block table (5 x uint32), its fused-path twin (outOffs in
     * BshufOutput space), and per-chunk decode status */                    \
    X(GzLz4BlockRef, 20, 0)                                                        \
    X(GzLz4BlockRefFused, 20, 0)                                                   \
    X(GzLz4Status, 4, 0)                                                           \
    /* batched unshuffle (Phase 3 step 3): uint2 {inByteOffset, outByteOffset} */   \
    X(GzUnshuffleRef, 8, 0)                                                        \
    /* P4 step 2c: bitshuffle chunk spans {offs, size} for the device chunk walk */ \
    X(ChunkSpans, 8, 0)

enum {
#define X(name, esz, raw) B_##name,
    SBZ_BUFFER_LIST(X)
#undef X
    B_COUNT
};

static const struct {
    const char *name;
    uint8_t esize;
    uint8_t raw;
} kBufInfo[B_COUNT] = {
#define X(name, esz, raw) { #name, esz, raw },
    SBZ_BUFFER_LIST(X)
#undef X
};

typedef struct sbz_buf {
    sbz_devptr ptr;
    size_t cap;   /* allocated bytes */
    size_t bytes; /* logical size for this request (count = bytes / esize) */
} sbz_buf;

struct sbz_dctx {
    sbz_backend *be;
    int own_be;
    sbz_buf buf[B_COUNT];
    sbz_counters counters;

    /* host scratch, pooled */
    /* zstd input: the caller's buffer, walked and uploaded where it is (no copy) */
    const uint8_t *src_ptr;
    size_t src_size;
    uint8_t src_tail[4]; /* the last partial word, zero-padded, for the upload */
    sbz_OffsetAndSize *frames;
    sbz_FrameInfo *frame_infos;
    size_t frames_cap;
    uint32_t *frame_status;

    /* per-request facts */
    uint32_t frame_count;
    uint32_t src_bytes_padded;
    sbz_CountFramesAndBlocksInfo host_count;
    int stage_done; /* last completed stage, -1 none */
    int lz4_fused;  /* 1: fused LZ4 decode+unshuffle kernel (default); 0: SLABITZ_LZ4_FUSED=0 */
    uint32_t lz4_fused_blocks, lz4_unfused_blocks; /* last bslz4 call's split (debug/tests) */
    /* P4 step 2a: a bslz4 call queued by sbz_dctx_bslz4_decompress_async and not yet
     * waited for; its per-chunk status (GzLz4Status, pending_nchunks entries) is
     * still on the device and is checked by sbz_dctx_wait. */
    int pending;
    uint32_t pending_nchunks;
    /* P4 step 2b (zstd). `exact`: this call reads the counters back after each
     * stage and sizes everything exactly (the first call of a context, a replay,
     * sbz_dctx_parse, or SLABITZ_CHECK_SYNC=1). Otherwise grids come from host
     * bounds, kernels read their counts on the device, and the literal and
     * sequence buffers have the capacities below (grown from exact calls); the
     * wait checks the counters once and replays the call exactly if they
     * overflowed or the input is corrupt (rare: the slow path). */
    int exact, force_exact, pending_zstd;
    uint32_t lit_cap, seq_cap;         /* capacities for the next non-exact call */
    uint32_t cur_lit_cap, cur_seq_cap; /* the buffers of the current call */
    uint32_t n_exact, n_replay;        /* exact calls and replays (sbz_dctx_zstd_stats) */
    /* P4 step 2c: a call on device-resident input. `geom` (during the call) says
     * the chunk geometry instead of a host walk; `ext_cd` is the caller's device
     * buffer, bound as CompressedData. */
    const sbz_bshuf_geom *geom;
    sbz_devptr ext_cd;
    size_t ext_cd_bytes;
    sbz_bshuf_geom rp_geom;
    sbz_devptr rp_dev;
    int rp_kind;                       /* call to replay: 1 decompress, 2 frames, 3 bshuf */
    const void *rp_src;
    size_t rp_size;
    const sbz_frame_span *rp_spans;
    uint32_t rp_n, rp_elem;
    /* page-locked staging for the library's own small uploads (block tables);
     * filled from offset 0 by each call, which only starts once the previous
     * call on this context is done (synchronous, or waited for) */
    uint8_t *stage;
    size_t stage_cap, stage_used;

    char err[320];
};

static int fail(sbz_dctx *dc, int code, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(dc->err, sizeof(dc->err), fmt, ap);
    va_end(ap);
    return code;
}

/* Propagates a backend error into dc->err. */
static int be_fail(sbz_dctx *dc, int code, const char *what)
{
    return fail(dc, code, "%s: %s", what, dc->be->err);
}

#define BE_TRY(expr, what)                                                        \
    do {                                                                          \
        int rc_ = (expr);                                                         \
        if (rc_ != SBZ_OK)                                                         \
            return be_fail(dc, rc_, what);                                        \
    } while (0)

#define TRY(expr)                                                                 \
    do {                                                                          \
        int rc_ = (expr);                                                         \
        if (rc_ != SBZ_OK)                                                         \
            return rc_;                                                           \
    } while (0)

/* ---- buffers ---- */

/* Sets buffer b's logical size to count elements, growing its allocation if
 * needed. Contents are NOT preserved or initialised (like DS's placed heaps):
 * every buffer is written by a memset or a kernel before it is read. */
static int buf_size(sbz_dctx *dc, int b, uint64_t count)
{
    sbz_buf *x = &dc->buf[b];
    uint64_t bytes = count * kBufInfo[b].esize;
    if (kBufInfo[b].raw)
        bytes = (bytes + 3u) & ~(uint64_t)3u;
    if (bytes > ((uint64_t)1 << 40))
        return fail(dc, SBZ_ERR_LIMIT, "buffer %s: %llu bytes", kBufInfo[b].name,
                    (unsigned long long)bytes);
    if (bytes > x->cap) {
        dc->be->ops->free(dc->be, x->ptr);
        x->ptr = 0;
        x->cap = 0;
        BE_TRY(dc->be->ops->alloc(dc->be, (size_t)bytes, &x->ptr), kBufInfo[b].name);
        x->cap = (size_t)bytes;
    }
    x->bytes = (size_t)bytes;
    return SBZ_OK;
}

static int buf_fill(sbz_dctx *dc, int b, uint32_t value)
{
    BE_TRY(dc->be->ops->fill32(dc->be, dc->buf[b].ptr, value, dc->buf[b].bytes / 4u),
           kBufInfo[b].name);
    return SBZ_OK;
}

static int buf_by_name(const char *name)
{
    int b;
    for (b = 0; b < B_COUNT; ++b)
        if (strcmp(kBufInfo[b].name, name) == 0)
            return b;
    return -1;
}

/* ---- launches ---- */

typedef struct { const char *param; int buf; } sbz_bind;
typedef struct { const char *param; uint32_t value; } sbz_scalar;

/* Kernel parameter name -> DS resource name: strip "inout"/"in", and the
 * "_Dwords" suffix of the u32 alias view of DecompressedLiterals. */
static int auto_buf(const char *param)
{
    char tmp[96];
    size_t n;
    if (strncmp(param, "inout", 5) == 0)
        param += 5;
    else if (strncmp(param, "in", 2) == 0)
        param += 2;
    n = strlen(param);
    if (n >= sizeof(tmp))
        return -1;
    memcpy(tmp, param, n + 1);
    if (n > 7 && strcmp(tmp + n - 7, "_Dwords") == 0)
        tmp[n - 7] = 0;
    return buf_by_name(tmp);
}

static int launch(sbz_dctx *dc, int k, uint32_t groups, const sbz_bind *binds, int nbinds,
                  const sbz_scalar *scalars, int nscalars)
{
    const sbz_kinfo *ki = &sbz_kernel_table[k];
    sbz_launch_arg args[64];
    unsigned used_b = 0, used_s = 0;
    uint32_t i;
    int j;

    if (ki->nparams > 64 || nbinds > 32 || nscalars > 32)
        return fail(dc, SBZ_ERR_ARGS, "%s: too many params", ki->name);
    for (i = 0; i < ki->nparams; ++i) {
        const sbz_kparam *p = &ki->params[i];
        memset(&args[i], 0, sizeof(args[i]));
        if (p->kind == SBZ_KP_U32) {
            for (j = 0; j < nscalars && strcmp(scalars[j].param, p->name) != 0; ++j)
                ;
            if (j == nscalars)
                return fail(dc, SBZ_ERR_ARGS, "%s: scalar %s not given", ki->name, p->name);
            used_s |= 1u << j;
            args[i].v.u32 = scalars[j].value;
        } else {
            int b = -1;
            const sbz_buf *x;
            for (j = 0; j < nbinds; ++j)
                if (strcmp(binds[j].param, p->name) == 0) {
                    b = binds[j].buf;
                    used_b |= 1u << j;
                    break;
                }
            if (b < 0)
                b = auto_buf(p->name);
            if (b < 0)
                return fail(dc, SBZ_ERR_ARGS, "%s: no buffer for %s", ki->name, p->name);
            if (p->esize != kBufInfo[b].esize &&
                !(kBufInfo[b].raw && (p->esize == 1 || p->esize == 4 || p->esize == 16)))
                return fail(dc, SBZ_ERR_ARGS, "%s: %s element size %u != buffer %s %u", ki->name,
                            p->name, p->esize, kBufInfo[b].name, kBufInfo[b].esize);
            x = &dc->buf[b];
            args[i].is_buf = 1;
            args[i].v.buf.ptr = x->ptr;
            args[i].v.buf.count = x->bytes / p->esize;
            if (b == B_CompressedData && dc->ext_cd) { /* the caller's device input */
                args[i].v.buf.ptr = dc->ext_cd;
                args[i].v.buf.count = dc->ext_cd_bytes / p->esize;
            }
        }
    }
    for (j = 0; j < nbinds; ++j)
        if (!(used_b & (1u << j)))
            return fail(dc, SBZ_ERR_ARGS, "%s: bind %s matches no param", ki->name, binds[j].param);
    for (j = 0; j < nscalars; ++j)
        if (!(used_s & (1u << j)))
            return fail(dc, SBZ_ERR_ARGS, "%s: scalar %s matches no param", ki->name,
                        scalars[j].param);
    if (groups == 0)
        return SBZ_OK; /* nothing to do (D3D12 Dispatch(0) is a no-op too) */
    BE_TRY(dc->be->ops->launch(dc->be, k, groups, args, ki->nparams), ki->name);
    return SBZ_OK;
}

/* One thread per work item, groups from the kernel's numthreads. */
static int launch_items(sbz_dctx *dc, int k, uint32_t items, const sbz_bind *binds, int nbinds,
                        const sbz_scalar *scalars, int nscalars)
{
    return launch(dc, k, TG_COUNT(items, sbz_kernel_table[k].block_x), binds, nbinds, scalars,
                  nscalars);
}

#define NB(a) ((int)(sizeof(a) / sizeof((a)[0])))

/* zstdgpu_Bind_PropagateFseIndex_{LLen,Offs,MLen}: propagate Repeat-mode FSE-table
 * indices across sequence streams, one per type. Grid = ceil(seqStreams / 32). */
/* `items` sizes the grid; `count` is the kernel's workItemCount (~0u: the kernel
 * reads Seq_Streams from Counters, P4 step 2b). */
static int propagate_fse_index(sbz_dctx *dc, const char *fseIdBuf, const char *lookbackBuf,
                               uint32_t items, uint32_t count)
{
    const sbz_bind b[] = { { "inoutFseIds", buf_by_name(fseIdBuf) },
                          { "inoutFseIndexLookback", buf_by_name(lookbackBuf) } };
    const sbz_scalar s[] = { { "tgOffset", 0 }, { "workItemCount", count } };
    return launch_items(dc, SBZK_PropagateFseIndex, items, b, NB(b), s, NB(s));
}

/* zstdgpu_Bind_PrefixSum_*: exclusive/inclusive prefix over `n` items, in place. */
static int prefix_sum(sbz_dctx *dc, int data, int lookback, uint32_t n, uint32_t inclusive)
{
    const sbz_bind b[] = { { "inoutInCountsOutPrefix", data },
                          { "inoutInCountsOutPrefixLookback", lookback } };
    const sbz_scalar s[] = { { "tgOffset", 0 }, { "workItemCount", n },
                            { "outputInclusive", inclusive } };
    return launch_items(dc, SBZK_PrefixSum, n, b, NB(b), s, NB(s));
}

/* zstdgpu_InitResources_GetDispatchSizeX, with our numthreads. */
static uint32_t init_resources_groups(uint32_t stage)
{
    uint32_t t = 1;
    if (stage == 1) {
        if (t < DS_FSE_DEFAULT_PROB_COUNT_LLEN) t = DS_FSE_DEFAULT_PROB_COUNT_LLEN;
        if (t < DS_FSE_DEFAULT_PROB_COUNT_OFFS) t = DS_FSE_DEFAULT_PROB_COUNT_OFFS;
        if (t < DS_FSE_DEFAULT_PROB_COUNT_MLEN) t = DS_FSE_DEFAULT_PROB_COUNT_MLEN;
        if (t < DS_FSE_RLE_TABLE_COUNT) t = DS_FSE_RLE_TABLE_COUNT;
    }
    return TG_COUNT(t, sbz_kernel_table[SBZK_InitResources].block_x);
}

static int read_counters(sbz_dctx *dc)
{
    BE_TRY(dc->be->ops->download(dc->be, &dc->counters, dc->buf[B_Counters].ptr,
                                 sizeof(sbz_counters)),
           "readback Counters");
    return SBZ_OK;
}

/* ---- host prep: zstdgpu_SetupInputsAsFramesInCpuMemory ---- */

/* spans == NULL: src is one or more back-to-back zstd frames.
 * spans != NULL: src holds nspans zstd frames at the given byte ranges, with
 * anything in between (e.g. bitshuffle's [BE u32 size] records). The frames are
 * copied back to back into the staging buffer (one host copy, as for the plain
 * case), so the same DS walk runs on them; it must then find exactly those
 * frames with exactly those sizes. */
/* Make room for `bytes` of staged uploads in this call (from offset 0). Called
 * before anything of the call is queued; the previous call on this context is
 * finished (check_idle), so the old staging memory may be freed. If page-locked
 * memory is not available, uploads fall back to the caller's memory. */
static void stage_reserve(sbz_dctx *dc, size_t bytes)
{
    dc->stage_used = 0;
    if (bytes <= dc->stage_cap)
        return;
    if (dc->stage)
        dc->be->ops->host_free(dc->be, dc->stage);
    dc->stage = NULL;
    dc->stage_cap = 0;
    bytes = (bytes + 0xFFFFu) & ~(size_t)0xFFFFu;
    if (dc->be->ops->host_alloc(dc->be, bytes, (void **)&dc->stage) == SBZ_OK && dc->stage)
        dc->stage_cap = bytes;
    else
        dc->stage = NULL;
}

/* Upload a host table through the staging memory when it fits (the copy is
 * then truly asynchronous and the table may be freed at once), else directly. */
static int upload_staged(sbz_dctx *dc, sbz_devptr dst, const void *src, size_t bytes)
{
    const size_t at = (dc->stage_used + 15u) & ~(size_t)15u;
    if (dc->stage && at + bytes <= dc->stage_cap) {
        memcpy(dc->stage + at, src, bytes);
        dc->stage_used = at + bytes;
        return dc->be->ops->upload(dc->be, dst, dc->stage + at, bytes);
    }
    return dc->be->ops->upload(dc->be, dst, src, bytes);
}

/* A call queued by sbz_dctx_bslz4_decompress_async must be waited for before the
 * context decodes again: the next call would overwrite its output and status. */
static int check_idle(sbz_dctx *dc)
{
    if (dc->pending)
        return fail(dc, SBZ_ERR_ARGS, "previous async call not waited for (sbz_dctx_wait)");
    return SBZ_OK;
}

/* zstd output per call (kzstdgpu_MaxRequestUncompressedByteSize) */
#define SBZ_MAX_CALL_OUTPUT 0x3FFFFFFFu

static int host_prepare(sbz_dctx *dc, const void *src, size_t srcSize,
                        const sbz_frame_span *spans, uint32_t nspans)
{
    size_t padded;
    sbz_CountFramesAndBlocksInfo ci;
    sbz_walk_error we;
    uint32_t i, nf;
    int rc;

    TRY(check_idle(dc));
    padded = (srcSize + 3u) & ~(size_t)3u;
    /* DS kernels use 32-bit byte offsets throughout */
    if (srcSize == 0 || padded > 0xFFFFFFF0u)
        return fail(dc, srcSize ? SBZ_ERR_LIMIT : SBZ_ERR_ARGS, "input size %zu", srcSize);
    if (spans) {
        if (nspans == 0)
            return fail(dc, SBZ_ERR_ARGS, "empty frame list");
        for (i = 0; i < nspans; ++i)
            if (spans[i].size == 0 || spans[i].offs > srcSize ||
                spans[i].size > srcSize - spans[i].offs)
                return fail(dc, SBZ_ERR_ARGS, "frame %u [%llu, +%llu) outside input of %zu bytes",
                            i, (unsigned long long)spans[i].offs,
                            (unsigned long long)spans[i].size, srcSize);
    }
    /* The walk reads bytes in place (any alignment, nothing past the size), and
     * the GPU reads the frames at their offsets in the uploaded caller's buffer:
     * no host copy of the input. */
    dc->src_ptr = (const uint8_t *)src;
    dc->src_size = srcSize;
    dc->src_bytes_padded = (uint32_t)padded;

    memset(&ci, 0, sizeof(ci));
    if (!spans) {
        memset(&we, 0, sizeof(we));
        rc = sbz_CountFramesAndBlocks(&ci, src, (uint32_t)srcSize, (uint32_t)srcSize, &we);
        if (rc != SBZ_OK)
            return fail(dc, rc, "%s", we.msg[0] ? we.msg : "invalid frame data");
        nf = ci.frameCount;
    } else {
        nf = nspans;
    }
    if (nf == 0)
        return fail(dc, SBZ_ERR_CORRUPT, "no zstd frame found");
    if (nf > dc->frames_cap) {
        sbz_OffsetAndSize *f = (sbz_OffsetAndSize *)realloc(dc->frames, nf * sizeof(*f));
        sbz_FrameInfo *fi;
        uint32_t *st;
        if (f)
            dc->frames = f;
        fi = (sbz_FrameInfo *)realloc(dc->frame_infos, nf * sizeof(*fi));
        if (fi)
            dc->frame_infos = fi;
        st = (uint32_t *)realloc(dc->frame_status, nf * sizeof(*st));
        if (st)
            dc->frame_status = st;
        if (!f || !fi || !st)
            return fail(dc, SBZ_ERR_NOMEM, "host frame arrays");
        dc->frames_cap = nf;
    }
    if (!spans) {
        memset(&we, 0, sizeof(we));
        rc = sbz_CollectFrames(dc->frames, dc->frame_infos, nf, src, (uint32_t)srcSize,
                              (uint32_t)srcSize, &we);
        if (rc != SBZ_OK)
            return fail(dc, rc, "%s", we.msg[0] ? we.msg : "invalid frame data");
    } else {
        /* each listed span must walk as exactly one frame of exactly its size */
        uint64_t raw = 0, rle = 0, cmp = 0, bytes = 0;
        for (i = 0; i < nspans; ++i) {
            const uint8_t *p = (const uint8_t *)src + spans[i].offs;
            const uint32_t n = (uint32_t)spans[i].size;
            sbz_CountFramesAndBlocksInfo one;
            memset(&one, 0, sizeof(one));
            memset(&we, 0, sizeof(we));
            rc = sbz_CountFramesAndBlocks(&one, p, n, n, &we);
            if (rc == SBZ_OK && one.frameCount != 1)
                return fail(dc, SBZ_ERR_CORRUPT, "frame %u: listed span walks as %u frames", i,
                            one.frameCount);
            if (rc == SBZ_OK)
                rc = sbz_CollectFrames(&dc->frames[i], &dc->frame_infos[i], 1, p, n, n, &we);
            if (rc != SBZ_OK)
                return fail(dc, rc, "frame %u: %s", i, we.msg[0] ? we.msg : "invalid frame data");
            if (dc->frames[i].size != n)
                return fail(dc, SBZ_ERR_CORRUPT, "frame %u: listed %u bytes, header walk %u", i,
                            n, dc->frames[i].size);
            dc->frames[i].offs += (uint32_t)spans[i].offs;
            raw += one.rawBlockCount;
            rle += one.rleBlockCount;
            cmp += one.cmpBlockCount;
            bytes += one.frameByteCount;
            if (raw > 0xFFFFFFF0u || rle > 0xFFFFFFF0u || cmp > 0xFFFFFFF0u ||
                bytes > 0xFFFFFFF0u)
                return fail(dc, SBZ_ERR_LIMIT, "frame %u: a 32-bit running total exceeds the limit",
                            i);
        }
        ci.rawBlockCount = (uint32_t)raw;
        ci.rleBlockCount = (uint32_t)rle;
        ci.cmpBlockCount = (uint32_t)cmp;
        ci.frameCount = nspans;
        ci.frameByteCount = bytes;
    }
    /* The GPU decodes at most kzstdgpu_MaxRequestUncompressedByteSize (0x3FFFFFFF) bytes
     * per call: block output offsets are prefix-summed with 30-bit lookback values.
     * Beyond it every frame would get status RequestTooLarge; say so before any GPU
     * work instead. */
    if (ci.frameByteCount > SBZ_MAX_CALL_OUTPUT)
        return fail(dc, SBZ_ERR_LIMIT, "call output %llu bytes exceeds 1 GiB (%u); split the request",
                    (unsigned long long)ci.frameByteCount, SBZ_MAX_CALL_OUTPUT);
    dc->host_count = ci;
    dc->frame_count = ci.frameCount;
    return SBZ_OK;
}

/* A per-frame GPU status (zstdgpu_shared_structs.h, kzstdgpu_FrameStatus_*, and the
 * T09 codes added by tools/transform_ds_for_slang.py) as the error message. */
static int frame_status_fail(sbz_dctx *dc, uint32_t f, uint32_t st)
{
    const char *what;
    switch (st) {
    case 0xA07A0001u: what = "not a zstd frame (NotZstdFrame)"; break;
    case 0xA07A0002u: what = "reserved header bit set (ReservedBitSet)"; break;
    case 0xA07A0003u: what = "dictionary (DictionaryUnsupported)"; break;
    case 0xA07A0004u: what = "window too large (WindowTooLarge)"; break;
    case 0xA07A0005u: what = "no content size (MissingContentSize)"; break;
    case 0xA07A0006u: what = "call output over 1 GiB (RequestTooLarge)"; break;
    case 0xA07A0007u: what = "decodes past its content size"; break;
    case 0xA07A0008u: what = "decodes short of its content size"; break;
    case 0xA07A0009u: what = "a block decodes to more than 128 KiB"; break;
    case 0xA07A000Au: what = "match offset before frame start"; break;
    default: what = "corrupt"; break;
    }
    return fail(dc, SBZ_ERR_CORRUPT, "frame %u: %s (status 0x%08x)", f, what, st);
}

static int check_frame_status(sbz_dctx *dc)
{
    uint32_t f;
    for (f = 0; f < dc->frame_count; ++f)
        if (dc->frame_status[f] != 0)
            return frame_status_fail(dc, f, dc->frame_status[f]);
    return SBZ_OK;
}

/* ---- Stage 0: zstdgpu_SubmitStage0 (count blocks per frame) ---- */

static int stage0(sbz_dctx *dc)
{
    const uint32_t nf = dc->frame_count;
    const uint32_t lb = lookback_count(nf);

    /* zstdgpu_ResourceInfo_Stage_0_InitSize */
    TRY(buf_size(dc, B_CompressedData, dc->src_bytes_padded / 4u));
    TRY(buf_size(dc, B_FramesRefs, nf));
    TRY(buf_size(dc, B_FseProbsDefault, NB(kFseProbsDefault)));
    TRY(buf_size(dc, B_Counters, 1));
    TRY(buf_size(dc, B_PerFrameBlockCountRAW, nf));
    TRY(buf_size(dc, B_PerFrameBlockCountRLE, nf));
    TRY(buf_size(dc, B_PerFrameBlockCountCMP, nf));
    TRY(buf_size(dc, B_PerFrameBlockCountAll, nf));
    TRY(buf_size(dc, B_PerFrameBlockCountRAWLookback, lb));
    TRY(buf_size(dc, B_PerFrameBlockCountRLELookback, lb));
    TRY(buf_size(dc, B_PerFrameBlockCountCMPLookback, lb));
    TRY(buf_size(dc, B_PerFrameBlockCountAllLookback, lb));
    TRY(buf_size(dc, B_PerFrameSeqStreamMinIdx, nf));
    TRY(buf_size(dc, B_FrameStatus, nf));
    TRY(buf_size(dc, B_DispatchArgs, 0));

    /* uploads */
    /* the caller's buffer as it is; its last partial word goes up zero-padded, so
     * the device buffer ends in zeros as the padded copy used to */
    {
        const size_t whole = dc->src_size & ~(size_t)3u;
        BE_TRY(dc->be->ops->upload(dc->be, dc->buf[B_CompressedData].ptr, dc->src_ptr, whole),
               "upload CompressedData");
        if (whole < dc->src_size) {
            memset(dc->src_tail, 0, sizeof(dc->src_tail));
            memcpy(dc->src_tail, dc->src_ptr + whole, dc->src_size - whole);
            BE_TRY(dc->be->ops->upload(dc->be, dc->buf[B_CompressedData].ptr + whole,
                                       dc->src_tail, 4),
                   "upload CompressedData tail");
        }
    }
    BE_TRY(dc->be->ops->upload(dc->be, dc->buf[B_FramesRefs].ptr, dc->frames,
                               (size_t)nf * sizeof(sbz_OffsetAndSize)),
           "upload FramesRefs");
    BE_TRY(dc->be->ops->upload(dc->be, dc->buf[B_FseProbsDefault].ptr, kFseProbsDefault,
                               sizeof(kFseProbsDefault)),
           "upload FseProbsDefault");

    /* [Init Resources :: Stage 0]: counters (stage-1 FSE buffers are unbound
     * here and untouched by the stage-0 path) */
    {
        const sbz_scalar s[] = { { "initResourcesStage", 0 } };
        TRY(launch(dc, SBZK_InitResources, init_resources_groups(0), NULL, 0, s, NB(s)));
    }
    /* [InitResources :: Memset :: Stage 0] */
    TRY(buf_fill(dc, B_PerFrameSeqStreamMinIdx, ~0u));
    TRY(buf_fill(dc, B_PerFrameBlockCountRAWLookback, 0));
    TRY(buf_fill(dc, B_PerFrameBlockCountRLELookback, 0));
    TRY(buf_fill(dc, B_PerFrameBlockCountCMPLookback, 0));
    TRY(buf_fill(dc, B_PerFrameBlockCountAllLookback, 0));
    /* [Parse Frames :: Count Blocks] */
    {
        const sbz_scalar s[] = { { "frameCount", nf },
                                { "compressedBufferSizeInBytes", dc->src_bytes_padded },
                                { "countBlocksOnly", 1 } };
        TRY(launch_items(dc, SBZK_ParseFrames, nf, NULL, 0, s, NB(s)));
    }
    /* [PrefixSum :: Block Counts] (exclusive) */
    TRY(prefix_sum(dc, B_PerFrameBlockCountRAW, B_PerFrameBlockCountRAWLookback, nf, 0));
    TRY(prefix_sum(dc, B_PerFrameBlockCountRLE, B_PerFrameBlockCountRLELookback, nf, 0));
    TRY(prefix_sum(dc, B_PerFrameBlockCountCMP, B_PerFrameBlockCountCMPLookback, nf, 0));
    TRY(prefix_sum(dc, B_PerFrameBlockCountAll, B_PerFrameBlockCountAllLookback, nf, 0));

    /* [Readback Counters :: After Block Count]. FrameStatus is not read here: the
     * count pass does not write it (only the collect pass in stage 1 does), so it
     * would be stale memory. A frame the count pass rejects contributes no blocks,
     * which the host-walk cross-check below catches. */
    if (!dc->exact) { /* checked once, at sbz_dctx_wait */
        dc->stage_done = 0;
        return SBZ_OK;
    }
    TRY(read_counters(dc));
    BE_TRY(dc->be->ops->sync(dc->be), "stage 0");
    /* the GPU walk and the host walk read the same headers: any difference is
     * corrupt input (or a bug), never something to size buffers from */
    if (dc->counters.Blocks_RAW != dc->host_count.rawBlockCount ||
        dc->counters.Blocks_RLE != dc->host_count.rleBlockCount ||
        dc->counters.Blocks_CMP != dc->host_count.cmpBlockCount)
        return fail(dc, SBZ_ERR_CORRUPT,
                    "block counts GPU raw/rle/cmp %u/%u/%u != host %u/%u/%u",
                    dc->counters.Blocks_RAW, dc->counters.Blocks_RLE, dc->counters.Blocks_CMP,
                    dc->host_count.rawBlockCount, dc->host_count.rleBlockCount,
                    dc->host_count.cmpBlockCount);
    dc->stage_done = 0;
    return SBZ_OK;
}

/* ---- Stage 1: zstdgpu_SubmitStage1 (collect blocks, parse compressed blocks) ---- */

static int stage1(sbz_dctx *dc)
{
    const uint32_t nf = dc->frame_count;
    /* the host walk's counts (an exact call has checked that the GPU's agree) */
    const uint32_t raw = dc->host_count.rawBlockCount;
    const uint32_t rle = dc->host_count.rleBlockCount;
    const uint32_t cmp = dc->host_count.cmpBlockCount;
    const uint64_t all = (uint64_t)raw + rle + cmp;
    const uint32_t lb_cmp = lookback_count(cmp);

    if (all > 0xFFFFFFFFu / 4u)
        return fail(dc, SBZ_ERR_LIMIT, "block count %llu", (unsigned long long)all);

    /* zstdgpu_ResourceInfo_Stage_1_InitSize */
    {
        const uint64_t seqFseElemMax = DS_FSE_ELEM_MAX_LLEN + DS_FSE_ELEM_MAX_OFFS +
                                       DS_FSE_ELEM_MAX_MLEN;
        const uint64_t nonRleFseTables = (uint64_t)cmp * 4u + 3u;
        const uint64_t fseTables = DS_FSE_RLE_TABLE_COUNT + nonRleFseTables;
        const uint64_t fseElems = DS_FSE_RLE_TABLE_COUNT +
                                  (uint64_t)cmp * (DS_FSE_ELEM_MAX_HUFW + seqFseElemMax) +
                                  seqFseElemMax;

        TRY(buf_size(dc, B_BlocksRAWRefs, raw));
        TRY(buf_size(dc, B_BlocksRLERefs, rle));
        TRY(buf_size(dc, B_BlocksCMPRefs, cmp));
        TRY(buf_size(dc, B_RawBlockSizePrefix, raw));
        TRY(buf_size(dc, B_RleBlockSizePrefix, rle));
        TRY(buf_size(dc, B_RawBlockSizePrefixLookback, lookback_count(raw)));
        TRY(buf_size(dc, B_RleBlockSizePrefixLookback, lookback_count(rle)));
        TRY(buf_size(dc, B_BlockSizePrefix, all));
        TRY(buf_size(dc, B_BlockSizePrefixLookback, lookback_count((uint32_t)all)));
        TRY(buf_size(dc, B_BlockDestOffs, all));
        TRY(buf_size(dc, B_GlobalBlockIndexPerRawBlock, raw));
        TRY(buf_size(dc, B_GlobalBlockIndexPerRleBlock, rle));
        TRY(buf_size(dc, B_GlobalBlockIndexPerCmpBlock, cmp));
        TRY(buf_size(dc, B_PerSeqStreamFinalOffset1, cmp));
        TRY(buf_size(dc, B_PerSeqStreamFinalOffset2, cmp));
        TRY(buf_size(dc, B_PerSeqStreamFinalOffset3, cmp));
        TRY(buf_size(dc, B_PerSeqStreamFinalOffset1Lookback, lb_cmp));
        TRY(buf_size(dc, B_PerSeqStreamFinalOffset2Lookback, lb_cmp));
        TRY(buf_size(dc, B_PerSeqStreamFinalOffset3Lookback, lb_cmp));
        TRY(buf_size(dc, B_PerSeqStreamSeqStart, cmp));
        TRY(buf_size(dc, B_FseInfos, fseTables));
        TRY(buf_size(dc, B_FseProbs, nonRleFseTables * DS_MAX_COUNT_FSE_PROBS));
        TRY(buf_size(dc, B_FseElems, fseElems));
        TRY(buf_size(dc, B_DecompressedHuffmanWeights, (uint64_t)cmp * DS_MAX_COUNT_HUFFMAN_WEIGHTS));
        TRY(buf_size(dc, B_DecompressedHuffmanWeightCount, cmp));
        TRY(buf_size(dc, B_HufRefs, cmp));
        TRY(buf_size(dc, B_LitRefs, (uint64_t)cmp * 4u));
        TRY(buf_size(dc, B_HufLitIdToHufWId_DBG, cmp));
        TRY(buf_size(dc, B_HufWIdToHufLitId, cmp));
        TRY(buf_size(dc, B_HufLitIdToLitStreamId, cmp));
        TRY(buf_size(dc, B_SeqStreamToRef, cmp));
        TRY(buf_size(dc, B_SeqStreamToLLenFseId, cmp));
        TRY(buf_size(dc, B_SeqStreamToOffsFseId, cmp));
        TRY(buf_size(dc, B_SeqStreamToMLenFseId, cmp));
        TRY(buf_size(dc, B_SeqStreamToBlockId, cmp));
        TRY(buf_size(dc, B_CompressedBlocks, cmp));
        TRY(buf_size(dc, B_FseIndexLookbackLLen, lb_cmp));
        TRY(buf_size(dc, B_FseIndexLookbackOffs, lb_cmp));
        TRY(buf_size(dc, B_FseIndexLookbackMLen, lb_cmp));
        TRY(buf_size(dc, B_LitGroupEndPerHuffmanTable, cmp));
        TRY(buf_size(dc, B_LitGroupEndPerHuffmanTableLookback, lb_cmp));
        TRY(buf_size(dc, B_BlockSeqCountPrefixLookback, lb_cmp));
        TRY(buf_size(dc, B_SeqCountPrefixLookback, lb_cmp));
        TRY(buf_size(dc, B_LitStreamCountPrefixLookback, lb_cmp));
        TRY(buf_size(dc, B_HufLitCompactionLookback, lb_cmp));
    }

    /* [Init Resources :: Stage 1]: default FSE infos/probs, RLE tables */
    {
        const sbz_scalar s[] = { { "initResourcesStage", 1 } };
        TRY(launch(dc, SBZK_InitResources, init_resources_groups(1), NULL, 0, s, NB(s)));
    }
    /* [InitResources :: Memset :: Stage 1] (+ BlockSize Lookback) */
    TRY(buf_fill(dc, B_RawBlockSizePrefixLookback, 0));
    TRY(buf_fill(dc, B_RleBlockSizePrefixLookback, 0));
    TRY(buf_fill(dc, B_LitGroupEndPerHuffmanTableLookback, 0));
    TRY(buf_fill(dc, B_PerSeqStreamFinalOffset1Lookback, 0));
    TRY(buf_fill(dc, B_PerSeqStreamFinalOffset2Lookback, 0));
    TRY(buf_fill(dc, B_PerSeqStreamFinalOffset3Lookback, 0));
    TRY(buf_fill(dc, B_SeqCountPrefixLookback, 0));
    TRY(buf_fill(dc, B_BlockSeqCountPrefixLookback, 0));
    TRY(buf_fill(dc, B_LitStreamCountPrefixLookback, 0));
    TRY(buf_fill(dc, B_HufLitCompactionLookback, 0));
    TRY(buf_fill(dc, B_FseIndexLookbackLLen, 0));
    TRY(buf_fill(dc, B_FseIndexLookbackOffs, 0));
    TRY(buf_fill(dc, B_FseIndexLookbackMLen, 0));
    TRY(buf_fill(dc, B_HufWIdToHufLitId, ~0u));
    TRY(buf_fill(dc, B_BlockSizePrefixLookback, 0));

    /* [Parse Frames :: Collect Blocks] */
    {
        const sbz_scalar s[] = { { "frameCount", nf },
                                { "compressedBufferSizeInBytes", dc->src_bytes_padded },
                                { "countBlocksOnly", 0 } };
        TRY(launch_items(dc, SBZK_ParseFrames, nf, NULL, 0, s, NB(s)));
    }
    /* [Parse Compressed Blocks] */
    {
        const sbz_scalar s[] = { { "tgOffset", 0 },
                                { "workItemCount", cmp },
                                { "compressedBufferSizeInBytes", dc->src_bytes_padded },
                                { "frameCount", nf },
                                { "compressedBlockCount", cmp } };
        TRY(launch_items(dc, SBZK_ParseCompressedBlocks, cmp, NULL, 0, s, NB(s)));
    }
    /* [Prefix RAW/RLE Block Sizes] (exclusive) */
    TRY(prefix_sum(dc, B_RawBlockSizePrefix, B_RawBlockSizePrefixLookback, raw, 0));
    TRY(prefix_sum(dc, B_RleBlockSizePrefix, B_RleBlockSizePrefixLookback, rle, 0));

    if (!dc->exact) {
        /* [Propagate FSE Index] x3 over at most one sequence stream per compressed
         * block; the kernel reads Seq_Streams on the device. No readback. */
        TRY(propagate_fse_index(dc, "SeqStreamToLLenFseId", "FseIndexLookbackLLen", cmp, ~0u));
        TRY(propagate_fse_index(dc, "SeqStreamToOffsFseId", "FseIndexLookbackOffs", cmp, ~0u));
        TRY(propagate_fse_index(dc, "SeqStreamToMLenFseId", "FseIndexLookbackMLen", cmp, ~0u));
        dc->stage_done = 1;
        return SBZ_OK;
    }
    /* [Readback Counters :: After Block Parse] */
    TRY(read_counters(dc));
    /* [Propagate FSE Index] x3: then in-place propagate Repeat-mode FSE-table indices
     * across sequence streams (workItemCount = Seq_Streams). Does not touch counters. */
    TRY(propagate_fse_index(dc, "SeqStreamToLLenFseId", "FseIndexLookbackLLen",
                            dc->counters.Seq_Streams, dc->counters.Seq_Streams));
    TRY(propagate_fse_index(dc, "SeqStreamToOffsFseId", "FseIndexLookbackOffs",
                            dc->counters.Seq_Streams, dc->counters.Seq_Streams));
    TRY(propagate_fse_index(dc, "SeqStreamToMLenFseId", "FseIndexLookbackMLen",
                            dc->counters.Seq_Streams, dc->counters.Seq_Streams));
    /* per-frame status, written by the collect pass of ParseFrames above */
    BE_TRY(dc->be->ops->download(dc->be, dc->frame_status, dc->buf[B_FrameStatus].ptr,
                                 (size_t)nf * 4u),
           "readback FrameStatus");
    BE_TRY(dc->be->ops->sync(dc->be), "stage 1");
    TRY(check_frame_status(dc));
    dc->stage_done = 1;
    return SBZ_OK;
}

/* ---- Stage 2: FSE/huffman/literal/sequence decode + execute ----
 * Follows zstdgpu_SubmitStage2 (zstdgpu.cpp:2879). DS drives every grid via indirect
 * dispatch through UpdateDispatchArgs; here the host reads Counters back and sizes the
 * grids itself (readback mode). The kernel-to-grid mapping is the one in
 * ZstdGpuUpdateDispatchArgs.hlsl: workItemCount per slot is the counter value, and
 * threadgroup size is the kernel's numthreads (the one exception is InitFseTable, which
 * builds one table per threadgroup, so grid = table count, not ceil(count/numthreads)). */

/* `groups`: one threadgroup per table; `count` the kernel's workItemCount (~0u: the
 * table count of this type is read on the device and extra groups exit). */
static int init_fse_table(sbz_dctx *dc, uint32_t groups, uint32_t count, uint32_t tableType,
                          uint32_t startIdx, uint32_t dataStart, uint32_t dataCount)
{
    const sbz_scalar s[] = { { "tgOffset", 0 },       { "workItemCount", count },
                            { "tableType", tableType }, { "tableStartIndex", startIdx },
                            { "tableDataStart", dataStart }, { "tableDataCount", dataCount } };
    return launch(dc, SBZK_InitFseTable, groups, NULL, 0, s, NB(s));
}

/* capacity for the next non-exact call: the count seen plus a quarter */
static uint32_t grow_cap(uint32_t cap, uint32_t seen, uint32_t minimum)
{
    uint64_t want = (uint64_t)seen + seen / 4u;
    if (want < minimum)
        want = minimum;
    if (want > 0xFFFFFFF0u)
        want = 0xFFFFFFF0u;
    return (uint32_t)want > cap ? (uint32_t)want : cap;
}

static int stage2(sbz_dctx *dc)
{
    const uint32_t nf = dc->frame_count;
    const uint32_t cmp = dc->host_count.cmpBlockCount;
    const uint32_t raw = dc->host_count.rawBlockCount;
    const uint32_t rle = dc->host_count.rleBlockCount;
    const uint32_t all = raw + rle + cmp;
    const int exact = dc->exact;
    /* exact: the counts read back after stage 1; otherwise host bounds (at most
     * one sequence stream per compressed block) and the context's capacities */
    const uint32_t seqStreams = exact ? dc->counters.Seq_Streams : cmp;
    uint32_t cntLit = exact ? dc->counters.HUF_Streams_DecodedBytes : dc->lit_cap;
    uint32_t cntSeq = exact ? dc->counters.Seq_Streams_DecodedItems : dc->seq_cap;
    uint32_t seqItems;
    uint64_t totalUncomp = 0;
    uint32_t f;

    if (cntLit < DS_MIN_COUNT_UNCOMPRESSED_LIT_BYTES)
        cntLit = DS_MIN_COUNT_UNCOMPRESSED_LIT_BYTES;
    if (cntSeq < DS_MIN_COUNT_UNCOMPRESSED_SEQ_ELEMS)
        cntSeq = DS_MIN_COUNT_UNCOMPRESSED_SEQ_ELEMS;
    seqItems = exact ? dc->counters.Seq_Streams_DecodedItems : cntSeq;
    dc->cur_lit_cap = cntLit;
    dc->cur_seq_cap = cntSeq;
    for (f = 0; f < nf; ++f)
        totalUncomp += dc->frame_infos[f].uncompSize;
    /* DS kernels address the output with 32-bit byte offsets */
    if (totalUncomp > 0xFFFFFFFFull)
        return fail(dc, SBZ_ERR_LIMIT, "total uncompressed size %llu > 4 GiB; split the request",
                    (unsigned long long)totalUncomp);

    /* zstdgpu_ResourceInfo_Stage_2_InitSize (UnCompressedFramesData/Refs are external). */
    TRY(buf_size(dc, B_DecompressedLiterals, cntLit));
    TRY(buf_size(dc, B_DecompressedSequenceLLen, cntSeq));
    TRY(buf_size(dc, B_DecompressedSequenceMLen, cntSeq));
    TRY(buf_size(dc, B_DecompressedSequenceOffs, cntSeq));
    TRY(buf_size(dc, B_UnCompressedFramesData, totalUncomp));
    TRY(buf_size(dc, B_UnCompressedFramesRefs, nf));

    /* UnCompressedFramesRefs[f] = {dest offset of frame f, uncompressed size}. */
    {
        sbz_OffsetAndSize *refs = (sbz_OffsetAndSize *)malloc((size_t)nf * sizeof(*refs));
        uint64_t offs = 0;
        if (!refs)
            return fail(dc, SBZ_ERR_NOMEM, "UnCompressedFramesRefs host array");
        for (f = 0; f < nf; ++f) {
            refs[f].offs = (uint32_t)(offs & 0xFFFFFFFFu);
            refs[f].size = (uint32_t)(dc->frame_infos[f].uncompSize & 0xFFFFFFFFu);
            offs += dc->frame_infos[f].uncompSize;
        }
        BE_TRY(dc->be->ops->upload(dc->be, dc->buf[B_UnCompressedFramesRefs].ptr, refs,
                                   (size_t)nf * sizeof(*refs)),
               "upload UnCompressedFramesRefs");
        free(refs);
    }

    /* [Compute `Per-Huffman Table` Literal Stream Count Prefix] -> DecompressLiteralsGroups */
    {
        const sbz_scalar s[] = { { "tgOffset", 0 },
                                { "workItemCount", cmp },
                                { "literalsPerGroup", DS_DECOMPRESS_LITERALS_STREAMS_PER_GROUP } };
        TRY(launch_items(dc, SBZK_ComputePrefixSum, cmp, NULL, 0, s, NB(s)));
    }
    if (getenv("SLABITZ_DEBUG_SYNC")) BE_TRY(dc->be->ops->sync(dc->be), "chk:ComputePrefixSum");
    if (exact)
        TRY(read_counters(dc)); /* pick up DecompressLiteralsGroups */

    /* [Init FSE Table] x4 (one threadgroup per table; grid = table count). weight/HufW
     * starts at index 0, LLen at cmp, Offs at 2*cmp+1, MLen at 3*cmp+2. Not exact:
     * at most one table of each kind per compressed block (+1 for the sequence
     * kinds' windows), the kernel reads the real count. */
    if (exact) {
        TRY(init_fse_table(dc, dc->counters.FseHufW, dc->counters.FseHufW, 0u, 0u, fse_data_start_huw(cmp), DS_FSE_ELEM_MAX_HUFW));
        TRY(init_fse_table(dc, dc->counters.FseLLen, dc->counters.FseLLen, 1u, cmp, fse_data_start_llen(cmp), DS_FSE_ELEM_MAX_LLEN));
        TRY(init_fse_table(dc, dc->counters.FseOffs, dc->counters.FseOffs, 2u, 2u * cmp + 1u, fse_data_start_offs(cmp), DS_FSE_ELEM_MAX_OFFS));
        TRY(init_fse_table(dc, dc->counters.FseMLen, dc->counters.FseMLen, 3u, 3u * cmp + 2u, fse_data_start_mlen(cmp), DS_FSE_ELEM_MAX_MLEN));
    } else {
        TRY(init_fse_table(dc, cmp, ~0u, 0u, 0u, fse_data_start_huw(cmp), DS_FSE_ELEM_MAX_HUFW));
        TRY(init_fse_table(dc, cmp + 1u, ~0u, 1u, cmp, fse_data_start_llen(cmp), DS_FSE_ELEM_MAX_LLEN));
        TRY(init_fse_table(dc, cmp + 1u, ~0u, 2u, 2u * cmp + 1u, fse_data_start_offs(cmp), DS_FSE_ELEM_MAX_OFFS));
        TRY(init_fse_table(dc, cmp + 1u, ~0u, 3u, 3u * cmp + 2u, fse_data_start_mlen(cmp), DS_FSE_ELEM_MAX_MLEN));
    }
    if (getenv("SLABITZ_DEBUG_SYNC")) BE_TRY(dc->be->ops->sync(dc->be), "chk:InitFseTable");

    /* [Decompress Huffman Weights] (FSE-compressed) and [Decode Uncompressed Huffman Weights]. */
    {
        /* the kernel bounds itself by Counters.FseHufW (at most one per block) */
        const uint32_t n = exact ? dc->counters.FseHufW : cmp;
        const sbz_scalar s[] = { { "tgOffset", 0 }, { "workItemCount", n } };
        TRY(launch_items(dc, SBZK_DecompressHuffmanWeights, n, NULL, 0, s, NB(s)));
    }
    {
        /* the kernel bounds itself by Counters.HUF_WgtStreams (at most one per block) */
        const uint32_t n = exact ? dc->counters.HUF_WgtStreams : cmp;
        const sbz_scalar s[] = { { "tgOffset", 0 },
                                { "workItemCount", n },
                                { "compressedBufferSizeInBytes", dc->src_bytes_padded } };
        TRY(launch_items(dc, SBZK_DecodeHuffmanWeights, n, NULL, 0, s, NB(s)));
    }
    if (getenv("SLABITZ_DEBUG_SYNC")) BE_TRY(dc->be->ops->sync(dc->be), "chk:HuffWeights");

    /* [Decompress Literals] (grid = DecompressLiteralsGroups). */
    {
        /* Not exact: a bound on DecompressLiteralsGroups (each Huffman table, at most
         * one per block, gets ceil(its streams / per group) groups, and there are at
         * most 4 streams per block); the entry exits above the real count. */
        const uint32_t n = exact ? dc->counters.DecompressLiteralsGroups
                                 : cmp + TG_COUNT(4u * cmp, DS_DECOMPRESS_LITERALS_STREAMS_PER_GROUP);
        const sbz_scalar s[] = { { "tgOffset", 0 },
                                { "workItemCount", n },
                                { "streamsPerGroup", DS_DECOMPRESS_LITERALS_STREAMS_PER_GROUP },
                                { "litCapacity", cntLit } };
        TRY(launch(dc, SBZK_InitHuffmanTableAndDecompressLiterals, n, NULL, 0, s, NB(s)));
    }
    if (getenv("SLABITZ_DEBUG_SYNC")) BE_TRY(dc->be->ops->sync(dc->be), "chk:Literals");

    /* [Decompress Sequences] (reads BlockSizePrefix as per-block literal size). */
    {
        const sbz_scalar s[] = { { "tgOffset", 0 }, { "workItemCount", seqStreams },
                                { "seqCapacity", cntSeq } };
        TRY(launch_items(dc, SBZK_DecompressSequencesMultiStream, seqStreams, NULL, 0, s, NB(s)));
    }
    if (getenv("SLABITZ_DEBUG_SYNC")) BE_TRY(dc->be->ops->sync(dc->be), "chk:Sequences");

    /* [Prefix Block Sizes] (inclusive) now turns BlockSizePrefix into cumulative-inclusive. */
    TRY(prefix_sum(dc, B_BlockSizePrefix, B_BlockSizePrefixLookback, all, 1));

    /* [Prefix Sequence Offsets]. */
    {
        const sbz_scalar s[] = { { "tgOffset", 0 }, { "workItemCount", seqStreams },
                                { "frameCount", nf } };
        TRY(launch_items(dc, SBZK_PrefixSequenceOffsets, seqStreams, NULL, 0, s, NB(s)));
    }
    if (getenv("SLABITZ_DEBUG_SYNC")) BE_TRY(dc->be->ops->sync(dc->be), "chk:PrefixSeqOffsets");

    /* [Compute Dest Block Offsets]. */
    {
        const sbz_scalar s[] = { { "tgOffset", 0 }, { "workItemCount", all }, { "frameCount", nf } };
        TRY(launch_items(dc, SBZK_ComputeDestBlockOffsets, all, NULL, 0, s, NB(s)));
    }

    /* [Finalise Sequence Offsets]. */
    {
        const sbz_scalar s[] = { { "tgOffset", 0 }, { "workItemCount", seqItems },
                                { "seqCapacity", cntSeq } };
        TRY(launch_items(dc, SBZK_FinaliseSequenceOffsets, seqItems, NULL, 0, s, NB(s)));
    }
    if (getenv("SLABITZ_DEBUG_SYNC")) BE_TRY(dc->be->ops->sync(dc->be), "chk:Finalise");

    /* [Memcpy RAW blocks, Memset RLE blocks]: one 256-thread group per block (grid.x =
     * block count; the kernel has no binary search). Bytes are stored through the u8 view
     * at block edges and as words through a second u32 view of the same buffer. */
    {
        const sbz_bind b[] = { { "inBlocksRefs", B_BlocksRAWRefs },
                              { "inGlobalBlockIndex", B_GlobalBlockIndexPerRawBlock },
                              { "inoutUnCompressedFramesData32", B_UnCompressedFramesData } };
        const sbz_scalar s[] = { { "tgOffset", 0 }, { "flags", 1 } };
        TRY(launch(dc, SBZK_MemsetMemcpy, raw, b, NB(b), s, NB(s)));
    }
    {
        const sbz_bind b[] = { { "inBlocksRefs", B_BlocksRLERefs },
                              { "inGlobalBlockIndex", B_GlobalBlockIndexPerRleBlock },
                              { "inoutUnCompressedFramesData32", B_UnCompressedFramesData } };
        const sbz_scalar s[] = { { "tgOffset", 0 }, { "flags", 0 } };
        TRY(launch(dc, SBZK_MemsetMemcpy, rle, b, NB(b), s, NB(s)));
    }
    if (getenv("SLABITZ_DEBUG_SYNC")) BE_TRY(dc->be->ops->sync(dc->be), "chk:MemsetMemcpy");

    /* [Execute Sequences] (one threadgroup per frame). */
    {
        const sbz_bind b[] = { { "inoutUnCompressedFramesData32", B_UnCompressedFramesData },
                               { "inoutUnCompressedFramesData16", B_UnCompressedFramesData } };
        const sbz_scalar s[] = { { "litCapacity", cntLit }, { "seqCapacity", cntSeq } };
        TRY(launch(dc, SBZK_ExecuteSequences, nf, b, NB(b), s, NB(s)));
    }
    if (getenv("SLABITZ_DEBUG_SYNC")) BE_TRY(dc->be->ops->sync(dc->be), "chk:Execute");

    if (!exact) { /* queued only: sbz_dctx_wait checks the counters */
        dc->pending = 1;
        dc->pending_zstd = 1;
        dc->stage_done = 2;
        return SBZ_OK;
    }
    /* [Readback Counters :: After Block Decompression], and the per-frame status:
     * ComputeDestBlockOffsets and ExecuteSequences report frames that do not fit
     * their content size or reach before their start (T09) */
    TRY(read_counters(dc));
    BE_TRY(dc->be->ops->download(dc->be, dc->frame_status, dc->buf[B_FrameStatus].ptr,
                                 (size_t)nf * 4u),
           "readback FrameStatus");
    BE_TRY(dc->be->ops->sync(dc->be), "stage 2");
    TRY(check_frame_status(dc));
    /* capacities for the next calls of this context, from what this one needed */
    dc->lit_cap = grow_cap(dc->lit_cap, dc->counters.HUF_Streams_DecodedBytes,
                           DS_MIN_COUNT_UNCOMPRESSED_LIT_BYTES);
    dc->seq_cap = grow_cap(dc->seq_cap, dc->counters.Seq_Streams_DecodedItems,
                           DS_MIN_COUNT_UNCOMPRESSED_SEQ_ELEMS);
    dc->stage_done = 2;
    return SBZ_OK;
}

/* ---- public API ---- */

sbz_dctx *sbz_dctx_create(sbz_backend *be, int own_backend)
{
    sbz_dctx *dc;
    if (!be)
        return NULL;
    dc = (sbz_dctx *)calloc(1, sizeof(*dc));
    if (!dc) {
        if (own_backend)
            be->ops->destroy(be);
        return NULL;
    }
    dc->be = be;
    dc->own_be = own_backend;
    dc->stage_done = -1;
    dc->lz4_fused = 1;
    if (getenv("SLABITZ_LZ4_FUSED") && atoi(getenv("SLABITZ_LZ4_FUSED")) == 0)
        dc->lz4_fused = 0;
    return dc;
}

sbz_dctx *sbz_dctx_create_cuda(int device, const char *kernel_dir, char *errbuf, size_t errlen)
{
    sbz_backend *be = sbz_cuda_create(device, kernel_dir, errbuf, errlen);
    sbz_dctx *dc = sbz_dctx_create(be, 1);
    if (be && !dc && errbuf && errlen)
        snprintf(errbuf, errlen, "out of host memory");
    return dc;
}

void sbz_dctx_destroy(sbz_dctx *dc)
{
    int b;
    if (!dc)
        return;
    for (b = 0; b < B_COUNT; ++b)
        dc->be->ops->free(dc->be, dc->buf[b].ptr);
    if (dc->stage)
        dc->be->ops->host_free(dc->be, dc->stage);
    if (dc->own_be)
        dc->be->ops->destroy(dc->be);
    free(dc->frames);
    free(dc->frame_infos);
    free(dc->frame_status);
    free(dc);
}

const char *sbz_dctx_error(const sbz_dctx *dc)
{
    return dc ? dc->err : "null context";
}

int sbz_dctx_parse(sbz_dctx *dc, const void *src, size_t srcSize)
{
    if (!dc || !src)
        return SBZ_ERR_ARGS;
    dc->err[0] = 0;
    dc->stage_done = -1;
    TRY(host_prepare(dc, src, srcSize, NULL, 0));
    dc->exact = 1; /* a debug call: the counters are read after each stage */
    TRY(stage0(dc));
    TRY(stage1(dc));
    return SBZ_OK;
}

static int decompress_frames_internal(sbz_dctx *dc, const void *src, size_t srcSize,
                                      const sbz_frame_span *frames, uint32_t nframes);

/* A call that may have been queued without the exact checks waits for itself. */
static int finish(sbz_dctx *dc)
{
    return dc->pending ? sbz_dctx_wait(dc) : SBZ_OK;
}

int sbz_dctx_decompress(sbz_dctx *dc, const void *src, size_t srcSize)
{
    if (!dc || !src)
        return SBZ_ERR_ARGS;
    dc->err[0] = 0;
    dc->stage_done = -1;
    TRY(check_idle(dc));
    dc->rp_kind = 1;
    dc->rp_src = src;
    dc->rp_size = srcSize;
    TRY(decompress_frames_internal(dc, src, srcSize, NULL, 0));
    return finish(dc);
}

static int decompress_frames_internal(sbz_dctx *dc, const void *src, size_t srcSize,
                                      const sbz_frame_span *frames, uint32_t nframes)
{
    TRY(host_prepare(dc, src, srcSize, frames, nframes));
    /* exact (readback) mode on a context's first call, on a replay, and on demand;
     * otherwise no host wait before sbz_dctx_wait (P4 step 2b) */
    dc->exact = dc->force_exact || dc->lit_cap == 0 || dc->seq_cap == 0 ||
                (getenv("SLABITZ_CHECK_SYNC") && atoi(getenv("SLABITZ_CHECK_SYNC")) > 0);
    dc->n_exact += dc->exact ? 1u : 0u;
    TRY(stage0(dc));
    TRY(stage1(dc));
    TRY(stage2(dc));
    return SBZ_OK;
}

int sbz_dctx_decompress_frames(sbz_dctx *dc, const void *src, size_t srcSize,
                              const sbz_frame_span *frames, uint32_t nframes)
{
    if (!dc || !src || !frames)
        return SBZ_ERR_ARGS;
    dc->err[0] = 0;
    dc->stage_done = -1;
    TRY(check_idle(dc));
    dc->rp_kind = 2;
    dc->rp_src = src;
    dc->rp_size = srcSize;
    dc->rp_spans = frames;
    dc->rp_n = nframes;
    TRY(decompress_frames_internal(dc, src, srcSize, frames, nframes));
    return finish(dc);
}

/* ---- P3: bitshuffle-zstd chunks (bshuf_h5filter.c / bitshuffle_core.c) ---- */

static uint64_t sbz_be64(const uint8_t *p)
{
    return ((uint64_t)p[0] << 56) | ((uint64_t)p[1] << 48) | ((uint64_t)p[2] << 40) |
           ((uint64_t)p[3] << 32) | ((uint64_t)p[4] << 24) | ((uint64_t)p[5] << 16) |
           ((uint64_t)p[6] << 8) | (uint64_t)p[7];
}

static uint32_t sbz_be32(const uint8_t *p)
{
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}

/* Per-chunk decode layout (host walk of a bitshuffle-zstd chunk). Sizes in
 * elements, e = elemSize: [BE u64 total][BE u32 block bytes]; B = block/e = the
 * block's element count (a multiple of 8); S = total/e. floor(S/B) full blocks of
 * B elements then one block of (S%B) - (S%B)%8 elements if nonzero; each block
 * [BE u32 c][c bytes = one zstd frame] decoding to (elems x e) bit-transposed
 * bytes; then (S%8) x e raw bytes, not transposed. */
typedef struct sbz_bshuf_chunk {
    uint64_t inOffs;      /* chunk's frames byte offset in UnCompressedFramesData */
    uint64_t outOffs;     /* chunk's output byte offset in BshufOutput */
    uint64_t frameBytes;  /* decoded bytes of the frames (nfull*B*e + last2*e) */
    uint64_t tailBytes;   /* (S%8)*e raw bytes, stored untransposed after frameBytes */
    uint64_t blockBytes;  /* decoded bytes of one full block = B*e */
    uint64_t tailOff;     /* src byte offset of the raw tail */
    uint32_t nfull;       /* floor(S/B) */
    uint32_t lastElems;   /* (S%B)-(S%B)%8, 0 if none */
    uint32_t blockElems;  /* B */
} sbz_bshuf_chunk;

/* P4 step 2c: a chunk's layout from the stated geometry (the rules of the host
 * walk below, without reading the chunk). SBZ_ERR_ARGS if the geometry is not
 * a valid bitshuffle layout. */
static int bshuf_geom_chunk(sbz_dctx *dc, const sbz_bshuf_geom *g, sbz_bshuf_chunk *ci)
{
    const uint32_t e = g->elemSize;
    uint64_t S, B, nfull, last2;
    if (e == 0 || e > 8 || g->chunkBytes % e || g->blockBytes == 0 || g->blockBytes % e ||
        (g->blockBytes / e) % 8u || g->chunkBytes > 0xFFFFFFFFull)
        return fail(dc, SBZ_ERR_ARGS, "geometry e=%u chunk %llu block %u is not a bitshuffle "
                    "layout", e, (unsigned long long)g->chunkBytes, g->blockBytes);
    S = g->chunkBytes / e;
    B = g->blockBytes / e;
    nfull = S / B;
    last2 = (S - nfull * B) - (S - nfull * B) % 8u;
    memset(ci, 0, sizeof(*ci));
    ci->frameBytes = (nfull * B + last2) * e;
    ci->tailBytes = (S % 8u) * e;
    ci->blockBytes = B * e;
    ci->nfull = (uint32_t)nfull;
    ci->lastElems = (uint32_t)last2;
    ci->blockElems = (uint32_t)B;
    return SBZ_OK;
}

/* Host walk of one chunk: header, every block record, and the raw tail.
 * Returns SBZ_OK and fills *ci, or a SBZ_ERR_* code on corrupt/failed input. */
static int bshuf_walk_chunk(sbz_dctx *dc, const uint8_t *c, uint64_t clen, uint64_t base,
                            uint32_t elemSize, sbz_bshuf_chunk *ci)
{
    uint64_t total, S, B, nfull, last, last2, pos, frameBytes, tail;
    uint32_t blockBytes, k, nf;

    if (clen < 12)
        return fail(dc, SBZ_ERR_CORRUPT, "chunk too short (%llu bytes)", (unsigned long long)clen);
    total = sbz_be64(c);
    blockBytes = sbz_be32(c + 8);
    if (total % elemSize)
        return fail(dc, SBZ_ERR_CORRUPT, "chunk total %llu not a multiple of e=%u",
                    (unsigned long long)total, elemSize);
    if (blockBytes == 0 || blockBytes % elemSize)
        return fail(dc, SBZ_ERR_CORRUPT, "block bytes %u not a multiple of e=%u", blockBytes,
                    elemSize);
    S = total / elemSize;
    B = blockBytes / elemSize;
    if (B == 0 || B % 8)
        return fail(dc, SBZ_ERR_CORRUPT, "block elems %llu not a multiple of 8",
                    (unsigned long long)B);
    nfull = S / B;
    last = S - nfull * B;
    last2 = last - last % 8;
    nf = (uint32_t)(nfull + (last2 ? 1u : 0u));

    /* walk the block records */
    pos = 12;
    for (k = 0; k < nf; ++k) {
        uint32_t cs;
        if (clen - pos < 4)
            return fail(dc, SBZ_ERR_CORRUPT, "chunk: block size record %u past the end", k);
        cs = sbz_be32(c + pos);
        if (cs == 0)
            return fail(dc, SBZ_ERR_CORRUPT, "chunk: block %u has size 0", k);
        if (clen - pos - 4 < cs)
            return fail(dc, SBZ_ERR_CORRUPT, "chunk: block %u frame (%u bytes) past the end", k, cs);
        pos += 4u + cs;
    }

    tail = (S % 8u) * elemSize;
    if (clen - pos < tail)
        return fail(dc, SBZ_ERR_CORRUPT, "chunk: raw tail (%llu bytes) past the end",
                    (unsigned long long)tail);
    frameBytes = (nfull * B + last2) * elemSize;

    ci->inOffs = 0;      /* set by the caller (running over prior chunks) */
    ci->outOffs = 0;     /* set by the caller */
    ci->frameBytes = frameBytes;
    ci->tailBytes = tail;
    ci->blockBytes = B * elemSize;
    ci->tailOff = base + pos;
    ci->nfull = (uint32_t)nfull;
    ci->lastElems = (uint32_t)last2;
    ci->blockElems = (uint32_t)B;
    return SBZ_OK;
}

/* Whether this chunk's full blocks take the fused LZ4 decode+unshuffle path
 * (Phase 3, step 2). The fused kernel only handles full blocks that fit the
 * 8 KB tile, with e in {1,2,4} and blockElems % 32 == 0 (the BitUnshuffleWarp
 * cases). The partial last block and every other case go unfused. */
static int bshuf_chunk_full_fused(sbz_dctx *dc, uint32_t elemSize, uint32_t blockElems)
{
    if (!dc->lz4_fused)
        return 0;
    if (elemSize != 1u && elemSize != 2u && elemSize != 4u)
        return 0;
    if (blockElems % 32u)
        return 0;
    if ((uint64_t)blockElems * elemSize > SBZ_LZ4_FUSED_MAX_BLOCK)
        return 0;
    return 1;
}

/* ---- batched unshuffle (Phase 3, step 3) ----
 * The unshuffle kernels take a block table (uint2 {inByteOffset, outByteOffset} per
 * block). All blocks of a call are collected into buckets keyed by
 * (kernel, elemSize, blockElems), then each kernel is launched ONCE per bucket instead
 * of once per chunk part: in practice one launch for the full blocks and one for the
 * partial last blocks, independent of the number of chunks. */

typedef struct sbz_unsh_ref { uint32_t inOff, outOff; } sbz_unsh_ref;
typedef struct sbz_unsh_bucket {
    int kernel;          /* SBZK_BitUnshuffleWarp or SBZK_BitUnshuffleSlow */
    uint32_t elemSize;
    uint32_t blockElems; /* every entry is a block of this many elements */
    uint32_t n;
    uint32_t cap;
    sbz_unsh_ref *refs;
} sbz_unsh_bucket;
/* The buckets of one call. Grown as needed: a call whose chunks have many different
 * partial-last-block sizes gets one bucket (one launch) per size, never an error.
 * `hint` is the last bucket used, so the common case (runs of blocks with the same
 * key) does not scan the list. */
typedef struct sbz_unsh_set {
    sbz_unsh_bucket *b;
    int n, cap, hint;
} sbz_unsh_set;

static void bshuf_set_free(sbz_unsh_set *set)
{
    int i;
    for (i = 0; i < set->n; ++i)
        free(set->b[i].refs);
    free(set->b);
    set->b = NULL;
    set->n = set->cap = set->hint = 0;
}

/* The kernel to unshuffle a block of `blockElems` elements of `elemSize`-byte elements:
 * the warp kernel handles e in {1,2,4} with blockElems % 32 == 0, else the slow one. */
static int bshuf_unshuffle_kernel(uint32_t elemSize, uint32_t blockElems)
{
    if ((elemSize == 1 || elemSize == 2 || elemSize == 4) && blockElems % 32u == 0)
        return SBZK_BitUnshuffleWarp;
    return SBZK_BitUnshuffleSlow;
}

/* Append one block {inOff, outOff} to the bucket keyed by (kernel, elemSize, blockElems). */
static int bshuf_bucket_add(sbz_dctx *dc, sbz_unsh_set *set, uint32_t elemSize,
                            uint32_t blockElems, uint32_t inOff, uint32_t outOff)
{
    const int kernel = bshuf_unshuffle_kernel(elemSize, blockElems);
    int i = set->hint;
    sbz_unsh_bucket *bk;
    if (!(i < set->n && set->b[i].kernel == kernel && set->b[i].elemSize == elemSize &&
          set->b[i].blockElems == blockElems)) {
        for (i = 0; i < set->n; ++i)
            if (set->b[i].kernel == kernel && set->b[i].elemSize == elemSize &&
                set->b[i].blockElems == blockElems)
                break;
    }
    if (i == set->n) {
        if (set->n == set->cap) {
            const int nc = set->cap ? set->cap * 2 : 8;
            sbz_unsh_bucket *nbk = (sbz_unsh_bucket *)realloc(set->b, (size_t)nc * sizeof(*nbk));
            if (!nbk)
                return fail(dc, SBZ_ERR_NOMEM, "unshuffle buckets");
            set->b = nbk;
            set->cap = nc;
        }
        bk = &set->b[set->n++];
        bk->kernel = kernel;
        bk->elemSize = elemSize;
        bk->blockElems = blockElems;
        bk->n = 0;
        bk->cap = 0;
        bk->refs = NULL;
    }
    set->hint = i;
    bk = &set->b[i];
    if (bk->n == bk->cap) {
        uint32_t nc = bk->cap ? bk->cap * 2u : 64u;
        sbz_unsh_ref *r = (sbz_unsh_ref *)realloc(bk->refs, (size_t)nc * sizeof(*r));
        if (!r)
            return fail(dc, SBZ_ERR_NOMEM, "unshuffle block table");
        bk->refs = r;
        bk->cap = nc;
    }
    bk->refs[bk->n].inOff = inOff;
    bk->refs[bk->n].outOff = outOff;
    ++bk->n;
    return SBZ_OK;
}

/* Collect one chunk's unshuffle parts into the buckets: its full blocks (unless the
 * fused kernel already handled them, `fullFused`) and its partial last block. */
static int bshuf_unshuffle_collect(sbz_dctx *dc, sbz_unsh_set *set,
                                   uint32_t blockElems, uint32_t nfull, uint32_t lastElems,
                                   uint32_t elemSize, uint32_t inOff, uint32_t outOff,
                                   uint32_t fullFused)
{
    uint32_t b;
    if (nfull > 0 && !fullFused) {
        const uint32_t step = (uint32_t)((uint64_t)blockElems * elemSize);
        for (b = 0; b < nfull; ++b)
            TRY(bshuf_bucket_add(dc, set, elemSize, blockElems, inOff + b * step,
                                 outOff + b * step));
    }
    if (lastElems > 0) {
        const uint32_t step = (uint32_t)((uint64_t)nfull * blockElems * elemSize);
        TRY(bshuf_bucket_add(dc, set, elemSize, lastElems, inOff + step, outOff + step));
    }
    return SBZ_OK;
}

/* Launch one unshuffle kernel per (non-empty) bucket; all blocks of the call go in the
 * one launch of each group. */
static int bshuf_unshuffle_launch(sbz_dctx *dc, const sbz_unsh_set *set)
{
    int i;
    uint32_t nmax = 0;
    const int bshuf = B_BshufOutput;
    const sbz_bind binds[] = { { "inUnCompressedFramesData", B_UnCompressedFramesData },
                              { "inoutBshufOutput", bshuf },
                              { "inBlockTable", B_GzUnshuffleRef } };
    /* the warp kernel also takes uint32 word views and a uint4 store view of the same
     * two buffers (the byte views are the only ones the slow kernel needs) */
    const sbz_bind wbinds[] = { { "inUnCompressedFramesData", B_UnCompressedFramesData },
                               { "inUnCompressedFramesData32", B_UnCompressedFramesData },
                               { "inoutBshufOutput", bshuf },
                               { "inoutBshufOutput32", bshuf },
                               { "inoutBshufOutput16", bshuf },
                               { "inBlockTable", B_GzUnshuffleRef } };
    /* size the table once for the largest bucket: growing it between launches would
     * free the buffer an earlier, still queued launch reads. Each bucket's upload is
     * stream-ordered after the previous bucket's launch. */
    for (i = 0; i < set->n; ++i)
        if (set->b[i].n > nmax)
            nmax = set->b[i].n;
    if (nmax == 0)
        return SBZ_OK;
    TRY(buf_size(dc, B_GzUnshuffleRef, nmax));
    for (i = 0; i < set->n; ++i) {
        const sbz_unsh_bucket *bk = &set->b[i];
        if (bk->n == 0)
            continue;
        BE_TRY(upload_staged(dc, dc->buf[B_GzUnshuffleRef].ptr, bk->refs,
                                   (size_t)bk->n * sizeof(*bk->refs)),
               "upload unshuffle table");
        if (bk->kernel == SBZK_BitUnshuffleWarp) {
            const uint32_t groups = bk->blockElems / 32u;
            const uint32_t nslices = (groups + 31u) / 32u;
            const sbz_scalar s[] = { { "elemSize", bk->elemSize }, { "blockElems", bk->blockElems },
                                    { "fullBlocks", bk->n }, { "tgOffset", 0u } };
            TRY(launch(dc, SBZK_BitUnshuffleWarp, (bk->n * nslices + 7u) / 8u, wbinds,
                       NB(wbinds), s, NB(s)));
        } else {
            const sbz_scalar s[] = { { "elemSize", bk->elemSize }, { "blockElems", bk->blockElems },
                                    { "fullBlocks", bk->n }, { "tgOffset", 0u } };
            TRY(launch(dc, SBZK_BitUnshuffleSlow, TG_COUNT((uint64_t)bk->n * bk->blockElems, 256u),
                       binds, NB(binds), s, NB(s)));
        }
    }
    return SBZ_OK;
}

/* One LZ4 block's decode record, mirrored in kernels/Lz4DecodeBlocks.slang and
 * kernels/Lz4DecodeUnshuffle.slang (Lz4BlockRef). Element size 20 (5 x uint32). */
typedef struct sbz_bshuf_block {
    uint32_t inputOffs;   /* byte offset in CompressedData of the block's LZ4 data */
    uint32_t compSize;    /* compressed bytes (the [BE u32 c] record's value) */
    uint32_t decodedSize; /* bytes the block must decode to */
    uint32_t outOffs;     /* the block's output offset (see each kernel's space) */
    uint32_t chunkIndex;
} sbz_bshuf_block;

/* Decode the bitshuffle-LZ4 blocks of all chunks and unshuffle the full blocks
 * here if the fused path is eligible for a chunk's full blocks. Two block tables:
 * `fbuf` (the fused path: full blocks, decoded AND unshuffled in one kernel into
 * BshufOutput, so outOffs is in BshufOutput space) and `ubuf` (the unfused path:
 * every other block — non-eligible full blocks and every partial last block — decoded
 * by Lz4DecodeBlocks into UnCompressedFramesData, then unshuffled afterwards, so
 * outOffs is in UnCompressedFramesData space). Both launches can occur in one call. */
static int bshuf_decode_lz4(sbz_dctx *dc, const void *src, size_t srcSize,
                            const sbz_frame_span *chunks, uint32_t nchunks, uint32_t elemSize,
                            const sbz_bshuf_chunk *ci, uint64_t nblocksTot, uint64_t decoded,
                            uint64_t totalOut)
{
    sbz_bshuf_block *ubuf = NULL;
    sbz_bshuf_block *fbuf = NULL;
    const uint32_t nblocks = (uint32_t)nblocksTot;
    uint32_t nu = 0, nf = 0, i;
    int rc = SBZ_OK;

    if (nblocks == 0)
        return SBZ_OK;

    if (dc->geom) {
        /* P4 step 2c: sizes from the geometry; the BshufWalkChunks kernel walks the
         * block records on the device and writes both tables and the raw tails */
        const sbz_bshuf_chunk *c0 = &ci[0];
        const int fuse = bshuf_chunk_full_fused(dc, elemSize, c0->blockElems);
        const uint32_t nblk = c0->nfull + (c0->lastElems ? 1u : 0u);
        uint32_t *spans = (uint32_t *)malloc((size_t)nchunks * 8u);
        (void)src;
        nf = fuse ? nchunks * c0->nfull : 0u;
        nu = fuse ? (c0->lastElems ? nchunks : 0u) : nchunks * nblk;
        if (!spans)
            return fail(dc, SBZ_ERR_NOMEM, "chunk spans");
        for (i = 0; i < nchunks; ++i) {
            spans[2 * i] = (uint32_t)chunks[i].offs;
            spans[2 * i + 1] = (uint32_t)chunks[i].size;
        }
        rc = buf_size(dc, B_ChunkSpans, nchunks);
        if (rc == SBZ_OK)
            rc = upload_staged(dc, dc->buf[B_ChunkSpans].ptr, spans, (size_t)nchunks * 8u);
        free(spans);
        if (rc != SBZ_OK)
            return be_fail(dc, rc, "upload chunk spans");
        TRY(buf_size(dc, B_UnCompressedFramesData, decoded));
        TRY(buf_size(dc, B_BshufOutput, totalOut));
        TRY(buf_size(dc, B_GzLz4BlockRef, nu ? nu : 1u));
        TRY(buf_size(dc, B_GzLz4BlockRefFused, nf ? nf : 1u));
        TRY(buf_size(dc, B_GzLz4Status, nchunks));
        TRY(buf_fill(dc, B_GzLz4Status, 0));
        {
            const sbz_scalar s[] = {
                { "chunkCount", nchunks },
                { "totalBytes", (uint32_t)dc->geom->chunkBytes },
                { "blockBytes", dc->geom->blockBytes },
                { "nfull", c0->nfull },
                { "lastBytes", c0->lastElems * elemSize },
                { "tailBytes", (uint32_t)c0->tailBytes },
                { "fused", fuse ? 1u : 0u },
                { "chunkOutBytes", (uint32_t)(c0->frameBytes + c0->tailBytes) },
                { "frameBytes", (uint32_t)c0->frameBytes },
                { "tgOffset", 0u } };
            const sbz_bind b[] = { { "inoutBlockTable", B_GzLz4BlockRef },
                                  { "inoutBlockTableFused", B_GzLz4BlockRefFused },
                                  { "inoutStatus", B_GzLz4Status } };
            TRY(launch(dc, SBZK_BshufWalkChunks, TG_COUNT(nchunks, 32u), b, NB(b), s, NB(s)));
        }
        goto decode;
    }

    ubuf = malloc((size_t)nblocks * sizeof(*ubuf));
    fbuf = malloc((size_t)nblocks * sizeof(*fbuf));
    if (!ubuf || !fbuf) {
        rc = fail(dc, SBZ_ERR_NOMEM, "LZ4 block table");
        goto out;
    }
    /* build the two tables from the walk */
    for (i = 0; i < nchunks; ++i) {
        const uint8_t *c = (const uint8_t *)src + chunks[i].offs;
        const uint64_t B = ci[i].blockElems;
        const uint32_t nfull = ci[i].nfull;
        const uint64_t last2 = ci[i].lastElems;
        const int fuse = bshuf_chunk_full_fused(dc, elemSize, (uint32_t)B);
        uint64_t pos = 12, ios = ci[i].inOffs, bs = 0;
        uint32_t nfblk;
        for (nfblk = 0; nfblk < nfull + (last2 ? 1u : 0u); ++nfblk) {
            const int isFull = nfblk < nfull;
            const uint32_t cs = sbz_be32(c + pos);
            const uint32_t dsz = (uint32_t)((isFull ? B : last2) * elemSize);
            if (fuse && isFull) {
                fbuf[nf].inputOffs = (uint32_t)(chunks[i].offs + pos + 4);
                fbuf[nf].compSize = cs;
                fbuf[nf].decodedSize = dsz;
                fbuf[nf].outOffs = (uint32_t)(ci[i].outOffs + bs);
                fbuf[nf].chunkIndex = i;
                ++nf;
                bs += dsz;
            } else {
                ubuf[nu].inputOffs = (uint32_t)(chunks[i].offs + pos + 4);
                ubuf[nu].compSize = cs;
                ubuf[nu].decodedSize = dsz;
                ubuf[nu].outOffs = (uint32_t)ios;
                ubuf[nu].chunkIndex = i;
                ++nu;
            }
            ios += dsz;
            pos += 4u + cs;
        }
    }

    TRY(buf_size(dc, B_UnCompressedFramesData, decoded));
    TRY(buf_size(dc, B_BshufOutput, totalOut));
    TRY(buf_size(dc, B_GzLz4BlockRef, nu));
    TRY(buf_size(dc, B_GzLz4BlockRefFused, nf));
    TRY(buf_size(dc, B_GzLz4Status, nchunks));
    /* the input is already queued for upload (bshuf_common, before the walk) */
    (void)srcSize;
    TRY(buf_fill(dc, B_GzLz4Status, 0));

decode:
    if (nu > 0) {
        if (!dc->geom)
            BE_TRY(upload_staged(dc, dc->buf[B_GzLz4BlockRef].ptr, ubuf,
                                 (size_t)nu * sizeof(*ubuf)),
                   "upload LZ4 block table");
        TRY(launch(dc, SBZK_Lz4DecodeBlocks, nu, (const sbz_bind[]){
            { "inCompressedData", B_CompressedData },
            { "inBlockTable", B_GzLz4BlockRef },
            { "inoutDecoded", B_UnCompressedFramesData },
            { "inoutStatus", B_GzLz4Status },
        }, 4, (const sbz_scalar[]){ { "blockCount", nu }, { "tgOffset", 0u } }, 2));
    }
    if (nf > 0) {
        if (!dc->geom)
            BE_TRY(upload_staged(dc, dc->buf[B_GzLz4BlockRefFused].ptr, fbuf,
                                 (size_t)nf * sizeof(*fbuf)),
                   "upload LZ4 fused block table");
        TRY(launch(dc, SBZK_Lz4DecodeUnshuffle, nf, (const sbz_bind[]){
            { "inCompressedData", B_CompressedData },
            { "inBlockTable", B_GzLz4BlockRefFused },
            { "inoutStatus", B_GzLz4Status },
            { "inoutBshufOutput", B_BshufOutput },
            { "inoutBshufOutput32", B_BshufOutput },
            { "inoutBshufOutput16", B_BshufOutput },
        }, 6, (const sbz_scalar[]){ { "blockCount", nf }, { "elemSize", elemSize },
                                   { "tgOffset", 0u } }, 3));
    }

    /* The per-chunk status (nonzero = a corrupt LZ4 block, fused or unfused) stays
     * on the device; sbz_dctx_wait reads it once the queued work is done. */
    dc->pending_nchunks = nchunks;
    dc->lz4_fused_blocks = nf;
    dc->lz4_unfused_blocks = nu;
out:
    free(ubuf);
    free(fbuf);
    return rc;
}
/* zstd: each block is one zstd frame, decoded by the stage-0/1/2 chain
 * (decompress_frames_internal), which also sizes UnCompressedFramesData.
 * Both produce the same layout (decoded bit-transposed blocks, back to back in
 * chunk order, chunk i at ci[i].inOffs) for the common unshuffle. `nframesTot` is
 * the frame count. */
static int bshuf_decode_zstd(sbz_dctx *dc, const void *src, size_t srcSize,
                             const sbz_frame_span *chunks, uint32_t nchunks, uint32_t elemSize,
                             const sbz_bshuf_chunk *ci, uint64_t nframesTot)
{
    sbz_frame_span *spans = NULL;
    uint32_t *expected = NULL;
    uint32_t fi = 0, i;
    int rc = SBZ_OK;

    if (nframesTot > 0) {
        spans = malloc((size_t)nframesTot * sizeof(*spans));
        expected = malloc((size_t)nframesTot * sizeof(*expected));
        if (!spans || !expected) {
            rc = fail(dc, SBZ_ERR_NOMEM, "frame list");
            goto out;
        }
        for (i = 0; i < nchunks; ++i) {
            const uint8_t *c = (const uint8_t *)src + chunks[i].offs;
            const uint64_t B = ci[i].blockElems;
            const uint32_t nfull = ci[i].nfull;
            const uint64_t last2 = ci[i].lastElems;
            uint64_t pos = 12;
            uint32_t nf;
            for (nf = 0; nf < nfull + (last2 ? 1u : 0u); ++nf) {
                uint32_t cs = sbz_be32(c + pos);
                spans[fi].offs = chunks[i].offs + pos + 4;
                spans[fi].size = cs;
                expected[fi] = (uint32_t)((nf < nfull ? B : last2) * elemSize);
                ++fi;
                pos += 4u + cs;
            }
        }
        rc = decompress_frames_internal(dc, src, srcSize, spans, (uint32_t)nframesTot);
        if (rc)
            goto out;
        for (fi = 0; fi < nframesTot; ++fi)
            if (dc->frame_infos[fi].uncompSize != expected[fi]) {
                rc = fail(dc, SBZ_ERR_CORRUPT,
                          "frame %u decoded %llu bytes, expected %u (block elems x e)", fi,
                          (unsigned long long)dc->frame_infos[fi].uncompSize, expected[fi]);
                goto out;
            }
    }
out:
    free(spans);
    free(expected);
    return rc;
}

/* Shared bitshuffle pipeline: walk every chunk (header, block records, raw tail),
 * run the decode stage (zstd frames or LZ4 blocks), then size the output, unshuffle
 * every decoded block and upload each chunk's raw tail. Only the decode stage
 * differs between zstd and LZ4. */
static int bshuf_common(sbz_dctx *dc, const void *src, size_t srcSize,
                        const sbz_frame_span *chunks, uint32_t nchunks, uint32_t elemSize,
                        int lz4)
{
    sbz_bshuf_chunk *ci = NULL;
    uint64_t totalOut = 0, decoded = 0, nblocksTot = 0;
    uint32_t i;
    int rc = SBZ_OK;
    if (!dc || (!src && !dc->geom) || !chunks || nchunks == 0)
        return SBZ_ERR_ARGS;
    if (elemSize == 0 || elemSize > 8)
        return fail(dc, SBZ_ERR_ARGS, "elemSize %u not in 1..8", elemSize);
    TRY(check_idle(dc));
    dc->err[0] = 0;
    dc->stage_done = -1;
    if (dc->geom && !lz4)
        return fail(dc, SBZ_ERR_ARGS, "device input: bitshuffle-zstd not supported yet");

    if (lz4 && !dc->geom) {
        /* Queue the input upload first: it needs only srcSize, so the copy runs
         * while the host walks the chunks and builds the block tables below
         * (about 1 ms for a 16-frame CeO2 call), instead of after them.
         * CompressedData is a padded buffer: its size is rounded up to a
         * multiple of 4 and src is uploaded up to srcSize. If the walk then
         * fails, the upload was wasted but harmless. */
        if (srcSize + 3u > 0xFFFFFFFFull)
            return fail(dc, SBZ_ERR_LIMIT, "input %zu bytes > 4 GiB", srcSize);
        TRY(buf_size(dc, B_CompressedData, (uint32_t)((srcSize + 3u) & ~(size_t)3u) / 4u));
        BE_TRY(dc->be->ops->upload(dc->be, dc->buf[B_CompressedData].ptr, src, srcSize),
               "upload LZ4 input");
    }

    ci = malloc((size_t)nchunks * sizeof(*ci));
    if (!ci)
        return fail(dc, SBZ_ERR_NOMEM, "chunk list");

    /* pass 1: walk every chunk, count blocks, accumulate buffer offsets */
    for (i = 0; i < nchunks; ++i) {
        uint64_t nb;
        if (chunks[i].offs > srcSize || chunks[i].size > srcSize - chunks[i].offs) {
            rc = fail(dc, SBZ_ERR_ARGS, "chunk %u outside input of %zu bytes", i, srcSize);
            goto out;
        }
        if (dc->geom)
            rc = bshuf_geom_chunk(dc, dc->geom, &ci[i]);
        else
            rc = bshuf_walk_chunk(dc, (const uint8_t *)src + chunks[i].offs, chunks[i].size,
                                  chunks[i].offs, elemSize, &ci[i]);
        if (rc)
            goto out;
        ci[i].inOffs = decoded;
        ci[i].outOffs = totalOut;
        nb = (uint64_t)ci[i].nfull + (ci[i].lastElems ? 1u : 0u);
        nblocksTot += nb;
        decoded += ci[i].frameBytes;
        totalOut += ci[i].frameBytes + ci[i].tailBytes;
        if (totalOut > 0xFFFFFFFFull) {
            rc = fail(dc, SBZ_ERR_LIMIT, "total output %llu bytes > 4 GiB",
                      (unsigned long long)totalOut);
            goto out;
        }
    }

    /* staged uploads of this call: the LZ4 block tables (both together hold each
     * block once, 20 bytes) and the unshuffle tables (8 bytes per block), plus
     * alignment */
    stage_reserve(dc, (size_t)nblocksTot * (20u + 8u) + (size_t)nchunks * 8u + 64u * 16u);
    if (lz4)
        rc = bshuf_decode_lz4(dc, src, srcSize, chunks, nchunks, elemSize, ci, nblocksTot,
                              decoded, totalOut);
    else
        rc = bshuf_decode_zstd(dc, src, srcSize, chunks, nchunks, elemSize, ci, nblocksTot);
    if (rc)
        goto out;

    /* size the output (the fused LZ4 kernel already wrote its full blocks here),
     * collect every chunk's unfused blocks into per-(kernel, blockElems) buckets and
     * launch one unshuffle per bucket, then upload the raw tails */
    TRY(buf_size(dc, B_BshufOutput, totalOut));
    {
        sbz_unsh_set set = { NULL, 0, 0, 0 };
        for (i = 0; i < nchunks; ++i) {
            const uint32_t fused = (lz4 && bshuf_chunk_full_fused(dc, elemSize, ci[i].blockElems));
            rc = bshuf_unshuffle_collect(dc, &set, ci[i].blockElems, ci[i].nfull,
                                         ci[i].lastElems, elemSize, (uint32_t)ci[i].inOffs,
                                         (uint32_t)ci[i].outOffs, fused);
            if (rc) {
                bshuf_set_free(&set);
                goto out;
            }
            if (ci[i].tailBytes > 0 && !dc->geom) { /* geom: the walk kernel copies it */
                rc = dc->be->ops->upload(dc->be, dc->buf[B_BshufOutput].ptr + ci[i].outOffs +
                                                     ci[i].frameBytes,
                                         (const uint8_t *)src + ci[i].tailOff,
                                         (size_t)ci[i].tailBytes);
                if (rc != SBZ_OK) {
                    rc = be_fail(dc, rc, "BshufOutput tail");
                    bshuf_set_free(&set);
                    goto out;
                }
            }
        }
        rc = bshuf_unshuffle_launch(dc, &set);
        bshuf_set_free(&set);
        if (rc)
            goto out;
    }
    if (lz4 || dc->pending_zstd) {
        /* queued only: sbz_dctx_wait syncs and checks the per-chunk status (LZ4)
         * or the counters and frame status (zstd, not exact) */
        dc->pending = 1;
    } else {
        rc = dc->be->ops->sync(dc->be);
        if (rc != SBZ_OK) {
            rc = be_fail(dc, rc, "bshuf unshuffle");
            goto out;
        }
    }
    dc->stage_done = 2;

out:
    if (rc != SBZ_OK) { /* nothing queued is waited for after a failure */
        dc->pending = 0;
        dc->pending_zstd = 0;
    }
    free(ci);
    return rc;
}

int sbz_dctx_bshuf_decompress_async(sbz_dctx *dc, const void *src, size_t srcSize,
                                   const sbz_frame_span *chunks, uint32_t nchunks,
                                   uint32_t elemSize)
{
    if (!dc)
        return SBZ_ERR_ARGS;
    TRY(check_idle(dc));
    dc->rp_kind = 3;
    dc->rp_src = src;
    dc->rp_size = srcSize;
    dc->rp_spans = chunks;
    dc->rp_n = nchunks;
    dc->rp_elem = elemSize;
    return bshuf_common(dc, src, srcSize, chunks, nchunks, elemSize, 0);
}

int sbz_dctx_bshuf_decompress(sbz_dctx *dc, const void *src, size_t srcSize,
                             const sbz_frame_span *chunks, uint32_t nchunks, uint32_t elemSize)
{
    TRY(sbz_dctx_bshuf_decompress_async(dc, src, srcSize, chunks, nchunks, elemSize));
    return finish(dc);
}

int sbz_dctx_bslz4_decompress_async(sbz_dctx *dc, const void *src, size_t srcSize,
                                   const sbz_frame_span *chunks, uint32_t nchunks,
                                   uint32_t elemSize)
{
    return bshuf_common(dc, src, srcSize, chunks, nchunks, elemSize, 1);
}

static int replay_device(sbz_dctx *dc);

/* The slow path: redo the last call with the counters read back after each stage
 * (exact sizes and checks). After a capacity overflow it decodes and grows the
 * capacities; on corrupt input it returns the precise error. */
static int replay_exact(sbz_dctx *dc)
{
    int rc;
    dc->pending = 0;
    dc->pending_zstd = 0;
    dc->force_exact = 1;
    ++dc->n_replay;
    switch (dc->rp_kind) {
    case 1:
        rc = sbz_dctx_decompress(dc, dc->rp_src, dc->rp_size);
        break;
    case 2:
        rc = sbz_dctx_decompress_frames(dc, dc->rp_src, dc->rp_size, dc->rp_spans, dc->rp_n);
        break;
    case 3:
        rc = sbz_dctx_bshuf_decompress(dc, dc->rp_src, dc->rp_size, dc->rp_spans, dc->rp_n,
                                      dc->rp_elem);
        break;
    default:
        rc = fail(dc, SBZ_ERR_ARGS, "no call to replay");
    }
    dc->force_exact = 0;
    return rc;
}

/* A non-exact zstd call is done: read the counters and the frame status once.
 * Block counts that differ from the host walk, a nonzero frame status, or more
 * literal bytes or sequences than the buffers held: replay exactly. */
static int wait_zstd(sbz_dctx *dc)
{
    const uint32_t nf = dc->frame_count;
    uint32_t f;
    int rc, bad = 0;

    dc->pending_zstd = 0;
    rc = dc->be->ops->download(dc->be, &dc->counters, dc->buf[B_Counters].ptr,
                               sizeof(sbz_counters));
    if (rc == SBZ_OK)
        rc = dc->be->ops->download(dc->be, dc->frame_status, dc->buf[B_FrameStatus].ptr,
                                   (size_t)nf * 4u);
    if (rc == SBZ_OK)
        rc = dc->be->ops->sync(dc->be);
    if (rc != SBZ_OK)
        return be_fail(dc, rc, "readback Counters");
    if (dc->counters.Blocks_RAW != dc->host_count.rawBlockCount ||
        dc->counters.Blocks_RLE != dc->host_count.rleBlockCount ||
        dc->counters.Blocks_CMP != dc->host_count.cmpBlockCount)
        bad = 1;
    for (f = 0; f < nf && !bad; ++f)
        if (dc->frame_status[f] != 0)
            bad = 1;
    if (dc->counters.HUF_Streams_DecodedBytes > dc->cur_lit_cap ||
        dc->counters.Seq_Streams_DecodedItems > dc->cur_seq_cap)
        bad = 1;
    if (bad)
        return replay_exact(dc);
    return SBZ_OK;
}

int sbz_dctx_wait(sbz_dctx *dc)
{
    uint32_t *status;
    uint32_t i, n;
    int rc;

    if (!dc)
        return SBZ_ERR_ARGS;
    BE_TRY(dc->be->ops->sync(dc->be), "wait");
    if (!dc->pending)
        return SBZ_OK;
    dc->pending = 0;
    if (dc->pending_zstd)
        return wait_zstd(dc);
    n = dc->pending_nchunks;
    if (n == 0)
        return SBZ_OK;
    status = (uint32_t *)malloc((size_t)n * sizeof(*status));
    if (!status)
        return fail(dc, SBZ_ERR_NOMEM, "LZ4 status");
    rc = dc->be->ops->download(dc->be, status, dc->buf[B_GzLz4Status].ptr,
                               (size_t)n * sizeof(*status));
    if (rc == SBZ_OK)
        rc = dc->be->ops->sync(dc->be);
    if (rc != SBZ_OK) {
        free(status);
        return be_fail(dc, rc, "read LZ4 status");
    }
    for (i = 0; i < n; ++i)
        if (status[i]) {
            free(status);
            if (dc->rp_kind == 4) /* device input: find out on the host */
                return replay_device(dc);
            return fail(dc, SBZ_ERR_CORRUPT, "chunk %u has a corrupt LZ4 block", i);
        }
    free(status);
    return SBZ_OK;
}

int sbz_dctx_bslz4_decompress_device(sbz_dctx *dc, uint64_t dev_src, size_t srcSize,
                                    const sbz_frame_span *chunks, uint32_t nchunks,
                                    const sbz_bshuf_geom *geom)
{
    int rc;
    if (!dc || !dev_src || !geom || !chunks)
        return SBZ_ERR_ARGS;
    TRY(check_idle(dc));
    if (srcSize == 0 || srcSize + 3u > 0xFFFFFFFFull)
        return fail(dc, SBZ_ERR_LIMIT, "input %zu bytes", srcSize);
    dc->rp_kind = 4;
    dc->rp_dev = (sbz_devptr)dev_src;
    dc->rp_size = srcSize;
    dc->rp_spans = chunks;
    dc->rp_n = nchunks;
    dc->rp_geom = *geom;
    dc->geom = &dc->rp_geom;
    dc->ext_cd = (sbz_devptr)dev_src;
    dc->ext_cd_bytes = srcSize & ~(size_t)3u; /* whole words; reads stay inside */
    rc = bshuf_common(dc, NULL, srcSize, chunks, nchunks, geom->elemSize, 1);
    dc->geom = NULL; /* everything is queued: the bindings are taken */
    dc->ext_cd = 0;
    dc->ext_cd_bytes = 0;
    return rc;
}

/* The slow path of a device-input call: copy the input back to the host and
 * redo the call there, exactly (the host walk gives the precise error). */
static int replay_device(sbz_dctx *dc)
{
    uint8_t *host;
    int rc;
    ++dc->n_replay;
    host = (uint8_t *)malloc(dc->rp_size);
    if (!host)
        return fail(dc, SBZ_ERR_NOMEM, "host copy for the replay");
    rc = dc->be->ops->download(dc->be, host, dc->rp_dev, dc->rp_size);
    if (rc == SBZ_OK)
        rc = dc->be->ops->sync(dc->be);
    if (rc != SBZ_OK) {
        free(host);
        return be_fail(dc, rc, "copy the input back for the replay");
    }
    rc = sbz_dctx_bslz4_decompress(dc, host, dc->rp_size, dc->rp_spans, dc->rp_n,
                                  dc->rp_geom.elemSize);
    free(host);
    return rc;
}

int sbz_dctx_bslz4_decompress(sbz_dctx *dc, const void *src, size_t srcSize,
                             const sbz_frame_span *chunks, uint32_t nchunks, uint32_t elemSize)
{
    int rc = sbz_dctx_bslz4_decompress_async(dc, src, srcSize, chunks, nchunks, elemSize);
    if (rc != SBZ_OK)
        return rc;
    return sbz_dctx_wait(dc);
}

int sbz_dctx_counters(const sbz_dctx *dc, sbz_counters *out)
{
    if (!dc || !out || dc->stage_done < 0)
        return SBZ_ERR_ARGS;
    *out = dc->counters;
    return SBZ_OK;
}

int sbz_dctx_zstd_stats(const sbz_dctx *dc, uint32_t *exact_calls, uint32_t *replays,
                       uint32_t *lit_cap, uint32_t *seq_cap)
{
    if (!dc)
        return SBZ_ERR_ARGS;
    if (exact_calls)
        *exact_calls = dc->n_exact;
    if (replays)
        *replays = dc->n_replay;
    if (lit_cap)
        *lit_cap = dc->lit_cap;
    if (seq_cap)
        *seq_cap = dc->seq_cap;
    return SBZ_OK;
}

int sbz_dctx_lz4_blocks(const sbz_dctx *dc, uint32_t *fused, uint32_t *unfused)
{
    if (!dc)
        return SBZ_ERR_ARGS;
    if (fused)
        *fused = dc->lz4_fused_blocks;
    if (unfused)
        *unfused = dc->lz4_unfused_blocks;
    return SBZ_OK;
}

int sbz_dctx_read_buffer(sbz_dctx *dc, const char *name, void *dst, size_t *bytes_inout)
{
    int b;
    size_t n;
    if (!dc || !name || !bytes_inout)
        return SBZ_ERR_ARGS;
    b = buf_by_name(name);
    if (b < 0)
        return fail(dc, SBZ_ERR_ARGS, "no buffer named %s", name);
    n = dc->buf[b].bytes;
    if (!dst) {
        *bytes_inout = n;
        return SBZ_OK;
    }
    if (*bytes_inout < n)
        n = *bytes_inout;
    BE_TRY(dc->be->ops->download(dc->be, dst, dc->buf[b].ptr, n), name);
    BE_TRY(dc->be->ops->sync(dc->be), name);
    *bytes_inout = n;
    return SBZ_OK;
}
