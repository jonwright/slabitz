/* slabitz device-resident decode chain (P2): the stage logic of DS zstdgpu.cpp
 * ported to C on top of the sbz_backend interface.
 *
 * Status: Stage 0 + Stage 1 (frame/block parse on the GPU). Stage 2 (FSE,
 * Huffman, literals, sequences, execute) is the next slice.
 *
 * DS runs in "readback mode" here (zstdgpu_IsReadbackRequired == 1): CUDA has
 * no ExecuteIndirect, so after each stage the Counters struct is read back and
 * the host does what ZstdGpuUpdateDispatchArgs.hlsl does (grid sizes) and what
 * zstdgpu_ResourceInfo_Stage_N_InitSize does (buffer sizes).
 */
#ifndef SBZ_CHAIN_H
#define SBZ_CHAIN_H

#include <stddef.h>
#include <stdint.h>

#include "sbz_backend.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Mirrors zstdgpu_Counters (zstdgpu_structs.h), 22 x uint32. */
typedef struct sbz_counters {
    uint32_t FseHufW, FseLLen, FseOffs, FseMLen;
    uint32_t DecompressLiteralsGroups;
    uint32_t HUF_WgtStreams;
    uint32_t Seq_Streams_DecodedItems;
    uint32_t HUF_Streams_DecodedBytes;
    uint32_t Seq_Streams;
    uint32_t HUF_Streams;
    uint32_t HufLit;
    uint32_t RAW_Streams;
    uint32_t RLE_Streams;
    uint32_t Blocks_RAW, Blocks_RLE, Blocks_CMP;
    uint32_t BlocksBytes_RAW, BlocksBytes_RLE;
    uint32_t Frames;
    uint32_t Frames_UncompressedByteSize;
    uint32_t Frames_AcceptedUncompressedByteSize;
    uint32_t Frames_RequestTooLarge;
} sbz_counters;

typedef struct sbz_dctx sbz_dctx;

/* Creates a decode context on a backend. If own_backend, sbz_dctx_destroy also
 * destroys the backend. Buffers are pooled in the context and grow on demand;
 * a context is not thread-safe (use one per thread/stream). */
SBZ_API sbz_dctx *sbz_dctx_create(sbz_backend *be, int own_backend);

/* Convenience: CUDA backend on `device`, cubins under kernel_dir (see
 * sbz_cuda_create). Returns NULL on failure with a message in errbuf. */
SBZ_API sbz_dctx *sbz_dctx_create_cuda(int device, const char *kernel_dir, char *errbuf, size_t errlen);

SBZ_API void sbz_dctx_destroy(sbz_dctx *dc);

/* Last error message (empty string if none). */
SBZ_API const char *sbz_dctx_error(const sbz_dctx *dc);

/* Runs Stage 0 + Stage 1 on `src` (one or more concatenated zstd frames).
 * Returns SBZ_OK or a SBZ_ERR_* code. After it returns SBZ_OK, the counters and
 * every stage-0/1 buffer can be inspected with the accessors below. */
SBZ_API int sbz_dctx_parse(sbz_dctx *dc, const void *src, size_t srcSize);

/* Runs the full decode chain (Stage 0 + 1 + 2) on `src` and returns SBZ_OK once the
 * decompressed output is resident on the device in the "UnCompressedFramesData"
 * buffer. Read it back with sbz_dctx_read_buffer(dc, "UnCompressedFramesData", ...).
 * The total output byte count is in Counters.Frames_UncompressedByteSize (or the sum
 * of the per-frame uncompressed sizes). */
SBZ_API int sbz_dctx_decompress(sbz_dctx *dc, const void *src, size_t srcSize);

/* One zstd frame inside a caller's buffer: byte offset and size. */
typedef struct sbz_frame_span {
    uint64_t offs;
    uint64_t size;
} sbz_frame_span;

/* Like sbz_dctx_decompress, but src holds `nframes` zstd frames at the given byte
 * ranges, with arbitrary bytes between them (e.g. a bitshuffle-zstd chunk, where
 * each frame follows a [BE u32 size] record). The frames are decoded in list order
 * into UnCompressedFramesData, back to back: frame i starts at the sum of the
 * content sizes of frames 0..i-1. SBZ_ERR_CORRUPT if the frame headers do not walk
 * as exactly these frames. */
SBZ_API int sbz_dctx_decompress_frames(sbz_dctx *dc, const void *src, size_t srcSize,
                                     const sbz_frame_span *frames, uint32_t nframes);

/* P3: decodes nchunks bitshuffle-zstd chunks (HDF5 filter 32008) held in src
 * (chunk i = bytes [chunks[i].offs, +chunks[i].size)) with element size elemSize.
 * Output in the device buffer "BshufOutput": the chunks' uncompressed data back to
 * back, in order (chunk i starts at the sum of the header totals of chunks 0..i-1).
 * Each chunk is [BE u64 total][BE u32 block bytes] then per block [BE u32 c][c bytes
 * = one zstd frame] decoding to (elems x elemSize) bit-transposed bytes, then the
 * (S % 8) x elemSize raw, untransposed trailing bytes. SBZ_ERR_CORRUPT on a record
 * outside a chunk, a total not a multiple of elemSize, or a block not a multiple of
 * 8 elements; SBZ_ERR_CORRUPT if a frame's content size differs from its block's
 * elems x elemSize; SBZ_ERR_LIMIT if the total output exceeds 4 GiB. */
SBZ_API int sbz_dctx_bshuf_decompress(sbz_dctx *dc, const void *src, size_t srcSize,
                                    const sbz_frame_span *chunks, uint32_t nchunks,
                                    uint32_t elemSize);

/* Same as sbz_dctx_bshuf_decompress, but the bytes are bitshuffle-LZ4 (filter 32008,
 * compression 2): each block is one raw LZ4 block (no LZ4 frame header) instead of
 * one zstd frame. The output layout, unshuffle and raw tail are identical.
 * SBZ_ERR_CORRUPT on a corrupt LZ4 block (offset 0, offset past the decoded start, a
 * literal or match run past the block's end, a truncated LSIC run, or a block that
 * does not decode to elems x elemSize). */
SBZ_API int sbz_dctx_bslz4_decompress(sbz_dctx *dc, const void *src, size_t srcSize,
                                    const sbz_frame_span *chunks, uint32_t nchunks,
                                    uint32_t elemSize);

/* P4 step 2a: the same decode, queued without waiting for the GPU. Walks the
 * chunks on the host, queues the uploads, the decode and the unshuffle on the
 * context's stream, and returns. Host-side errors (bad chunk headers, limits) are
 * returned at once; a corrupt LZ4 block is reported by sbz_dctx_wait. Until
 * sbz_dctx_wait returns, `src` must stay valid when it is pinned host memory (a
 * pageable src is copied before this call returns), and the context must not
 * decode again (SBZ_ERR_ARGS): its output (BshufOutput) and status are in use.
 * For overlap, alternate two contexts on the same CUDA context (one stream each). */
SBZ_API int sbz_dctx_bslz4_decompress_async(sbz_dctx *dc, const void *src, size_t srcSize,
                                          const sbz_frame_span *chunks, uint32_t nchunks,
                                          uint32_t elemSize);

/* P4 step 2b: bitshuffle-zstd, queued as sbz_dctx_bslz4_decompress_async is. On
 * the context's first call (and with SLABITZ_CHECK_SYNC=1) it runs exactly, with
 * the counters read back after each stage; later calls size the literal and
 * sequence buffers from capacities learned before and are checked once by
 * sbz_dctx_wait, which redoes the call exactly if they overflowed or the input
 * is corrupt. So `src` and `chunks` must stay valid until sbz_dctx_wait returns. */
SBZ_API int sbz_dctx_bshuf_decompress_async(sbz_dctx *dc, const void *src, size_t srcSize,
                                          const sbz_frame_span *chunks, uint32_t nchunks,
                                          uint32_t elemSize);

/* zstd calls of this context so far that ran exactly (first call, replays,
 * SLABITZ_CHECK_SYNC) and that were replayed, and the current literal-byte and
 * sequence capacities (debug/test/monitoring helper). Any argument may be NULL. */
SBZ_API int sbz_dctx_zstd_stats(const sbz_dctx *dc, uint32_t *exact_calls, uint32_t *replays,
                              uint32_t *lit_cap, uint32_t *seq_cap);

/* P4 step 2c: the geometry of a stream of bitshuffle chunks, as the dataset
 * states it (all chunks alike, e.g. one detector frame per chunk): element size,
 * uncompressed bytes per chunk, and the bitshuffle block size in bytes (the
 * filter's block size x element size; 8192 when the filter says 0). */
typedef struct sbz_bshuf_geom {
    uint32_t elemSize;
    uint64_t chunkBytes;
    uint32_t blockBytes;
} sbz_bshuf_geom;

/* Bitshuffle-LZ4 from device memory, with no host pass over the compressed
 * bytes: `dev_src` (a CUdeviceptr in this context) holds srcSize bytes, and
 * `chunks` (host array) gives each chunk's offset and size in it. Sizes, offsets
 * and the unshuffle tables come from `geom`; a kernel walks each chunk's block
 * records on the device, checks its header against `geom`, and copies its raw
 * tail. Queued like sbz_dctx_bslz4_decompress_async; sbz_dctx_wait checks the
 * per-chunk status and, if a chunk is corrupt or does not match `geom`, copies
 * the input back to the host and redoes the call there (slow path: the precise
 * error, or a correct decode of a chunk with another geometry). `dev_src` and
 * `chunks` must stay valid until sbz_dctx_wait returns. */
SBZ_API int sbz_dctx_bslz4_decompress_device(sbz_dctx *dc, uint64_t dev_src, size_t srcSize,
                                           const sbz_frame_span *chunks, uint32_t nchunks,
                                           const sbz_bshuf_geom *geom);

/* Waits for the context's queued work. After an async bslz4 call, reads the
 * per-chunk status: SBZ_ERR_CORRUPT ("chunk N has a corrupt LZ4 block") if any
 * chunk failed. SBZ_OK with nothing queued. */
SBZ_API int sbz_dctx_wait(sbz_dctx *dc);

/* Counters as read back after the last completed stage. */
SBZ_API int sbz_dctx_counters(const sbz_dctx *dc, sbz_counters *out);

/* Last bslz4 call's kernel split (debug/test helper): how many LZ4 blocks were
 * decoded by the fused Lz4DecodeUnshuffle kernel (fused) and how many by
 * Lz4DecodeBlocks + the separate unshuffle (unfused). */
SBZ_API int sbz_dctx_lz4_blocks(const sbz_dctx *dc, uint32_t *fused, uint32_t *unfused);

/* Debug hook: copies device buffer `name` (DS resource name, e.g.
 * "BlocksCMPRefs") into dst. *bytes_inout: capacity in, bytes written out.
 * With dst == NULL only the size is returned. Synchronous. */
SBZ_API int sbz_dctx_read_buffer(sbz_dctx *dc, const char *name, void *dst, size_t *bytes_inout);

#ifdef __cplusplus
}
#endif

#endif /* SBZ_CHAIN_H */
