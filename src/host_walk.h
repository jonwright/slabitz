/* Host-side, pure-CPU walk of a zstd frame list (P2 Stage 0).
 *
 * These mirror the DS zstdgpu_CountFramesAndBlocks / CollectFrames walk on the
 * host to (a) count frames/blocks so the caller can size buffers and (b) extract
 * the per-frame byte offsets/sizes. They feed buffer sizing and the eventual GPU
 * dispatch; no GPU or D3D12 machinery is involved. The walk is fully bounded: any
 * input returns SBZ_OK, SBZ_ERR_CORRUPT or SBZ_ERR_LIMIT, and it never reads outside
 * the input buffer or past contentSizeInBytes.
 */
#ifndef GPUS_ZSTD_HOST_WALK_H
#define GPUS_ZSTD_HOST_WALK_H

#include <stdint.h>

#include "sbz_backend.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sbz_OffsetAndSize {
    uint32_t offs;
    uint32_t size;
} sbz_OffsetAndSize;

/* Mirrors zstdgpu_FrameInfo: window/uncompressed sizes plus the running prefix
 * block-count/byte-count starts for each block type. */
typedef struct sbz_FrameInfo {
    uint64_t windowSize;
    uint64_t uncompSize;
    uint32_t dictionary;
    uint32_t rawBlockStart;
    uint32_t rleBlockStart;
    uint32_t cmpBlockStart;
    uint32_t rawBlockBytesStart;
    uint32_t rleBlockBytesStart;
    /* T10: the frame has no Frame_Content_Size. uncompSize is then an upper
     * bound: raw bytes + RLE bytes + Block_Maximum_Size per compressed block. */
    uint32_t noContentSize;
} sbz_FrameInfo;

typedef struct sbz_CountFramesAndBlocksInfo {
    uint32_t rawBlockCount;
    uint32_t rleBlockCount;
    uint32_t cmpBlockCount;
    uint32_t frameCount;
    uint64_t frameByteCount;
} sbz_CountFramesAndBlocksInfo;

/* Error detail for the walk functions. `code` is SBZ_OK on success, else
 * SBZ_ERR_CORRUPT or SBZ_ERR_LIMIT; `msg` is a human-readable reason naming the
 * frame index and byte offset where applicable. Pass NULL to discard. */
typedef struct sbz_walk_error {
    int code;
    char msg[192];
} sbz_walk_error;

/* memoryBlock: 4-byte-aligned, memoryBlockSizeInBytes a multiple of 4, actual
 * compressed content is contentSizeInBytes bytes (rest is zero padding), plus 4
 * more readable zero bytes for the forward bit-buffer. Counts the frames and the
 * raw/RLE/compressed blocks. Returns SBZ_OK, SBZ_ERR_CORRUPT or SBZ_ERR_LIMIT. */
int sbz_CountFramesAndBlocks(sbz_CountFramesAndBlocksInfo *outInfo,
                            const void *memoryBlock,
                            uint32_t memoryBlockSizeInBytes,
                            uint32_t contentSizeInBytes,
                            sbz_walk_error *err);

/* Collects the per-frame byte offsets/sizes and headers for `frameCount` frames.
 * Never writes more than frameCount entries, and returns SBZ_ERR_CORRUPT if the
 * walk does not find exactly frameCount frames. */
int sbz_CollectFrames(sbz_OffsetAndSize *outFrames, sbz_FrameInfo *outFrameInfos,
                     uint32_t frameCount, const void *memoryBlock,
                     uint32_t memoryBlockSizeInBytes, uint32_t contentSizeInBytes,
                     sbz_walk_error *err);

#ifdef __cplusplus
}
#endif

#endif /* GPUS_ZSTD_HOST_WALK_H */
