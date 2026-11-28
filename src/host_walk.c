/* Bounded host-side walk of a zstd frame list (P2 Stage 0).
 *
 * Pure-CPU analogues of the DS zstdgpu_CountFramesAndBlocks / CollectFrames
 * walks, driven by a small forward bit-buffer. They are used to size buffers and
 * to extract per-frame offsets before the GPU stage. Every read and every skip is
 * bounds-checked against contentSizeInBytes: a truncated or garbled frame returns
 * SBZ_ERR_CORRUPT, a request that would overflow a 32-bit DS running total returns
 * SBZ_ERR_LIMIT, and a well-formed list returns SBZ_OK. The walk never reads past
 * its input.
 */
#include "host_walk.h"

#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#define SBZ_MAGIC_ZSTD 0xFD2FB528u
#define SBZ_MAGIC_SKIP_START 0x184D2A50u
#define SBZ_MAGIC_SKIP_END 0x184D2A5Fu

/* Largest block size a decodable frame may hold (Block_Maximum_Size, RFC 8878):
 * min(Window_Size, 128 KiB). */
#define SBZ_BLOCK_MAX_SIZE (128u * 1024u)

/* The GPU packs match offsets in 29 bits (zstdgpu_structs.h,
 * kzstdgpu_SeqOffset_Encoded_BitBase), so a window of 2^29 bytes or more is not
 * decodable here (decision 2026-11-25: reject, do not widen the packing). */
#define SBZ_WINDOW_LOG_MAX 29u

/* The DS kernels keep counts and byte totals in 32 bits; anything that would
 * exceed 0xFFFFFFF0 (leaving headroom) is a SBZ_ERR_LIMIT. */
#define SBZ_32BIT_LIMIT 0xFFFFFFF0u

/* Internal return for "input exhausted cleanly" (distinct from SBZ_OK and the
 * negative SBZ_ERR_* codes). */
#define SBZ_WALK_DONE 1

typedef struct sbz_FwdBB {
    const uint8_t *buffer;
    uint64_t bitbuf;
    uint32_t offset;
    uint32_t bitcnt;
    uint32_t datasz;
    uint32_t bytesz;
} sbz_FwdBB;

static uint32_t sbz_BitFieldExtractU32(uint32_t x, uint32_t start, uint32_t count) {
    const uint32_t mask = ~(~0u << count);
    return (x >> start) & mask;
}

static uint32_t sbz_MinU32(uint32_t a, uint32_t b) { return a < b ? a : b; }

/* Word `i` of the buffer, assembled from bytes (any alignment); bytes at or past
 * bytesz read as 0, so the caller's buffer needs no padding and no copy. */
static uint32_t sbz_FFBB_Word(const sbz_FwdBB *b, uint32_t i) {
    const uint64_t at = (uint64_t)i * 4u;
    uint32_t w = 0, k;
    if (at + 4u <= b->bytesz) {
        memcpy(&w, b->buffer + at, 4); /* little-endian hosts (as the DS walk) */
        return w;
    }
    for (k = 0; k < 4u && at + k < b->bytesz; ++k)
        w |= (uint32_t)b->buffer[at + k] << (8u * k);
    return w;
}

static void sbz_FFBB_Init(sbz_FwdBB *b, const uint8_t *buffer, uint32_t datasz, uint32_t bytesz) {
    b->buffer = buffer;
    b->bitbuf = 0;
    b->offset = 0;
    b->bitcnt = 0;
    b->datasz = datasz;
    b->bytesz = bytesz;
}

static uint32_t sbz_FFBB_GetByteOffset(const sbz_FwdBB *b) {
    return (b->offset << 2) - (b->bitcnt >> 3);
}

/* Bytes of real content left from the current logical position. Never underflows:
 * if the bit-buffer has somehow advanced past the content, reports 0. */
static uint32_t sbz_FFBB_Available(const sbz_FwdBB *b) {
    const uint32_t pos = sbz_FFBB_GetByteOffset(b);
    return pos < b->datasz ? b->datasz - pos : 0;
}

/* Byte position (from the start of the content) of the next unread byte. */
static uint32_t sbz_FFBB_Pos(const sbz_FwdBB *b) { return sbz_FFBB_GetByteOffset(b); }

static void sbz_FFBB_Refill(sbz_FwdBB *b, uint32_t bitcnt) {
    if (b->bitcnt < bitcnt) {
        b->bitbuf |= (uint64_t)sbz_FFBB_Word(b, b->offset) << b->bitcnt;
        b->bitcnt += 32;
        b->offset += 1;
    }
}

static uint32_t sbz_FFBB_Top(const sbz_FwdBB *b, uint32_t bitcnt) {
    return (uint32_t)(b->bitbuf & ~(~0ull << bitcnt));
}

static void sbz_FFBB_Pop(sbz_FwdBB *b, uint32_t bitcnt) {
    b->bitbuf >>= bitcnt;
    b->bitcnt -= bitcnt;
}

static uint32_t sbz_FFBB_Get(sbz_FwdBB *b, uint32_t bitcnt) {
    sbz_FFBB_Refill(b, bitcnt);
    const uint32_t bits = sbz_FFBB_Top(b, bitcnt);
    sbz_FFBB_Pop(b, bitcnt);
    return bits;
}

static uint32_t sbz_FFBB_GetNoRefill(sbz_FwdBB *b, uint32_t bitcnt) {
    const uint32_t bits = sbz_FFBB_Top(b, bitcnt);
    sbz_FFBB_Pop(b, bitcnt);
    return bits;
}

static void sbz_FFBB_Skip(sbz_FwdBB *b, uint32_t bytecnt) {
    const uint32_t leftcnt = b->bitcnt >> 3;
    const uint32_t skipcnt = sbz_MinU32(leftcnt, bytecnt);
    sbz_FFBB_Pop(b, skipcnt << 3u);
    if (bytecnt > skipcnt) {
        bytecnt -= skipcnt;
        b->offset += bytecnt >> 2;
        b->bitcnt = 0;
        const uint32_t reqbits = (bytecnt & 3u) << 3u;
        if (reqbits > 0) {
            b->bitbuf = (uint64_t)sbz_FFBB_Word(b, b->offset) >> reqbits;
            b->offset += 1;
            b->bitcnt = 32 - reqbits;
        }
    }
}

static int sbz_walk_err(sbz_walk_error *err, int code, const char *fmt, ...) {
    if (err) {
        va_list ap;
        err->code = code;
        va_start(ap, fmt);
        vsnprintf(err->msg, sizeof(err->msg), fmt, ap);
        va_end(ap);
    }
    return code;
}

/* Parses one zstd frame (after its 4-byte magic has been consumed). Fills
 * outFrameInfo and advances `bits`. Returns SBZ_OK, SBZ_ERR_CORRUPT or
 * SBZ_ERR_LIMIT; on failure `err` names the frame index and byte offset.
 * outBlockInfo/outBlocks* are gone: a compressed block is skipped by its size
 * (the GPU does the literal/sequence work). */
static int sbz_ParseFrame(uint32_t frameIndex, sbz_FrameInfo *outFrameInfo, sbz_FwdBB *bits,
                         sbz_walk_error *err) {
    uint32_t pos, descriptor, singleSegmentFlag, dictionaryIdFlag, frameContentSizeFlag;
    uint32_t windowDescriptor = 0, contentCheckSumFlag;
    uint32_t lastBlock, blockType, blockSize, blockBase, byteCount, blockMax;
    uint64_t windowBase, windowAdd;
    uint64_t rawBytesTotal = 0, rleBytesTotal = 0;
    static const uint8_t dictBytes[4] = { 0u, 1u, 2u, 4u };
    static const uint8_t fcsBytes[4] = { 1u, 2u, 4u, 8u };

    memset(outFrameInfo, 0, sizeof(*outFrameInfo));

    pos = sbz_FFBB_Pos(bits);
    if (sbz_FFBB_Available(bits) < 1)
        return sbz_walk_err(err, SBZ_ERR_CORRUPT,
                           "frame %u @ byte %u: frame header runs past content", frameIndex, pos);
    descriptor = sbz_FFBB_Get(bits, 8);
    if (descriptor & 0x08u)
        return sbz_walk_err(err, SBZ_ERR_CORRUPT,
                           "frame %u @ byte %u: reserved descriptor bit set", frameIndex, pos);

    singleSegmentFlag = sbz_BitFieldExtractU32(descriptor, 5, 1);
    if (0 == singleSegmentFlag) {
        pos = sbz_FFBB_Pos(bits);
        if (sbz_FFBB_Available(bits) < 1)
            return sbz_walk_err(err, SBZ_ERR_CORRUPT,
                               "frame %u @ byte %u: window descriptor runs past content",
                               frameIndex, pos);
        windowDescriptor = sbz_FFBB_Get(bits, 8);
        windowBase = 1ull << (10 + (windowDescriptor >> 3));
        windowAdd = (windowBase / 8) * (windowDescriptor & 7);
        outFrameInfo->windowSize = windowBase + windowAdd;
    }

    dictionaryIdFlag = sbz_BitFieldExtractU32(descriptor, 0, 2);
    if (0 != dictionaryIdFlag) {
        byteCount = dictBytes[dictionaryIdFlag];
        pos = sbz_FFBB_Pos(bits);
        if (sbz_FFBB_Available(bits) < byteCount)
            return sbz_walk_err(err, SBZ_ERR_CORRUPT,
                               "frame %u @ byte %u: dictionary id runs past content", frameIndex,
                               pos);
        outFrameInfo->dictionary = sbz_FFBB_Get(bits, byteCount * 8);
    }

    frameContentSizeFlag = sbz_BitFieldExtractU32(descriptor, 6, 2);
    if (0 != singleSegmentFlag || 0 != frameContentSizeFlag) {
        byteCount = fcsBytes[frameContentSizeFlag];
        pos = sbz_FFBB_Pos(bits);
        if (sbz_FFBB_Available(bits) < byteCount)
            return sbz_walk_err(err, SBZ_ERR_CORRUPT,
                               "frame %u @ byte %u: content size runs past content", frameIndex,
                               pos);
        if (8 == byteCount) {
            outFrameInfo->uncompSize = sbz_FFBB_Get(bits, 32);
            outFrameInfo->uncompSize |= (uint64_t)sbz_FFBB_Get(bits, 32) << 32;
        } else {
            outFrameInfo->uncompSize = sbz_FFBB_Get(bits, byteCount * 8);
            if (16u == byteCount * 8)
                outFrameInfo->uncompSize += 256;
        }
        if (outFrameInfo->uncompSize > SBZ_32BIT_LIMIT)
            return sbz_walk_err(err, SBZ_ERR_LIMIT,
                               "frame %u @ byte %u: content size %llu exceeds %u", frameIndex, pos,
                               (unsigned long long)outFrameInfo->uncompSize, SBZ_32BIT_LIMIT);
    }
    if (0 != singleSegmentFlag)
        outFrameInfo->windowSize = outFrameInfo->uncompSize;

    /* Frame-level rejects, in the GPU parser's order (zstdgpu_FrameStatusFromHeader):
     * valid zstd that this decoder does not handle. The GPU keeps its own checks. */
    if (0 != dictionaryIdFlag)
        return sbz_walk_err(err, SBZ_ERR_UNSUPPORTED, "frame %u: dictionary (id %u)", frameIndex,
                           outFrameInfo->dictionary);
    if (outFrameInfo->windowSize >= (1ull << SBZ_WINDOW_LOG_MAX)) {
        if (0 == singleSegmentFlag && 0 == (windowDescriptor & 7u))
            return sbz_walk_err(err, SBZ_ERR_UNSUPPORTED, "frame %u: window 2^%u exceeds 2^%u",
                               frameIndex, 10u + (windowDescriptor >> 3), SBZ_WINDOW_LOG_MAX);
        return sbz_walk_err(err, SBZ_ERR_UNSUPPORTED, "frame %u: window %llu bytes exceeds 2^%u",
                           frameIndex, (unsigned long long)outFrameInfo->windowSize,
                           SBZ_WINDOW_LOG_MAX);
    }
    /* T10: no Frame_Content_Size is accepted; uncompSize becomes an upper bound
     * after the block walk and the caller decides (sbz_chain.c host_prepare) */
    if (0 == singleSegmentFlag && 0 == frameContentSizeFlag)
        outFrameInfo->noContentSize = 1;
    blockMax = (uint32_t)(outFrameInfo->windowSize < SBZ_BLOCK_MAX_SIZE ? outFrameInfo->windowSize
                                                                        : SBZ_BLOCK_MAX_SIZE);

    do {
        pos = sbz_FFBB_Pos(bits);
        if (sbz_FFBB_Available(bits) < 3)
            return sbz_walk_err(err, SBZ_ERR_CORRUPT,
                               "frame %u @ byte %u: block header runs past content (frame has no "
                               "terminating last block)", frameIndex, pos);
        sbz_FFBB_Refill(bits, 1 + 2 + 21);
        lastBlock = sbz_FFBB_GetNoRefill(bits, 1);
        blockType = sbz_FFBB_GetNoRefill(bits, 2);
        blockSize = sbz_FFBB_GetNoRefill(bits, 21);
        blockBase = sbz_FFBB_Pos(bits);

        if (3 == blockType)
            return sbz_walk_err(err, SBZ_ERR_CORRUPT,
                               "frame %u @ byte %u: reserved block type 3", frameIndex,
                               blockBase - 3);
        /* Block_Size (compressed size, raw size, or an RLE block's regenerated
         * size) is at most Block_Maximum_Size = min(window, 128 KiB) */
        if (blockSize > blockMax)
            return sbz_walk_err(err, SBZ_ERR_CORRUPT,
                               "frame %u @ byte %u: block size %u exceeds the block maximum %u",
                               frameIndex, blockBase - 3, blockSize, blockMax);

        if (2 == blockType) {
            if (0 == blockSize)
                return sbz_walk_err(err, SBZ_ERR_CORRUPT,
                                   "frame %u @ byte %u: compressed block of size 0", frameIndex,
                                   blockBase);
            if (sbz_FFBB_Available(bits) < blockSize)
                return sbz_walk_err(err, SBZ_ERR_CORRUPT,
                                   "frame %u @ byte %u: compressed block (%u bytes) runs past "
                                   "content", frameIndex, blockBase, blockSize);
            sbz_FFBB_Skip(bits, blockSize);
            outFrameInfo->cmpBlockStart += 1;
        } else if (0 == blockType) {
            if (sbz_FFBB_Available(bits) < blockSize)
                return sbz_walk_err(err, SBZ_ERR_CORRUPT,
                                   "frame %u @ byte %u: raw block (%u bytes) runs past content",
                                   frameIndex, blockBase, blockSize);
            sbz_FFBB_Skip(bits, blockSize);
            outFrameInfo->rawBlockStart += 1;
            rawBytesTotal += blockSize;
        } else { /* 1 == blockType: RLE, one symbol byte */
            if (sbz_FFBB_Available(bits) < 1)
                return sbz_walk_err(err, SBZ_ERR_CORRUPT,
                                   "frame %u @ byte %u: RLE block symbol runs past content",
                                   frameIndex, blockBase);
            sbz_FFBB_Skip(bits, 1);
            outFrameInfo->rleBlockStart += 1;
            rleBytesTotal += blockSize;
        }
        /* the DS kernels keep raw/RLE byte totals in 32 bits; a frame whose decoded
         * byte total would exceed 0xFFFFFFF0 is a SBZ_ERR_LIMIT (64-bit tally so the
         * value cannot wrap below the limit mid-frame). */
        if (rawBytesTotal > SBZ_32BIT_LIMIT || rleBytesTotal > SBZ_32BIT_LIMIT)
            return sbz_walk_err(err, SBZ_ERR_LIMIT,
                               "frame %u @ byte %u: a block byte total exceeds %u", frameIndex,
                               blockBase, SBZ_32BIT_LIMIT);
    } while (0 == lastBlock);

    outFrameInfo->rawBlockBytesStart = (uint32_t)rawBytesTotal;
    outFrameInfo->rleBlockBytesStart = (uint32_t)rleBytesTotal;
    if (outFrameInfo->noContentSize)
        outFrameInfo->uncompSize = rawBytesTotal + rleBytesTotal +
                                   (uint64_t)outFrameInfo->cmpBlockStart * blockMax;

    contentCheckSumFlag = sbz_BitFieldExtractU32(descriptor, 2, 1);
    if (0 != contentCheckSumFlag) {
        pos = sbz_FFBB_Pos(bits);
        if (sbz_FFBB_Available(bits) < 4)
            return sbz_walk_err(err, SBZ_ERR_CORRUPT,
                               "frame %u @ byte %u: content checksum runs past content", frameIndex,
                               pos);
        sbz_FFBB_Get(bits, 32);
    }
    return SBZ_OK;
}

/* Reads the next 4-byte magic at the current position, consuming any skippable
 * frames first. Returns SBZ_OK with *magic set (the magic of a real frame, zstd or
 * not), SBZ_WALK_DONE if the input ends cleanly (0 bytes left) or right after a
 * skippable frame, or SBZ_ERR_CORRUPT on a bounds violation. */
static int sbz_WalkNextMagic(sbz_FwdBB *bits, uint32_t *magic, sbz_walk_error *err) {
    uint32_t pos, frameSize;
    for (;;) {
        pos = sbz_FFBB_Pos(bits);
        if (sbz_FFBB_Available(bits) == 0)
            return SBZ_WALK_DONE; /* clean end of the input */
        if (sbz_FFBB_Available(bits) < 4)
            return sbz_walk_err(err, SBZ_ERR_CORRUPT,
                               "byte %u: fewer than 4 bytes left where a frame should start", pos);
        *magic = sbz_FFBB_Get(bits, 32);
        while (*magic >= SBZ_MAGIC_SKIP_START && *magic <= SBZ_MAGIC_SKIP_END) {
            pos = sbz_FFBB_Pos(bits);
            if (sbz_FFBB_Available(bits) < 4)
                return sbz_walk_err(err, SBZ_ERR_CORRUPT,
                                   "byte %u: skippable frame size field runs past content", pos);
            frameSize = sbz_FFBB_Get(bits, 32);
            pos = sbz_FFBB_Pos(bits);
            if (sbz_FFBB_Available(bits) < frameSize)
                return sbz_walk_err(err, SBZ_ERR_CORRUPT,
                                   "byte %u: skippable frame (%u bytes) runs past content", pos,
                                   frameSize);
            sbz_FFBB_Skip(bits, frameSize);
            if (sbz_FFBB_Available(bits) == 0)
                return SBZ_WALK_DONE;
            *magic = sbz_FFBB_Get(bits, 32);
        }
        return SBZ_OK;
    }
}

int sbz_CountFramesAndBlocks(sbz_CountFramesAndBlocksInfo *outInfo, const void *memoryBlock,
                            uint32_t memoryBlockSizeInBytes, uint32_t contentSizeInBytes,
                            sbz_walk_error *err) {
    sbz_FwdBB bits;
    uint32_t magic, frameIndex = 0, pos;
    uint64_t frameByteCount = 0, rawBytes = 0, rleBytes = 0;
    uint64_t rawBlocks = 0, rleBlocks = 0, cmpBlocks = 0;

    outInfo->rawBlockCount = 0;
    outInfo->rleBlockCount = 0;
    outInfo->cmpBlockCount = 0;
    outInfo->frameCount = 0;
    outInfo->frameByteCount = 0;

    sbz_FFBB_Init(&bits, (const uint8_t *)memoryBlock, contentSizeInBytes, memoryBlockSizeInBytes);

    for (;;) {
        sbz_FrameInfo frameInfo;
        int rc = sbz_WalkNextMagic(&bits, &magic, err);
        if (rc == SBZ_WALK_DONE)
            break;
        if (rc != SBZ_OK)
            return rc;
        if (magic != SBZ_MAGIC_ZSTD) {
            pos = sbz_FFBB_Pos(&bits) - 4;
            return sbz_walk_err(err, SBZ_ERR_CORRUPT,
                               "byte %u: magic 0x%08X is neither zstd nor skippable", pos, magic);
        }
        memset(&frameInfo, 0, sizeof(frameInfo));
        rc = sbz_ParseFrame(frameIndex, &frameInfo, &bits, err);
        if (rc != SBZ_OK)
            return rc;
        rawBytes += frameInfo.rawBlockBytesStart;
        rleBytes += frameInfo.rleBlockBytesStart;
        rawBlocks += frameInfo.rawBlockStart;
        rleBlocks += frameInfo.rleBlockStart;
        cmpBlocks += frameInfo.cmpBlockStart;
        frameByteCount += frameInfo.uncompSize;
        frameIndex += 1u;
        if (rawBytes > SBZ_32BIT_LIMIT || rleBytes > SBZ_32BIT_LIMIT ||
            frameByteCount > SBZ_32BIT_LIMIT || rawBlocks > SBZ_32BIT_LIMIT ||
            rleBlocks > SBZ_32BIT_LIMIT || cmpBlocks > SBZ_32BIT_LIMIT ||
            frameIndex > SBZ_32BIT_LIMIT)
            return sbz_walk_err(err, SBZ_ERR_LIMIT,
                               "frame %u @ byte %u: a 32-bit running total exceeds %u", frameIndex,
                               sbz_FFBB_Pos(&bits), SBZ_32BIT_LIMIT);
    }

    outInfo->rawBlockCount = (uint32_t)rawBlocks;
    outInfo->rleBlockCount = (uint32_t)rleBlocks;
    outInfo->cmpBlockCount = (uint32_t)cmpBlocks;
    outInfo->frameCount = frameIndex;
    outInfo->frameByteCount = frameByteCount;
    return SBZ_OK;
}

int sbz_CollectFrames(sbz_OffsetAndSize *outFrames, sbz_FrameInfo *outFrameInfos,
                     uint32_t frameCount, const void *memoryBlock,
                     uint32_t memoryBlockSizeInBytes, uint32_t contentSizeInBytes,
                     sbz_walk_error *err) {
    sbz_FwdBB bits;
    sbz_FrameInfo frameInfo;
    uint32_t frameId = 0;

    memset(&frameInfo, 0, sizeof(frameInfo));
    sbz_FFBB_Init(&bits, (const uint8_t *)memoryBlock, contentSizeInBytes, memoryBlockSizeInBytes);

    for (;;) {
        uint32_t magic, sofs;
        int rc = sbz_WalkNextMagic(&bits, &magic, err);
        if (rc == SBZ_WALK_DONE)
            break;
        if (rc != SBZ_OK)
            return rc;
        if (magic != SBZ_MAGIC_ZSTD)
            break;
        if (frameId >= frameCount)
            return sbz_walk_err(err, SBZ_ERR_CORRUPT,
                               "byte %u: second walk found more than the counted %u frames",
                               sbz_FFBB_Pos(&bits) - 4, frameCount);
        sofs = sbz_FFBB_Pos(&bits) - 4;
        outFrames[frameId].offs = sofs;
        outFrameInfos[frameId].rawBlockStart = (uint32_t)frameInfo.rawBlockStart;
        outFrameInfos[frameId].rleBlockStart = (uint32_t)frameInfo.rleBlockStart;
        outFrameInfos[frameId].cmpBlockStart = (uint32_t)frameInfo.cmpBlockStart;
        outFrameInfos[frameId].rawBlockBytesStart = (uint32_t)frameInfo.rawBlockBytesStart;
        outFrameInfos[frameId].rleBlockBytesStart = (uint32_t)frameInfo.rleBlockBytesStart;

        frameInfo.windowSize = 0;
        frameInfo.uncompSize = 0;
        frameInfo.dictionary = 0;
        frameInfo.rawBlockStart = 0;
        frameInfo.rleBlockStart = 0;
        frameInfo.cmpBlockStart = 0;
        frameInfo.rawBlockBytesStart = 0;
        frameInfo.rleBlockBytesStart = 0;

        rc = sbz_ParseFrame(frameId, &frameInfo, &bits, err);
        if (rc != SBZ_OK)
            return rc;

        outFrameInfos[frameId].windowSize = frameInfo.windowSize;
        outFrameInfos[frameId].uncompSize = frameInfo.uncompSize;
        outFrameInfos[frameId].dictionary = frameInfo.dictionary;
        outFrameInfos[frameId].noContentSize = frameInfo.noContentSize;
        outFrames[frameId].size = sbz_FFBB_Pos(&bits) - outFrames[frameId].offs;

        frameInfo.rawBlockStart += outFrameInfos[frameId].rawBlockStart;
        frameInfo.rleBlockStart += outFrameInfos[frameId].rleBlockStart;
        frameInfo.cmpBlockStart += outFrameInfos[frameId].cmpBlockStart;
        frameInfo.rawBlockBytesStart += outFrameInfos[frameId].rawBlockBytesStart;
        frameInfo.rleBlockBytesStart += outFrameInfos[frameId].rleBlockBytesStart;
        if (frameInfo.rawBlockBytesStart > SBZ_32BIT_LIMIT ||
            frameInfo.rleBlockBytesStart > SBZ_32BIT_LIMIT ||
            frameInfo.rawBlockStart > SBZ_32BIT_LIMIT ||
            frameInfo.rleBlockStart > SBZ_32BIT_LIMIT ||
            frameInfo.cmpBlockStart > SBZ_32BIT_LIMIT)
            return sbz_walk_err(err, SBZ_ERR_LIMIT,
                               "frame %u @ byte %u: a 32-bit running total exceeds %u", frameId,
                               sbz_FFBB_Pos(&bits), SBZ_32BIT_LIMIT);
        frameId += 1u;
    }

    if (frameId != frameCount)
        return sbz_walk_err(err, SBZ_ERR_CORRUPT, "second walk found %u frames, expected %u",
                           frameId, frameCount);
    return SBZ_OK;
}
