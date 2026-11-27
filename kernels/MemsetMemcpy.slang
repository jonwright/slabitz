// Slang entry for zstdgpu_ShaderEntry_MemsetMemcpy (DirectStorage, MIT).
// Shaders/ZstdGpuMemsetMemcpy.hlsl, but the D3D12 RootSignature/ConstantBuffer SRT fill is
// replaced by a by-value parameter list (there is no SharedEntry in shaders.h -- the body is
// inline in the .hlsl).
//
// Mapping (changed from DS): ONE WORKGROUP PER BLOCK, grid.x = Blocks_RAW or Blocks_RLE.
// No per-byte binary search: blockIdx = group index, blockRef = inBlocksRefs[blockIdx],
// dst = inBlockDestOffs[inGlobalBlockIndex[blockIdx]], size = blockRef.size. The 256 threads
// stride over the block. flags bit0 = RAW (copy from CompressedData at byte offset
// blockRef.offs), else RLE (fill with the byte stored in blockRef.offs).
//
// Stores: the unaligned head (dst up to the next multiple of 4) and the tail (< 4 bytes) are
// single-byte stores to the u8 view; the aligned interior is written as 32-bit words through
// a second view (inoutUnCompressedFramesData32) of the SAME buffer, consecutive threads to
// consecutive words (coalesced). Neighbouring blocks may share an edge word, so a word is
// never read-modified-written; only whole interior words are stored as words.
// RAW source words that are not 4-aligned are assembled from two CompressedData words; the
// second word is only read when all 4 source bytes lie inside the block (so no overrun).
// Plain Slang, no wave ops, same code on every arch.
#include "slang_compat.h"

#include "zstdgpu_shaders.h"
#include "srt_headers/zstdgpu_srt_structs.h"

#define MM_THREADS 256u

[shader("compute")]
[numthreads(MM_THREADS, 1, 1)]
void main(StructuredBuffer<zstdgpu_Counters> inCounters,
          StructuredBuffer<uint32_t> inCompressedData,
          StructuredBuffer<uint32_t> inBlockDestOffs,
          StructuredBuffer<zstdgpu_OffsetAndSize> inBlocksRefs,
          StructuredBuffer<uint32_t> inGlobalBlockIndex,
          RWStructuredBuffer<uint8_t> inoutUnCompressedFramesData,
          RWStructuredBuffer<uint32_t> inoutUnCompressedFramesData32,
          uint32_t tgOffset, uint32_t flags,
          uint2 groupId2 : SV_GroupID, uint32_t t : SV_GroupThreadID)
{
    const uint32_t blockIdx = zstdgpu_ConvertTo32BitGroupId(groupId2, tgOffset);
    const bool isRaw = (flags & 0x1u) != 0u;
    const uint32_t blockCnt = isRaw ? inCounters[0].Blocks_RAW : inCounters[0].Blocks_RLE;
    if (blockIdx >= blockCnt)
    {
        return;
    }

    const zstdgpu_OffsetAndSize blockRef = inBlocksRefs[blockIdx];
    const uint32_t dst = inBlockDestOffs[inGlobalBlockIndex[blockIdx]];
    if (dst == kzstdgpu_BlockDestOffs_Invalid)
    {
        return; // T09: outside its frame (ComputeDestBlockOffsets set the frame status)
    }
    const uint32_t size = blockRef.size;
    const uint32_t src = blockRef.offs; // RAW: byte offset; RLE: fill byte

    // head bytes up to the next 4-aligned destination
    uint32_t head = (4u - (dst & 3u)) & 3u;
    if (head > size)
    {
        head = size;
    }
    const uint32_t nWords = (size - head) >> 2u;
    const uint32_t tail = (size - head) & 3u;
    const uint32_t wordBase = (dst + head) >> 2u;
    const uint32_t tailStart = head + (nWords << 2u);

    const uint32_t fill = (src & 0xffu) * 0x01010101u;

    // byte edges (head < 4, tail < 4: at most one byte per thread each)
    if (t < head)
    {
        if (isRaw)
        {
            const uint32_t s = src + t;
            inoutUnCompressedFramesData[dst + t] =
                (uint8_t)((inCompressedData[s >> 2u] >> ((s & 3u) << 3u)) & 0xffu);
        }
        else
        {
            inoutUnCompressedFramesData[dst + t] = (uint8_t)fill;
        }
    }
    if (t < tail)
    {
        const uint32_t k = tailStart + t;
        if (isRaw)
        {
            const uint32_t s = src + k;
            inoutUnCompressedFramesData[dst + k] =
                (uint8_t)((inCompressedData[s >> 2u] >> ((s & 3u) << 3u)) & 0xffu);
        }
        else
        {
            inoutUnCompressedFramesData[dst + k] = (uint8_t)fill;
        }
    }

    // aligned interior, one word per thread per iteration
    if (isRaw)
    {
        const uint32_t sh = ((src + head) & 3u) << 3u;
        const uint32_t sw = (src + head) >> 2u;
        for (uint32_t w = t; w < nWords; w += MM_THREADS)
        {
            uint32_t v = inCompressedData[sw + w];
            if (sh != 0u)
            {
                v = (v >> sh) | (inCompressedData[sw + w + 1u] << (32u - sh));
            }
            inoutUnCompressedFramesData32[wordBase + w] = v;
        }
    }
    else
    {
        for (uint32_t w = t; w < nWords; w += MM_THREADS)
        {
            inoutUnCompressedFramesData32[wordBase + w] = fill;
        }
    }
}
