// Slang entry for Shaders/ZstdGpuComputePrefixSum.hlsl (DirectStorage, MIT).
// One thread per Huffman-lit table id; computes per-table literal-group count and
// the (decoupled-lookback) inclusive group-prefix, storing it in-place.
#include "slang_compat.h"

#include "zstdgpu_shaders.h"
#include "srt_headers/zstdgpu_srt_structs.h"

[shader("compute")]
[numthreads(kzstdgpu_TgSizeX_PrefixSum_LiteralCount, 1, 1)]
void main(StructuredBuffer<uint32_t> inHufWIdToHufLitId,
          StructuredBuffer<uint32_t> inHufLitIdToLitStreamId,
          RWStructuredBuffer<uint32_t> inoutLitGroupEndPerHuffmanTable,
          RWStructuredBuffer<uint32_t> inoutLitGroupEndPerHuffmanTableLookback,
          RWStructuredBuffer<zstdgpu_Counters> inoutCounters,
          StructuredBuffer<uint32_t> inDispatchArgs,
          uint32_t tgOffset, uint32_t workItemCount, uint32_t literalsPerGroup,
          uint2 groupId2 : SV_GroupID, uint32_t threadId : SV_GroupThreadID)
{
    zstdgpu_ComputePrefixSum_SRT srt;
    srt.inHufWIdToHufLitId                      = inHufWIdToHufLitId;
    srt.inHufLitIdToLitStreamId                 = inHufLitIdToLitStreamId;
    srt.inoutLitGroupEndPerHuffmanTable         = inoutLitGroupEndPerHuffmanTable;
    srt.inoutLitGroupEndPerHuffmanTableLookback = inoutLitGroupEndPerHuffmanTableLookback;
    srt.inoutCounters                           = inoutCounters;
    srt.inDispatchArgs                          = inDispatchArgs;
    srt.tgOffset                                = tgOffset;
    srt.workItemCount                           = workItemCount;
    srt.literalsPerGroup                        = literalsPerGroup;
    if (int32_t(srt.workItemCount) < 0)
    {
        const uint32_t slot = ~srt.workItemCount;
        const uint32_t baseIdx = slot * kzstdgpu_DispatchSlot_StrideInUInt32;
        srt.workItemCount = inDispatchArgs[baseIdx + 1];
    }

    const uint32_t i = zstdgpu_ConvertTo32BitGroupId(groupId2, srt.tgOffset) * kzstdgpu_TgSizeX_PrefixSum_LiteralCount + threadId;
    const uint32_t blockSize = min(kzstdgpu_TgSizeX_PrefixSum_LiteralCount, WaveGetLaneCount());
    const uint32_t thisBlockIndex = WaveReadLaneFirst(i / blockSize);
    const uint32_t thisLocalIndex = i % blockSize;

    if (i >= srt.workItemCount)
        return;

    const uint32_t lastLocalIndex = WaveActiveCountBits(true) - 1u;

    const uint32_t hufLitId = srt.inHufWIdToHufLitId[i];
    uint32_t hufLitStreamCount = 0;

    if (~0u != hufLitId)
    {
        const uint32_t hufLitStreamStart = srt.inHufLitIdToLitStreamId[hufLitId];

        if (hufLitId + 1u < srt.inoutCounters[0].HufLit)
        {
            hufLitStreamCount = srt.inHufLitIdToLitStreamId[hufLitId + 1u] - hufLitStreamStart;
        }
        else
        {
            hufLitStreamCount = srt.inoutCounters[0].HUF_Streams - hufLitStreamStart;
        }
    }
    const uint32_t groupCount = ZSTDGPU_TG_COUNT(hufLitStreamCount, srt.literalsPerGroup);
    const uint32_t waveExclusiveGroupPrefix = WavePrefixSum(groupCount);
    const uint32_t globalExclusiveGroupPrefix = zstdgpu_GlobalExclusivePrefixSum(
        srt.inoutLitGroupEndPerHuffmanTableLookback, waveExclusiveGroupPrefix, groupCount, i, kzstdgpu_TgSizeX_PrefixSum_LiteralCount);

    srt.inoutLitGroupEndPerHuffmanTable[i] = globalExclusiveGroupPrefix + groupCount;

    if (i == srt.workItemCount - 1)
    {
        srt.inoutCounters[0].DecompressLiteralsGroups = srt.inoutLitGroupEndPerHuffmanTable[i];
    }
}
