// Slang entry for ZstdGpuPrefixSum.hlsl (DirectStorage, MIT).
// One thread per element; WavePrefixSum + decoupled-lookback global exclusive prefix.
#include "slang_compat.h"

#include "zstdgpu_shaders.h"
#include "srt_headers/zstdgpu_srt_structs.h"

[shader("compute")]
[numthreads(kzstdgpu_TgSizeX_PrefixSum, 1, 1)]
void main(RWStructuredBuffer<uint32_t> inoutInCountsOutPrefix,
          RWStructuredBuffer<uint32_t> inoutInCountsOutPrefixLookback,
          StructuredBuffer<uint32_t> inDispatchArgs,
          uint32_t tgOffset, uint32_t workItemCount, uint32_t outputInclusive,
          uint2 groupId2 : SV_GroupID, uint threadId : SV_GroupThreadID)
{
    zstdgpu_PrefixSum_SRT srt;
    srt.inoutInCountsOutPrefix         = inoutInCountsOutPrefix;
    srt.inoutInCountsOutPrefixLookback = inoutInCountsOutPrefixLookback;
    srt.inDispatchArgs                 = inDispatchArgs;
    srt.tgOffset                       = tgOffset;
    srt.workItemCount                  = workItemCount;
    srt.outputInclusive                = outputInclusive;
    if (int32_t(srt.workItemCount) < 0)
    {
        const uint32_t slot = ~srt.workItemCount;
        const uint32_t baseIdx = slot * kzstdgpu_DispatchSlot_StrideInUInt32;
        srt.workItemCount = inDispatchArgs[baseIdx + 1];
    }

    const uint32_t i = zstdgpu_ConvertTo32BitGroupId(groupId2, srt.tgOffset) * kzstdgpu_TgSizeX_PrefixSum + threadId;
    if (i >= srt.workItemCount)
        return;

    const uint32_t count = srt.inoutInCountsOutPrefix[i];
    const uint32_t countExclusiveBlockPrefix = WavePrefixSum(count);
    const uint32_t exclusivePrefixSum = zstdgpu_GlobalExclusivePrefixSum(
        srt.inoutInCountsOutPrefixLookback, countExclusiveBlockPrefix, count, i, kzstdgpu_TgSizeX_PrefixSum);
    srt.inoutInCountsOutPrefix[i] = exclusivePrefixSum + (srt.outputInclusive > 0 ? count : 0);
}
