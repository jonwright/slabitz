// Slang entry for zstdgpu_PropagateFseTableIndex (DirectStorage, MIT).
// Replicates Shaders/ZstdGpuPropagateFseIndex.hlsl with a by-value SRT.
#include "slang_compat.h"

#include "zstdgpu_shaders.h"
#include "srt_headers/zstdgpu_srt_structs.h"

[shader("compute")]
[numthreads(kzstdgpu_TgSizeX_PropagateFseIndex, 1, 1)]
void main(RWStructuredBuffer<uint32_t> inoutFseIds,
          RWStructuredBuffer<uint32_t> inoutFseIndexLookback,
          StructuredBuffer<uint32_t> inDispatchArgs,
          StructuredBuffer<zstdgpu_Counters> inCounters,
          uint32_t tgOffset, uint32_t workItemCount,
          uint2 groupId2 : SV_GroupID, uint threadId : SV_GroupThreadID)
{
    zstdgpu_PropagateFseIndex_SRT srt;
    srt.inoutFseIds           = inoutFseIds;
    srt.inoutFseIndexLookback = inoutFseIndexLookback;
    srt.inDispatchArgs        = inDispatchArgs;
    srt.tgOffset              = tgOffset;
    srt.workItemCount         = workItemCount;
    // P4 step 2b: workItemCount ~0u means the count is on the device (Seq_Streams)
    if (srt.workItemCount == 0xFFFFFFFFu)
        srt.workItemCount = inCounters[0].Seq_Streams;
    // fixup code for executeIndirectWorkaround
    else if (int32_t(srt.workItemCount) < 0)
    {
        const uint32_t slot = ~srt.workItemCount;
        const uint32_t baseIdx = slot * kzstdgpu_DispatchSlot_StrideInUInt32;
        srt.workItemCount = inDispatchArgs[baseIdx + 1];
    }

    const uint32_t i = zstdgpu_ConvertTo32BitGroupId(groupId2, srt.tgOffset) * kzstdgpu_TgSizeX_PropagateFseIndex + threadId;

    if (i >= srt.workItemCount)
    {
        return;
    }
    srt.inoutFseIds[i] = zstdgpu_PropagateFseTableIndex(srt.inoutFseIndexLookback, srt.inoutFseIds[i], i);
}
