// Slang entry for zstdgpu_ShaderEntry_InitResources (DirectStorage, MIT).
// Replicates Shaders/ZstdGpuInitResources.hlsl with a by-value SRT parameter.
#include "slang_compat.h"

#include "zstdgpu_shaders.h"
#include "srt_headers/zstdgpu_srt_structs.h"

[shader("compute")]
[numthreads(kzstdgpu_TgSizeX_InitCounters, 1, 1)]
void main(RWStructuredBuffer<zstdgpu_Counters> inoutCounters,
          StructuredBuffer<int32_t> inFseProbsDefault,
          RWStructuredBuffer<int32_t> inoutFseProbs,
          RWStructuredBuffer<zstdgpu_FseInfo> inoutFseInfos,
          RWStructuredBuffer<uint32_t> inoutFseElems,
          uint32_t initResourcesStage,
          uint i : SV_DispatchThreadId)
{
    zstdgpu_InitResources_SRT srt;
    srt.inoutCounters      = inoutCounters;
    srt.inFseProbsDefault  = inFseProbsDefault;
    srt.inoutFseProbs      = inoutFseProbs;
    srt.inoutFseInfos      = inoutFseInfos;
    srt.inoutFseElems      = inoutFseElems;
    srt.initResourcesStage = initResourcesStage;

    zstdgpu_ShaderEntry_InitResources(srt, i);
}
