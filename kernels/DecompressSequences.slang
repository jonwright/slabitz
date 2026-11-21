// Slang entry for zstdgpu_ShaderEntry_DecompressSequences_SingleStream (DirectStorage, MIT).
// Models Shaders/ZstdGpuDecompressSequences_SingleStream_ScalarFseLoad*.hlsl
// (kzstdgpu_DecompressSequences_SingleStream_NoLdsFseCache=1). PC/CUDA.
// The SingleStream decode is fully scalar (only thread 0 of each group runs the recurrence;
// the FSE LDS-cache preload path is disabled by NoLdsFseCache), so numthreads=1. One seq
// stream per group: launched with grid=(Seq_Streams,1,1).
#include "slang_compat.h"

#include "zstdgpu_shaders.h"
#include "srt_headers/zstdgpu_srt_structs.h"

// SingleStream decode is fully scalar (thread 0 only; NoLdsFseCache disables the FSE LDS
// preload), so one thread per group suffices. tgSize is only consumed by the disabled preload.
#define kzstdgpu_TgSizeX_DecompressSequences_SingleStream 1

[shader("compute")]
[numthreads(kzstdgpu_TgSizeX_DecompressSequences_SingleStream, 1, 1)]
void main(RWStructuredBuffer<uint32_t> inoutDecompressedSequenceLLen,
          RWStructuredBuffer<uint32_t> inoutDecompressedSequenceMLen,
          RWStructuredBuffer<uint32_t> inoutDecompressedSequenceOffs,
          RWStructuredBuffer<uint32_t> inoutBlockSizePrefix,
          RWStructuredBuffer<uint32_t> inoutPerSeqStreamFinalOffset1,
          RWStructuredBuffer<uint32_t> inoutPerSeqStreamFinalOffset2,
          RWStructuredBuffer<uint32_t> inoutPerSeqStreamFinalOffset3,
          StructuredBuffer<zstdgpu_Counters> inCounters,
          ByteAddressBuffer inCompressedData,
          StructuredBuffer<zstdgpu_OffsetAndSize> inSeqStreamToRef,
          StructuredBuffer<uint32_t> inSeqStreamToLLenFseId,
          StructuredBuffer<uint32_t> inSeqStreamToOffsFseId,
          StructuredBuffer<uint32_t> inSeqStreamToMLenFseId,
          StructuredBuffer<uint32_t> inSeqStreamToBlockId,
          StructuredBuffer<zstdgpu_FseInfo> inFseInfos,
          StructuredBuffer<uint32_t> inPerSeqStreamSeqStart,
          StructuredBuffer<uint32_t> inFseElems,
          StructuredBuffer<uint32_t> inDispatchArgs,
          uint32_t tgOffset, uint32_t workItemCount,
          uint2 groupId2 : SV_GroupID, uint32_t threadId : SV_GroupThreadID)
{
    zstdgpu_DecompressSequences_SRT srt;
    srt.inoutDecompressedSequenceLLen  = inoutDecompressedSequenceLLen;
    srt.inoutDecompressedSequenceMLen  = inoutDecompressedSequenceMLen;
    srt.inoutDecompressedSequenceOffs  = inoutDecompressedSequenceOffs;
    srt.inoutBlockSizePrefix           = inoutBlockSizePrefix;
    srt.inoutPerSeqStreamFinalOffset1  = inoutPerSeqStreamFinalOffset1;
    srt.inoutPerSeqStreamFinalOffset2  = inoutPerSeqStreamFinalOffset2;
    srt.inoutPerSeqStreamFinalOffset3  = inoutPerSeqStreamFinalOffset3;
    srt.inCounters                     = inCounters;
    srt.inCompressedData               = inCompressedData;
    srt.inSeqStreamToRef               = inSeqStreamToRef;
    srt.inSeqStreamToLLenFseId         = inSeqStreamToLLenFseId;
    srt.inSeqStreamToOffsFseId         = inSeqStreamToOffsFseId;
    srt.inSeqStreamToMLenFseId         = inSeqStreamToMLenFseId;
    srt.inSeqStreamToBlockId           = inSeqStreamToBlockId;
    srt.inFseInfos                     = inFseInfos;
    srt.inPerSeqStreamSeqStart         = inPerSeqStreamSeqStart;
    srt.inFseElems                     = inFseElems;
    srt.inDispatchArgs                 = inDispatchArgs;
    srt.tgOffset                       = tgOffset;
    srt.workItemCount                  = workItemCount;

    const uint32_t groupId = zstdgpu_ConvertTo32BitGroupId(groupId2, tgOffset);
    zstdgpu_ShaderEntry_DecompressSequences_SingleStream(
        srt, groupId, threadId, kzstdgpu_TgSizeX_DecompressSequences_SingleStream);
}
