// Slang entry for zstdgpu_ShaderEntry_DecompressSequences_MultiStream (DirectStorage, MIT).
// One seq stream per thread: seqStreamIdx = groupId * streamsPerGroup + threadId. The decode loop
// is identical to SingleStream but uses the global inFseElems buffer (no NoLdsFseCache single-thread
// scalarization); numthreads = streamsPerGroup. Output is identical to SingleStream; this exercises
// the parallel stream distribution (and any wave-diverge risk) on the GPU.
#include "slang_compat.h"

#include "zstdgpu_shaders.h"
#include "srt_headers/zstdgpu_srt_structs.h"

#define STREAMS_PER_GROUP 32

[shader("compute")]
[numthreads(STREAMS_PER_GROUP, 1, 1)]
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
          uint32_t tgOffset, uint32_t workItemCount, uint32_t seqCapacity,
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

    // P4 step 2b: the host sizes the literal and sequence buffers from per-context
    // capacities instead of reading the counts back. If this call's real counts
    // exceed them, every kernel touching those buffers does nothing, and the host
    // finds out at sbz_dctx_wait (one readback) and redoes the call exactly.
    if (inCounters[0].Seq_Streams_DecodedItems > seqCapacity)
        return;
    const uint32_t groupId = zstdgpu_ConvertTo32BitGroupId(groupId2, tgOffset);
    zstdgpu_ShaderEntry_DecompressSequences_MultiStream(srt, groupId, threadId, STREAMS_PER_GROUP);
}
