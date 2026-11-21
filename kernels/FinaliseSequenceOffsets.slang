// Slang entry for zstdgpu_ShaderEntry_FinaliseSequenceOffsets (DirectStorage, MIT).
// PC/CUDA: numthreads = kzstdgpu_TgSizeX_FinaliseSequenceOffsets = 256 (one thread per sequence,
// flattened global index = groupThreadId + groupId*TgSizeX). Resolves DecompressSequences' encoded
// offsets to the real zstd byte distance.
#include "slang_compat.h"

#include "zstdgpu_shaders.h"
#include "srt_headers/zstdgpu_srt_structs.h"

[shader("compute")]
[numthreads(kzstdgpu_TgSizeX_FinaliseSequenceOffsets, 1, 1)]
void main(RWStructuredBuffer<uint32_t> inoutDecompressedSequenceOffs,
          StructuredBuffer<zstdgpu_Counters> inCounters,
          StructuredBuffer<uint32_t> inPerSeqStreamFinalOffset1,
          StructuredBuffer<uint32_t> inPerSeqStreamFinalOffset2,
          StructuredBuffer<uint32_t> inPerSeqStreamFinalOffset3,
          StructuredBuffer<uint32_t> inPerSeqStreamSeqStart,
          StructuredBuffer<uint32_t> inPerFrameBlockCountAll,
          StructuredBuffer<uint32_t> inPerFrameSeqStreamMinIdx,
          StructuredBuffer<uint32_t> inSeqStreamToBlockId,
          StructuredBuffer<uint32_t> inDispatchArgs,
          uint32_t tgOffset, uint32_t workItemCount, uint32_t seqCapacity,
          uint2 groupId2 : SV_GroupID, uint32_t threadId : SV_GroupThreadID)
{
    zstdgpu_FinaliseSequenceOffsets_SRT srt;
    srt.inoutDecompressedSequenceOffs = inoutDecompressedSequenceOffs;
    srt.inCounters                    = inCounters;
    srt.inPerSeqStreamFinalOffset1    = inPerSeqStreamFinalOffset1;
    srt.inPerSeqStreamFinalOffset2    = inPerSeqStreamFinalOffset2;
    srt.inPerSeqStreamFinalOffset3    = inPerSeqStreamFinalOffset3;
    srt.inPerSeqStreamSeqStart        = inPerSeqStreamSeqStart;
    srt.inPerFrameBlockCountAll       = inPerFrameBlockCountAll;
    srt.inPerFrameSeqStreamMinIdx     = inPerFrameSeqStreamMinIdx;
    srt.inSeqStreamToBlockId          = inSeqStreamToBlockId;
    srt.inDispatchArgs                = inDispatchArgs;
    srt.tgOffset                      = tgOffset;
    srt.workItemCount                 = workItemCount;

    // P4 step 2b: the host sizes the literal and sequence buffers from per-context
    // capacities instead of reading the counts back. If this call's real counts
    // exceed them, every kernel touching those buffers does nothing, and the host
    // finds out at sbz_dctx_wait (one readback) and redoes the call exactly.
    if (inCounters[0].Seq_Streams_DecodedItems > seqCapacity)
        return;
    const uint32_t groupId = zstdgpu_ConvertTo32BitGroupId(groupId2, tgOffset);
    const uint32_t i = threadId + groupId * kzstdgpu_TgSizeX_FinaliseSequenceOffsets;
    zstdgpu_ShaderEntry_FinaliseSequenceOffsets(srt, i);
}
