// Slang entry for zstdgpu_ShaderEntry_ExecuteSequences (DirectStorage, MIT).
// PC/CUDA: numthreads = kzstdgpu_MAX_COPY_SIZE... the 32 specialization
// (WaveGetLaneCount()=32) so MAX_COPY_SIZE=32. One threadgroup per frame (groupId = frameIdx),
// single wave (32 lanes) since WaveGetLaneCount()=32 retires the redundant lanes. Emits literal +
// match runs into the output using the FinaliseSequenceOffsets-resolved byte offsets.
// T09: blocks marked kzstdgpu_BlockDestOffs_Invalid (ComputeDestBlockOffsets,
// CheckSequenceOffsets) are skipped.
#include "slang_compat.h"

#include "zstdgpu_shaders.h"
#include "srt_headers/zstdgpu_srt_structs.h"

#define MAX_COPY_SIZE 32

[shader("compute")]
[numthreads(MAX_COPY_SIZE, 1, 1)]
void main(StructuredBuffer<uint8_t> inCompressedData,
          StructuredBuffer<uint8_t> inDecompressedLiterals,
          RWStructuredBuffer<uint8_t> inoutUnCompressedFramesData,
          RWStructuredBuffer<uint32_t> inoutUnCompressedFramesData32,
          RWStructuredBuffer<uint4> inoutUnCompressedFramesData16,
          StructuredBuffer<zstdgpu_Counters> inCounters,
          StructuredBuffer<uint32_t> inPerFrameBlockCountCMP,
          StructuredBuffer<uint32_t> inBlockSizePrefix,
          StructuredBuffer<uint32_t> inBlockDestOffs,
          StructuredBuffer<uint32_t> inDecompressedSequenceLLen,
          StructuredBuffer<uint32_t> inDecompressedSequenceMLen,
          StructuredBuffer<uint32_t> inDecompressedSequenceOffs,
          StructuredBuffer<uint32_t> inGlobalBlockIndexPerCmpBlock,
          StructuredBuffer<uint32_t> inPerSeqStreamSeqStart,
          StructuredBuffer<zstdgpu_CompressedBlockData> inCompressedBlocks,
          uint32_t litCapacity, uint32_t seqCapacity,
          uint2 groupId2 : SV_GroupID, uint32_t i : SV_GroupThreadID)
{
    // P4 step 2b: the host sizes the literal and sequence buffers from per-context
    // capacities instead of reading the counts back. If this call's real counts
    // exceed them, every kernel touching those buffers does nothing, and the host
    // finds out at sbz_dctx_wait (one readback) and redoes the call exactly.
    if (inCounters[0].HUF_Streams_DecodedBytes > litCapacity ||
        inCounters[0].Seq_Streams_DecodedItems > seqCapacity)
        return;
    // Retire redundant waves (just recomputing the same result) when the GPU wave is narrower than the group.
    if (i >= WaveGetLaneCount())
        return;

    zstdgpu_ExecuteSequences_SRT srt;
    srt.inCompressedData                = inCompressedData;
    srt.inDecompressedLiterals          = inDecompressedLiterals;
    srt.inoutUnCompressedFramesData     = inoutUnCompressedFramesData;
    srt.inoutUnCompressedFramesData32   = inoutUnCompressedFramesData32;
    srt.inoutUnCompressedFramesData16   = inoutUnCompressedFramesData16;
    srt.inCounters                      = inCounters;
    srt.inPerFrameBlockCountCMP         = inPerFrameBlockCountCMP;
    srt.inBlockSizePrefix               = inBlockSizePrefix;
    srt.inBlockDestOffs                 = inBlockDestOffs;
    srt.inDecompressedSequenceLLen      = inDecompressedSequenceLLen;
    srt.inDecompressedSequenceMLen      = inDecompressedSequenceMLen;
    srt.inDecompressedSequenceOffs      = inDecompressedSequenceOffs;
    srt.inGlobalBlockIndexPerCmpBlock   = inGlobalBlockIndexPerCmpBlock;
    srt.inPerSeqStreamSeqStart          = inPerSeqStreamSeqStart;
    srt.inCompressedBlocks              = inCompressedBlocks;

    const uint32_t frameIdx = zstdgpu_ConvertTo32BitGroupId(groupId2, 0);
    zstdgpu_ShaderEntry_ExecuteSequences(srt, frameIdx);
}
