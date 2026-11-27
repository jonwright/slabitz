// T09 (slabitz): every match offset against the bytes its frame has produced.
// One 32-lane wave per compressed block, after ComputeDestBlockOffsets and
// FinaliseSequenceOffsets, before the writers. The wave walks the block's
// sequences 32 at a time; a lane's match starts after the frame's bytes before
// the block, the block's earlier sequences (a wave prefix sum of llen + mlen) and
// its own literals. A match whose offset reaches before the frame start (a
// `zstd --patch-from` frame, or a corrupt one) marks the block
// kzstdgpu_BlockDestOffs_Invalid, so neither MemsetMemcpy nor ExecuteSequences
// writes it, and sets the frame status kzstdgpu_FrameStatus_OffsetBeforeFrame.
// The check is here, in parallel, rather than in ExecuteSequences' serial
// per-match loop, where it cost 25% on worst/min_matches.
#include "slang_compat.h"

#include "zstdgpu_shaders.h"
#include "srt_headers/zstdgpu_srt_structs.h"

#define CSO_LANES 32u

[shader("compute")]
[numthreads(CSO_LANES, 1, 1)]
void main(StructuredBuffer<zstdgpu_Counters> inCounters,
          StructuredBuffer<uint32_t> inPerFrameBlockCountCMP,
          RWStructuredBuffer<uint32_t> inoutBlockDestOffs,
          StructuredBuffer<uint32_t> inGlobalBlockIndexPerCmpBlock,
          StructuredBuffer<uint32_t> inPerSeqStreamSeqStart,
          StructuredBuffer<zstdgpu_CompressedBlockData> inCompressedBlocks,
          StructuredBuffer<uint32_t> inDecompressedSequenceLLen,
          StructuredBuffer<uint32_t> inDecompressedSequenceMLen,
          StructuredBuffer<uint32_t> inDecompressedSequenceOffs,
          StructuredBuffer<zstdgpu_OffsetAndSize> inUnCompressedFramesRefs,
          RWStructuredBuffer<uint32_t> inoutFrameStatus,
          uint32_t seqCapacity, uint32_t frameCount,
          uint2 groupId2 : SV_GroupID, uint32_t lane : SV_GroupThreadID)
{
    // P4 step 2b: a queued call whose sequences overflowed the capacity is redone
    // exactly at the wait; the sequence buffers hold nothing usable now.
    if (inCounters[0].Seq_Streams_DecodedItems > seqCapacity)
        return;
    const uint32_t cmpBlockIdx = zstdgpu_ConvertTo32BitGroupId(groupId2, 0);
    if (cmpBlockIdx >= inCounters[0].Blocks_CMP)
        return;
    const uint32_t seqStreamIdx = inCompressedBlocks[cmpBlockIdx].seqStreamIndex;
    if (seqStreamIdx == ~0u)
        return; // no sequences: literals only
    const uint32_t blockIdx = inGlobalBlockIndexPerCmpBlock[cmpBlockIdx];
    const uint32_t dst = inoutBlockDestOffs[blockIdx];
    if (dst == kzstdgpu_BlockDestOffs_Invalid)
        return; // already outside its frame (ComputeDestBlockOffsets)

    const uint32_t seqOfs = inPerSeqStreamSeqStart[seqStreamIdx];
    const uint32_t seqEnd = (seqStreamIdx + 1u == inCounters[0].Seq_Streams)
                          ? inCounters[0].Seq_Streams_DecodedItems
                          : inPerSeqStreamSeqStart[seqStreamIdx + 1u];
    // the last frame whose first compressed block is <= this one
    const uint32_t frameIdx = zstdgpu_BinarySearch(inPerFrameBlockCountCMP, 0, frameCount, cmpBlockIdx);

    // bytes of this frame before the block (ComputeDestBlockOffsets kept the
    // block inside its frame, so this does not wrap)
    uint32_t produced = dst - inUnCompressedFramesRefs[frameIdx].offs;
    uint32_t bad = 0u;
    // uniform trip count: seqOfs and seqEnd are the same on every lane
    for (uint32_t base = seqOfs; base < seqEnd; base += CSO_LANES)
    {
        const uint32_t i = base + lane;
        uint32_t llen = 0u, mlen = 0u, offs = 0u;
        if (i < seqEnd)
        {
            llen = inDecompressedSequenceLLen[i];
            mlen = inDecompressedSequenceMLen[i];
            offs = inDecompressedSequenceOffs[i];
        }
        const uint32_t len = llen + mlen;
        // where this lane's match starts, counted from the frame start
        const uint32_t matchStart = produced + WavePrefixSum(len) + llen;
        if (mlen != 0u && offs > matchStart)
            bad = 1u;
        produced += WaveActiveSum(len);
    }
    if (WaveActiveAnyTrue(bad != 0u) && lane == 0u)
    {
        inoutBlockDestOffs[blockIdx] = kzstdgpu_BlockDestOffs_Invalid;
        inoutFrameStatus[frameIdx] = kzstdgpu_FrameStatus_OffsetBeforeFrame;
    }
}
