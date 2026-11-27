// T09 (slabitz): every match offset against the bytes its frame has produced.
// One 256-thread group per compressed block, after ComputeDestBlockOffsets and
// FinaliseSequenceOffsets, before the writers. A match starts after the frame's
// bytes before the block, the block's earlier sequences (llen + mlen) and its
// own literals. A match whose offset reaches before the frame start (a
// `zstd --patch-from` frame, or a corrupt one) marks the block
// kzstdgpu_BlockDestOffs_Invalid, so neither MemsetMemcpy nor ExecuteSequences
// writes it, and sets the frame status kzstdgpu_FrameStatus_OffsetBeforeFrame.
// The check is here, in parallel, rather than in ExecuteSequences' serial
// per-match loop, where it cost 25% on worst/min_matches.
#include "slang_compat.h"

#include "zstdgpu_shaders.h"
#include "srt_headers/zstdgpu_srt_structs.h"

// 256 threads per block: each thread walks a contiguous slice of the block's
// sequences twice (sum, then check), with one group scan of the slice sums in
// between. One 32-lane wave walking 32 sequences per step took 631 us on
// worst/min_matches (~43k sequences per block).
#define CSO_THREADS 256u
#define CSO_MAX_WAVES (CSO_THREADS / 4u) // any wave width down to 4 lanes

groupshared uint32_t gsWaveSum[CSO_MAX_WAVES];
groupshared uint32_t gsBad;

[shader("compute")]
[numthreads(CSO_THREADS, 1, 1)]
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
          uint2 groupId2 : SV_GroupID, uint32_t t : SV_GroupThreadID)
{
    // Every early return below depends only on the group, so the whole group
    // returns together (the barriers further down are uniform).
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
    const uint32_t produced = dst - inUnCompressedFramesRefs[frameIdx].offs;

    // this thread's slice [beg, end) of the block's sequences
    const uint32_t n = seqEnd > seqOfs ? seqEnd - seqOfs : 0u;
    const uint32_t per = (n + CSO_THREADS - 1u) / CSO_THREADS;
    const uint32_t beg = seqOfs + min(t * per, n);
    const uint32_t end = seqOfs + min(t * per + per, n);

    // pass 1: bytes the slice produces
    uint32_t sum = 0u;
    for (uint32_t i = beg; i < end; ++i)
        sum += inDecompressedSequenceLLen[i] + inDecompressedSequenceMLen[i];

    // exclusive scan of the slice sums over the group
    const uint32_t laneCount = WaveGetLaneCount();
    const uint32_t waveIdx = t / laneCount;
    const uint32_t lanePrefix = WavePrefixSum(sum);
    const uint32_t waveTotal = WaveActiveSum(sum);
    if (WaveGetLaneIndex() == 0u)
        gsWaveSum[waveIdx] = waveTotal;
    if (t == 0u)
        gsBad = 0u;
    GroupMemoryBarrierWithGroupSync();
    uint32_t start = produced + lanePrefix;
    for (uint32_t w = 0u; w < waveIdx; ++w)
        start += gsWaveSum[w];

    // pass 2: each match against the bytes of the frame before it
    uint32_t bad = 0u;
    for (uint32_t i = beg; i < end; ++i)
    {
        const uint32_t llen = inDecompressedSequenceLLen[i];
        const uint32_t mlen = inDecompressedSequenceMLen[i];
        const uint32_t offs = inDecompressedSequenceOffs[i];
        if (mlen != 0u && offs > start + llen)
            bad = 1u;
        start += llen + mlen;
    }
    if (bad != 0u)
        gsBad = 1u;
    GroupMemoryBarrierWithGroupSync();
    if (t == 0u && gsBad != 0u)
    {
        inoutBlockDestOffs[blockIdx] = kzstdgpu_BlockDestOffs_Invalid;
        inoutFrameStatus[frameIdx] = kzstdgpu_FrameStatus_OffsetBeforeFrame;
    }
}
