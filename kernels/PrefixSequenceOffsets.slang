// Slang entry for Shaders/ZstdGpuPrefixSequenceOffsets.hlsl (DirectStorage, MIT).
// The reference shader has no ShaderEntry helper: its whole main() body is
// inline (prefixed by the WAVE_*/LOOKBACK_* macros). We reproduce it directly with
// a by-value SRT. The `#if 0` fences are collapsed to their active (#else) lines;
// `#if kzstdgpu_TgSizeX_PrefixSequenceOffsets > .. #error` blocks are inactive (TgSizeX = 32).
#include "slang_compat.h"

#include "zstdgpu_shaders.h"
#include "srt_headers/zstdgpu_srt_structs.h"

[shader("compute")]
[numthreads(kzstdgpu_TgSizeX_PrefixSequenceOffsets, 1, 1)]
void main(RWStructuredBuffer<uint32_t> inoutPerSeqStreamFinalOffset1,
          RWStructuredBuffer<uint32_t> inoutPerSeqStreamFinalOffset2,
          RWStructuredBuffer<uint32_t> inoutPerSeqStreamFinalOffset3,
          RWStructuredBuffer<uint32_t> inoutPerSeqStreamFinalOffset1Lookback,
          RWStructuredBuffer<uint32_t> inoutPerSeqStreamFinalOffset2Lookback,
          RWStructuredBuffer<uint32_t> inoutPerSeqStreamFinalOffset3Lookback,
          StructuredBuffer<uint32_t> inPerFrameSeqStreamMinIdx,
          StructuredBuffer<uint32_t> inPerFrameBlockCountAll,
          StructuredBuffer<uint32_t> inSeqStreamToBlockId,
          StructuredBuffer<zstdgpu_Counters> inCounters,
          StructuredBuffer<uint32_t> inDispatchArgs,
          uint32_t tgOffset, uint32_t workItemCount, uint32_t frameCount,
          uint2 groupId2 : SV_GroupID, uint32_t threadId : SV_GroupThreadID)
{
    zstdgpu_PrefixSequenceOffsets_SRT srt;
    srt.inoutPerSeqStreamFinalOffset1 = inoutPerSeqStreamFinalOffset1;
    srt.inoutPerSeqStreamFinalOffset2 = inoutPerSeqStreamFinalOffset2;
    srt.inoutPerSeqStreamFinalOffset3 = inoutPerSeqStreamFinalOffset3;
    srt.inoutPerSeqStreamFinalOffset1Lookback = inoutPerSeqStreamFinalOffset1Lookback;
    srt.inoutPerSeqStreamFinalOffset2Lookback = inoutPerSeqStreamFinalOffset2Lookback;
    srt.inoutPerSeqStreamFinalOffset3Lookback = inoutPerSeqStreamFinalOffset3Lookback;
    srt.inPerFrameSeqStreamMinIdx = inPerFrameSeqStreamMinIdx;
    srt.inPerFrameBlockCountAll = inPerFrameBlockCountAll;
    srt.inSeqStreamToBlockId = inSeqStreamToBlockId;
    srt.inCounters = inCounters;
    srt.inDispatchArgs = inDispatchArgs;
    srt.tgOffset = tgOffset;
    srt.workItemCount = workItemCount;
    srt.frameCount = frameCount;
    if (int32_t(srt.workItemCount) < 0)
    {
        const uint32_t slot = ~srt.workItemCount;
        const uint32_t baseIdx = slot * kzstdgpu_DispatchSlot_StrideInUInt32;
        srt.workItemCount = inDispatchArgs[baseIdx + 1];
    }

    const uint32_t i = zstdgpu_ConvertTo32BitGroupId(groupId2, srt.tgOffset) * kzstdgpu_TgSizeX_PrefixSequenceOffsets + threadId;
    const uint32_t blockSize = min(kzstdgpu_TgSizeX_PrefixSequenceOffsets, WaveGetLaneCount());
    const uint32_t thisBlockIndex = WaveReadLaneFirst(i / blockSize);
    const uint32_t thisLocalIndex = i % blockSize;

    if (i >= srt.inCounters[0].Seq_Streams)
        return;

    const uint32_t blockId = srt.inSeqStreamToBlockId[i];
    const uint32_t frameId = zstdgpu_BinarySearch(srt.inPerFrameBlockCountAll, 0, srt.frameCount, blockId);

    const uint32_t seqStreamIdxFirstInFrame = srt.inPerFrameSeqStreamMinIdx[frameId];

    uint32_t3 o = uint32_t3(
        srt.inoutPerSeqStreamFinalOffset1[i],
        srt.inoutPerSeqStreamFinalOffset2[i],
        srt.inoutPerSeqStreamFinalOffset3[i]
    );

    bool needsPropagationAfterLookback = false;

    const bool isFirstSequenceStreamInFrame = seqStreamIdxFirstInFrame != ~0u && seqStreamIdxFirstInFrame == i;
    if (isFirstSequenceStreamInFrame)
    {
        const uint32_t3 b = uint32_t3(1, 4, 8) + 3;

        zstdgpu_DecodeSeqRepeatOffsetsAndApplyPreviousOffsets(o.x, o.y, o.z, b.x, b.y, b.z);

        needsPropagationAfterLookback = true;
    }

    const uint32_t lastLocalIndex = WaveActiveCountBits(true) - 1u;

    #define LOOKBACK_STORE(name, prev, value) InterlockedCompareStore(srt.inout##name##Lookback[thisBlockIndex], prev, (value))

    #define WAVE_SHUFFLE(v, and_mask, or_mask, xor_mask) WaveReadLaneAt(v, ((WaveGetLaneIndex() & (and_mask)) | (or_mask)) ^ (xor_mask))

    #define WAVE_BROADCAST(v, group_size, group_lane) WAVE_SHUFFLE(v, ~(group_size - 1u), group_lane, 0)

    #define WAVE_PROPAGATE_STEP(o, group_size)  \
        if (blockSize >= group_size) \
        {                                                                                                           \
            const uint32_t offsetOR = o.x | o.y | o.z;                                                              \
            if (WaveActiveAnyTrue(zstdgpu_DecodeSeqRepeatOffsetEncoded(offsetOR) > 0))                              \
            {                                                                                                       \
                const uint32_t3 b = WAVE_BROADCAST(o, group_size, group_size / 2u - 1u);                            \
                [branch] if ((WaveGetLaneIndex() & (group_size / 2u)))                                              \
                {                                                                                                   \
                    zstdgpu_DecodeSeqRepeatOffsetsAndApplyPreviousOffsets(o.x, o.y, o.z, b.x, b.y, b.z);            \
                }                                                                                                   \
            }                                                                                                       \
        }

    const uint32_t offsetOR = o.x | o.y | o.z;

    uint32_t3 lastLaneSelfResult = o;

    /** wave-uniform branch */  if (0 == zstdgpu_DecodeSeqRepeatOffsetEncoded(WaveReadLaneAt(offsetOR, lastLocalIndex)))
    {
        needsPropagationAfterLookback = zstdgpu_DecodeSeqRepeatOffsetEncoded(offsetOR) > 0;

        if (WaveGetLaneIndex() == lastLocalIndex)
        {
            LOOKBACK_STORE(PerSeqStreamFinalOffset1, 0, zstdgpu_Encode30BitLookbackFull(o.x));
            LOOKBACK_STORE(PerSeqStreamFinalOffset2, 0, zstdgpu_Encode30BitLookbackFull(o.y));
            LOOKBACK_STORE(PerSeqStreamFinalOffset3, 0, zstdgpu_Encode30BitLookbackFull(o.z));
        }
    }
    else
    {
        uint32_t3 p = o;
        WAVE_PROPAGATE_STEP(p, 2);
        WAVE_PROPAGATE_STEP(p, 4);
        WAVE_PROPAGATE_STEP(p, 8);
        WAVE_PROPAGATE_STEP(p, 16);
        WAVE_PROPAGATE_STEP(p, 32);

        needsPropagationAfterLookback = zstdgpu_DecodeSeqRepeatOffsetEncoded(p.x | p.y | p.z) > 0;

        if (WaveActiveAnyTrue(needsPropagationAfterLookback) == false)
        {
            srt.inoutPerSeqStreamFinalOffset1[i] = p.x;
            srt.inoutPerSeqStreamFinalOffset2[i] = p.y;
            srt.inoutPerSeqStreamFinalOffset3[i] = p.z;
        }

        if (WaveGetLaneIndex() == lastLocalIndex)
        {
            if (needsPropagationAfterLookback)
            {
                lastLaneSelfResult.x = zstdgpu_Encode30BitLookbackSelf(p.x);
                lastLaneSelfResult.y = zstdgpu_Encode30BitLookbackSelf(p.y);
                lastLaneSelfResult.z = zstdgpu_Encode30BitLookbackSelf(p.z);

                LOOKBACK_STORE(PerSeqStreamFinalOffset1, 0, lastLaneSelfResult.x);
                LOOKBACK_STORE(PerSeqStreamFinalOffset2, 0, lastLaneSelfResult.y);
                LOOKBACK_STORE(PerSeqStreamFinalOffset3, 0, lastLaneSelfResult.z);
            }
            else
            {
                LOOKBACK_STORE(PerSeqStreamFinalOffset1, 0, zstdgpu_Encode30BitLookbackFull(p.x));
                LOOKBACK_STORE(PerSeqStreamFinalOffset2, 0, zstdgpu_Encode30BitLookbackFull(p.y));
                LOOKBACK_STORE(PerSeqStreamFinalOffset3, 0, zstdgpu_Encode30BitLookbackFull(p.z));
            }
        }
    }

    if (0 != thisBlockIndex && WaveActiveAnyTrue(needsPropagationAfterLookback))
    {
        uint32_t prevBlockIndex = thisBlockIndex - 1u;

        #define LOOKBACK_READ(name)     \
            uint32_t prev_##name;       \
            InterlockedAdd(srt.inout##name##Lookback[prevBlockIndex], 0, prev_##name)

        if (WaveIsFirstLane())
        {
        for (;;)
        {
            LOOKBACK_READ(PerSeqStreamFinalOffset1);
            LOOKBACK_READ(PerSeqStreamFinalOffset2);
            LOOKBACK_READ(PerSeqStreamFinalOffset3);

            const uint32_t prevBlock_Bits = prev_PerSeqStreamFinalOffset1 & prev_PerSeqStreamFinalOffset2 & prev_PerSeqStreamFinalOffset3;

            const uint32_t prevBlock_Flags = zstdgpu_Decode30BitLookbackFlags(prevBlock_Bits);
            if (prevBlock_Flags > 0)
            {
                prev_PerSeqStreamFinalOffset1 = zstdgpu_Decode30BitLookbackValue(prev_PerSeqStreamFinalOffset1);
                prev_PerSeqStreamFinalOffset2 = zstdgpu_Decode30BitLookbackValue(prev_PerSeqStreamFinalOffset2);
                prev_PerSeqStreamFinalOffset3 = zstdgpu_Decode30BitLookbackValue(prev_PerSeqStreamFinalOffset3);
                {
                    zstdgpu_DecodeSeqRepeatOffsetsAndApplyPreviousOffsets(o.x, o.y, o.z, prev_PerSeqStreamFinalOffset1, prev_PerSeqStreamFinalOffset2, prev_PerSeqStreamFinalOffset3);

                    if (zstdgpu_Check30BitLookbackFull(prevBlock_Flags) > 0 || prevBlockIndex == 0)
                    {
                        break;
                    }
                    else
                    {
                        --prevBlockIndex;
                    }
                }
            }
            /* else -- just loop waiting for the current block */
        }
        }
    }

    if (WaveActiveAnyTrue(needsPropagationAfterLookback))
    {
        uint32_t3 p = o;
        WAVE_PROPAGATE_STEP(p, 2);
        WAVE_PROPAGATE_STEP(p, 4);
        WAVE_PROPAGATE_STEP(p, 8);
        WAVE_PROPAGATE_STEP(p, 16);
        WAVE_PROPAGATE_STEP(p, 32);

        if (needsPropagationAfterLookback)
        {
            if (WaveGetLaneIndex() == lastLocalIndex)
            {
                LOOKBACK_STORE(PerSeqStreamFinalOffset1, lastLaneSelfResult.x, zstdgpu_Encode30BitLookbackFull(p.x));
                LOOKBACK_STORE(PerSeqStreamFinalOffset2, lastLaneSelfResult.y, zstdgpu_Encode30BitLookbackFull(p.y));
                LOOKBACK_STORE(PerSeqStreamFinalOffset3, lastLaneSelfResult.z, zstdgpu_Encode30BitLookbackFull(p.z));
            }
        }

        srt.inoutPerSeqStreamFinalOffset1[i] = p.x;
        srt.inoutPerSeqStreamFinalOffset2[i] = p.y;
        srt.inoutPerSeqStreamFinalOffset3[i] = p.z;
    }
    else if (isFirstSequenceStreamInFrame)
    {
        srt.inoutPerSeqStreamFinalOffset1[i] = o.x;
        srt.inoutPerSeqStreamFinalOffset2[i] = o.y;
        srt.inoutPerSeqStreamFinalOffset3[i] = o.z;
    }
}
