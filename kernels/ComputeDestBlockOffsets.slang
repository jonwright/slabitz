// Slang entry for zstdgpu_ShaderEntry_ComputeDestBlockOffsets (DirectStorage, MIT).
// Shaders/ZstdGpuComputeDestBlockOffsets.hlsl (body inline). One thread per block:
// find the frame containing blockIdx, then emit the block's absolute destination
// byte offset = frame base offset + (block offset within the frame). PC/CUDA:
// numthreads = kzstdgpu_TgSizeX_ComputeDestBlockOffset = 128.
// T09: the block must lie inside its frame's [base, base + content size), decode to
// at most 128 KiB, and the frame's last block must end exactly at the content
// size. Otherwise the block's offset is kzstdgpu_BlockDestOffs_Invalid (no kernel
// writes it) and the frame gets a status, so a frame that decodes to more than it
// declared cannot overwrite its neighbour.
#include "slang_compat.h"

#include "zstdgpu_shaders.h"
#include "srt_headers/zstdgpu_srt_structs.h"

[shader("compute")]
[numthreads(kzstdgpu_TgSizeX_ComputeDestBlockOffset, 1, 1)]
void main(RWStructuredBuffer<uint32_t> inoutBlockDestOffs,
          StructuredBuffer<uint32_t> inBlockSizePrefix,
          StructuredBuffer<uint32_t> inPerFrameBlockCountAll,
          StructuredBuffer<zstdgpu_OffsetAndSize> inUnCompressedFramesRefs,
          RWStructuredBuffer<uint32_t> inoutFrameStatus,
          uint32_t tgOffset, uint32_t workItemCount, uint32_t frameCount,
          uint2 groupId2 : SV_GroupID, uint32_t threadId : SV_GroupThreadID)
{
    const uint32_t blockIdx = zstdgpu_ConvertTo32BitGroupId(groupId2, tgOffset) *
                              kzstdgpu_TgSizeX_ComputeDestBlockOffset + threadId;
    if (blockIdx >= workItemCount)
    {
        return;
    }

    const uint32_t frameIdx = zstdgpu_BinarySearch(inPerFrameBlockCountAll, 0, frameCount, blockIdx);
    const uint32_t firstFrameBlockIdx = inPerFrameBlockCountAll[frameIdx];

    uint32_t firstFrameBlockOffset = 0;
    if (firstFrameBlockIdx > 0)
    {
        firstFrameBlockOffset = inBlockSizePrefix[firstFrameBlockIdx - 1];
    }

    uint32_t blockOffset = 0;
    if (blockIdx > 0)
    {
        blockOffset = inBlockSizePrefix[blockIdx - 1];
    }

    const zstdgpu_OffsetAndSize frame = inUnCompressedFramesRefs[frameIdx];
    // inBlockSizePrefix is inclusive here: this block's decoded size
    const uint32_t blockSize = inBlockSizePrefix[blockIdx] - blockOffset;
    const uint32_t nextFrameBlockIdx = (frameIdx + 1u < frameCount)
                                     ? inPerFrameBlockCountAll[frameIdx + 1u]
                                     : workItemCount;
    // 64-bit: corrupt sizes may make the 32-bit prefix wrap
    const uint64_t inFrameEnd = (uint64_t)(blockOffset - firstFrameBlockOffset) + blockSize;
    uint32_t status = 0u;
    if (blockSize > 128u * 1024u)
    {
        status = kzstdgpu_FrameStatus_BlockTooLarge;
    }
    else if (blockOffset < firstFrameBlockOffset || inFrameEnd > (uint64_t)frame.size)
    {
        status = kzstdgpu_FrameStatus_OutputPastContentSize;
    }
    else if (blockIdx + 1u == nextFrameBlockIdx && inFrameEnd != (uint64_t)frame.size)
    {
        status = kzstdgpu_FrameStatus_OutputShortOfContentSize;
    }
    if (status != 0u)
    {
        inoutBlockDestOffs[blockIdx] = kzstdgpu_BlockDestOffs_Invalid;
        inoutFrameStatus[frameIdx] = status;
        return;
    }
    inoutBlockDestOffs[blockIdx] = frame.offs + blockOffset - firstFrameBlockOffset;
}
