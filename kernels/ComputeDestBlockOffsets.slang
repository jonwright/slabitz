// Slang entry for zstdgpu_ShaderEntry_ComputeDestBlockOffsets (DirectStorage, MIT).
// Shaders/ZstdGpuComputeDestBlockOffsets.hlsl (body inline). One thread per block:
// find the frame containing blockIdx, then emit the block's absolute destination
// byte offset = frame base offset + (block offset within the frame). PC/CUDA:
// numthreads = kzstdgpu_TgSizeX_ComputeDestBlockOffset = 128.
#include "slang_compat.h"

#include "zstdgpu_shaders.h"
#include "srt_headers/zstdgpu_srt_structs.h"

[shader("compute")]
[numthreads(kzstdgpu_TgSizeX_ComputeDestBlockOffset, 1, 1)]
void main(RWStructuredBuffer<uint32_t> inoutBlockDestOffs,
          StructuredBuffer<uint32_t> inBlockSizePrefix,
          StructuredBuffer<uint32_t> inPerFrameBlockCountAll,
          StructuredBuffer<zstdgpu_OffsetAndSize> inUnCompressedFramesRefs,
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

    inoutBlockDestOffs[blockIdx] = inUnCompressedFramesRefs[frameIdx].offs +
                                   blockOffset - firstFrameBlockOffset;
}
