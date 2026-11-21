// Slang entry for zstdgpu_ShaderEntry_ParseFrames (DirectStorage, MIT).
// One thread per compressed-data group; scans the frame/block structure and fills
// per-frame block counts, block refs and block-size prefixes.
#include "slang_compat.h"
#include "zstdgpu_shaders.h"
#include "srt_headers/zstdgpu_srt_structs.h"

[shader("compute")]
[numthreads(kzstdgpu_TgSizeX_ParseCompressedBlocks, 1, 1)]
void main(
    StructuredBuffer<uint32_t> inCompressedData,
    StructuredBuffer<zstdgpu_OffsetAndSize> inFramesRefs,
    RWStructuredBuffer<zstdgpu_Counters> inoutCounters,
    RWStructuredBuffer<uint32_t> inoutPerFrameBlockCountRAW,
    RWStructuredBuffer<uint32_t> inoutPerFrameBlockCountRLE,
    RWStructuredBuffer<uint32_t> inoutPerFrameBlockCountCMP,
    RWStructuredBuffer<uint32_t> inoutPerFrameBlockCountAll,
    RWStructuredBuffer<uint32_t> inoutRawBlockSizePrefix,
    RWStructuredBuffer<uint32_t> inoutRleBlockSizePrefix,
    RWStructuredBuffer<zstdgpu_OffsetAndSize> inoutBlocksRAWRefs,
    RWStructuredBuffer<zstdgpu_OffsetAndSize> inoutBlocksRLERefs,
    RWStructuredBuffer<zstdgpu_OffsetAndSize> inoutBlocksCMPRefs,
    RWStructuredBuffer<uint32_t> inoutBlockSizePrefix,
    RWStructuredBuffer<uint32_t> inoutGlobalBlockIndexPerRawBlock,
    RWStructuredBuffer<uint32_t> inoutGlobalBlockIndexPerRleBlock,
    RWStructuredBuffer<uint32_t> inoutGlobalBlockIndexPerCmpBlock,
    RWStructuredBuffer<uint32_t> inoutFrameStatus,
    uint32_t frameCount, uint32_t compressedBufferSizeInBytes, uint32_t countBlocksOnly,
    uint32_t i : SV_DispatchThreadId)
{
    zstdgpu_ParseFrames_SRT srt;
    srt.inCompressedData = inCompressedData;
    srt.inFramesRefs = inFramesRefs;
    srt.inoutCounters = inoutCounters;
    srt.inoutPerFrameBlockCountRAW = inoutPerFrameBlockCountRAW;
    srt.inoutPerFrameBlockCountRLE = inoutPerFrameBlockCountRLE;
    srt.inoutPerFrameBlockCountCMP = inoutPerFrameBlockCountCMP;
    srt.inoutPerFrameBlockCountAll = inoutPerFrameBlockCountAll;
    srt.inoutRawBlockSizePrefix = inoutRawBlockSizePrefix;
    srt.inoutRleBlockSizePrefix = inoutRleBlockSizePrefix;
    srt.inoutBlocksRAWRefs = inoutBlocksRAWRefs;
    srt.inoutBlocksRLERefs = inoutBlocksRLERefs;
    srt.inoutBlocksCMPRefs = inoutBlocksCMPRefs;
    srt.inoutBlockSizePrefix = inoutBlockSizePrefix;
    srt.inoutGlobalBlockIndexPerRawBlock = inoutGlobalBlockIndexPerRawBlock;
    srt.inoutGlobalBlockIndexPerRleBlock = inoutGlobalBlockIndexPerRleBlock;
    srt.inoutGlobalBlockIndexPerCmpBlock = inoutGlobalBlockIndexPerCmpBlock;
    srt.inoutFrameStatus = inoutFrameStatus;
    srt.frameCount = frameCount;
    srt.compressedBufferSizeInBytes = compressedBufferSizeInBytes;
    srt.countBlocksOnly = countBlocksOnly;
    zstdgpu_ShaderEntry_ParseFrames(srt, i);
}
