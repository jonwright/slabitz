// Slang entry for zstdgpu_ShaderEntry_ParseCompressedBlocks (DirectStorage, MIT).
// Maps one compressed block per thread. This is the zstd block-parse pass: it reads
// the literal section header, FSE probability tables, Huffman weights and sequence
// section, and emits refs/probabilities/lookback state consumed by later passes.
//
// Compile check first; byte/halfword semantics for the TYPED_BUFFER members are
// applied later via the masked-access layer (PORTING_PLAN item 4).
#include "slang_compat.h"
#include "zstdgpu_shaders.h"
#include "srt_headers/zstdgpu_srt_structs.h"

[shader("compute")]
[numthreads(kzstdgpu_TgSizeX_ParseCompressedBlocks, 1, 1)]
void main(
    RWStructuredBuffer<zstdgpu_Counters> inoutCounters,
    RWStructuredBuffer<zstdgpu_FseInfo> inoutFseInfos,
    RWStructuredBuffer<zstdgpu_CompressedBlockData> inoutCompressedBlocks,
    RWStructuredBuffer<zstdgpu_OffsetAndSize> inoutHufRefs,
    RWStructuredBuffer<zstdgpu_LitStreamInfo> inoutLitRefs,
    RWStructuredBuffer<zstdgpu_OffsetAndSize> inoutSeqStreamToRef,
    RWStructuredBuffer<uint32_t> inoutSeqStreamToLLenFseId,
    RWStructuredBuffer<uint32_t> inoutSeqStreamToOffsFseId,
    RWStructuredBuffer<uint32_t> inoutSeqStreamToMLenFseId,
    RWStructuredBuffer<uint32_t> inoutSeqStreamToBlockId,
    RWStructuredBuffer<uint32_t> inoutBlockSizePrefix,
    RWStructuredBuffer<uint32_t> inoutPerFrameSeqStreamMinIdx,
    RWStructuredBuffer<uint32_t> inoutPerSeqStreamSeqStart,
    RWStructuredBuffer<uint32_t> inoutSeqCountPrefixLookback,
    RWStructuredBuffer<uint32_t> inoutBlockSeqCountPrefixLookback,
    RWStructuredBuffer<int32_t> inoutFseProbs,
    RWStructuredBuffer<uint8_t> inoutDecompressedHuffmanWeightCount,
    RWStructuredBuffer<uint32_t> inoutLitStreamCountPrefixLookback,
    RWStructuredBuffer<uint32_t> inoutHufWIdToHufLitId,
    RWStructuredBuffer<uint32_t> inoutHufLitIdToLitStreamId,
    RWStructuredBuffer<uint32_t> inoutHufLitIdToHufWId_DBG,
    RWStructuredBuffer<uint32_t> inoutHufLitCompactionLookback,
    StructuredBuffer<uint32_t> inCompressedData,
    StructuredBuffer<zstdgpu_OffsetAndSize> inBlocksCMPRefs,
    StructuredBuffer<uint32_t> inPerFrameBlockCountCMP,
    StructuredBuffer<uint32_t> inGlobalBlockIndexPerCmpBlock,
    StructuredBuffer<uint32_t> inDispatchArgs,
    uint32_t tgOffset, uint32_t workItemCount,
    uint32_t compressedBufferSizeInBytes, uint32_t frameCount,
    uint32_t compressedBlockCount,
    uint2 groupId2 : SV_GroupID, uint32_t i : SV_GroupThreadID)
{
    zstdgpu_ParseCompressedBlocks_SRT srt;
    srt.inoutCounters = inoutCounters;
    srt.inoutFseInfos = inoutFseInfos;
    srt.inoutCompressedBlocks = inoutCompressedBlocks;
    srt.inoutHufRefs = inoutHufRefs;
    srt.inoutLitRefs = inoutLitRefs;
    srt.inoutSeqStreamToRef = inoutSeqStreamToRef;
    srt.inoutSeqStreamToLLenFseId = inoutSeqStreamToLLenFseId;
    srt.inoutSeqStreamToOffsFseId = inoutSeqStreamToOffsFseId;
    srt.inoutSeqStreamToMLenFseId = inoutSeqStreamToMLenFseId;
    srt.inoutSeqStreamToBlockId = inoutSeqStreamToBlockId;
    srt.inoutBlockSizePrefix = inoutBlockSizePrefix;
    srt.inoutPerFrameSeqStreamMinIdx = inoutPerFrameSeqStreamMinIdx;
    srt.inoutPerSeqStreamSeqStart = inoutPerSeqStreamSeqStart;
    srt.inoutSeqCountPrefixLookback = inoutSeqCountPrefixLookback;
    srt.inoutBlockSeqCountPrefixLookback = inoutBlockSeqCountPrefixLookback;
    srt.inoutFseProbs = inoutFseProbs;
    srt.inoutDecompressedHuffmanWeightCount = inoutDecompressedHuffmanWeightCount;
    srt.inoutLitStreamCountPrefixLookback = inoutLitStreamCountPrefixLookback;
    srt.inoutHufWIdToHufLitId = inoutHufWIdToHufLitId;
    srt.inoutHufLitIdToLitStreamId = inoutHufLitIdToLitStreamId;
    srt.inoutHufLitIdToHufWId_DBG = inoutHufLitIdToHufWId_DBG;
    srt.inoutHufLitCompactionLookback = inoutHufLitCompactionLookback;
    srt.inCompressedData = inCompressedData;
    srt.inBlocksCMPRefs = inBlocksCMPRefs;
    srt.inPerFrameBlockCountCMP = inPerFrameBlockCountCMP;
    srt.inGlobalBlockIndexPerCmpBlock = inGlobalBlockIndexPerCmpBlock;
    srt.inDispatchArgs = inDispatchArgs;
    srt.tgOffset = tgOffset;
    srt.workItemCount = workItemCount;
    srt.compressedBufferSizeInBytes = compressedBufferSizeInBytes;
    srt.frameCount = frameCount;
    srt.compressedBlockCount = compressedBlockCount;

    const uint32_t threadId = i + zstdgpu_ConvertTo32BitGroupId(groupId2, tgOffset) *
                              kzstdgpu_TgSizeX_ParseCompressedBlocks;
    zstdgpu_ShaderEntry_ParseCompressedBlocks(srt, threadId);
}
