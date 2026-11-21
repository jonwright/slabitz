// Slang entry for zstdgpu_ShaderEntry_DecodeHuffmanWeights (DirectStorage, MIT).
// Replicates Shaders/ZstdGpuDecodeHuffmanWeights.hlsl with a by-value SRT.
// One thread per Huffman-weight table; unpacks 4-bit nibble weights.
#include "slang_compat.h"

#include "zstdgpu_shaders.h"
#include "srt_headers/zstdgpu_srt_structs.h"

[shader("compute")]
[numthreads(kzstdgpu_TgSizeX_DecodeHuffmanWeights, 1, 1)]
void main(RWStructuredBuffer<uint8_t> inoutDecompressedHuffmanWeights,
          RWStructuredBuffer<uint8_t> inoutDecompressedHuffmanWeightCount,
          StructuredBuffer<zstdgpu_Counters> inCounters,
          StructuredBuffer<uint32_t> inCompressedData,
          StructuredBuffer<zstdgpu_OffsetAndSize> inHufRefs,
          StructuredBuffer<uint32_t> inDispatchArgs,
          uint32_t tgOffset, uint32_t workItemCount, uint32_t compressedBufferSizeInBytes,
          uint2 groupId2 : SV_GroupID, uint32_t i : SV_GroupThreadID)
{
    zstdgpu_DecodeHuffmanWeights_SRT srt;
    srt.inoutDecompressedHuffmanWeights     = inoutDecompressedHuffmanWeights;
    srt.inoutDecompressedHuffmanWeightCount = inoutDecompressedHuffmanWeightCount;
    srt.inCounters                  = inCounters;
    srt.inCompressedData            = inCompressedData;
    srt.inHufRefs                   = inHufRefs;
    srt.inDispatchArgs              = inDispatchArgs;
    srt.tgOffset                    = tgOffset;
    srt.workItemCount               = workItemCount;
    srt.compressedBufferSizeInBytes = compressedBufferSizeInBytes;
    // fixup code for executeIndirectWorkaround
    if (int32_t(srt.workItemCount) < 0)
    {
        const uint32_t slot = ~srt.workItemCount;
        const uint32_t baseIdx = slot * kzstdgpu_DispatchSlot_StrideInUInt32;
        srt.workItemCount = inDispatchArgs[baseIdx + 1];
    }

    const uint32_t threadId = i + zstdgpu_ConvertTo32BitGroupId(groupId2, tgOffset) *
                              kzstdgpu_TgSizeX_DecodeHuffmanWeights;
    zstdgpu_ShaderEntry_DecodeHuffmanWeights(srt, threadId);
}
