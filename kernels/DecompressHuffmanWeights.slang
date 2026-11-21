// Slang entry for zstdgpu_ShaderEntry_DecompressHuffmanWeights (DirectStorage, MIT).
// Models Shaders/ZstdGpuDecompressHuffmanWeights.hlsl. One thread per Huffman-weight
// FSE stream (numthreads = kzstdgpu_TgSizeX_DecompressHuffmanWeights = 8 on PC).
//
// NOTE: this first pass is a *compile* check that the Backward_BitBuffer region lowers on
// CUDA. The byte-buffer outputs are currently StructuredBuffer<uint32_t> (the Buffer<T>
// remap); byte semantics come later via the masked-access layer (PORTING_PLAN item 4).
#include "slang_compat.h"
#include "zstdgpu_shaders.h"
#include "srt_headers/zstdgpu_srt_structs.h"

[shader("compute")]
[numthreads(kzstdgpu_TgSizeX_DecompressHuffmanWeights, 1, 1)]
void main(RWStructuredBuffer<uint8_t> inoutDecompressedHuffmanWeights,
          RWStructuredBuffer<uint8_t> inoutDecompressedHuffmanWeightCount,
          StructuredBuffer<zstdgpu_Counters> inCounters,
          ByteAddressBuffer inCompressedData,
          StructuredBuffer<zstdgpu_OffsetAndSize> inHufRefs,
          StructuredBuffer<zstdgpu_FseInfo> inFseInfos,
          StructuredBuffer<uint32_t> inFseElems,
          StructuredBuffer<uint32_t> inDispatchArgs,
          uint32_t tgOffset, uint32_t workItemCount,
          uint2 groupId2 : SV_GroupID, uint32_t i : SV_GroupThreadID)
{
    zstdgpu_DecompressHuffmanWeights_SRT srt;
    srt.inoutDecompressedHuffmanWeights = inoutDecompressedHuffmanWeights;
    srt.inoutDecompressedHuffmanWeightCount = inoutDecompressedHuffmanWeightCount;
    srt.inCounters = inCounters;
    srt.inCompressedData = inCompressedData;
    srt.inHufRefs = inHufRefs;
    srt.inFseInfos = inFseInfos;
    srt.inFseElems = inFseElems;
    srt.inDispatchArgs = inDispatchArgs;
    srt.tgOffset = tgOffset;
    srt.workItemCount = workItemCount;

    const uint32_t threadId = i + zstdgpu_ConvertTo32BitGroupId(groupId2, tgOffset) *
                              kzstdgpu_TgSizeX_DecompressHuffmanWeights;
    zstdgpu_ShaderEntry_DecompressHuffmanWeights(srt, threadId);
}
