// Slang entry for zstdgpu_ShaderEntry_InitHuffmanTable_And_DecompressLiterals (DirectStorage, MIT).
// Models Shaders/ZstdGpuInitHuffmanTableAndDecompressLiterals.hlsl. PC/CUDA:
// TgSizeX_DecompressLiterals=32 (one wave), StreamsPerGroup_DecompressLiterals=32.
// Replaces the D3D12 RootSignature/ConstantBuffer SRT fill with a by-value SRT parameter.
#include "slang_compat.h"

#include "zstdgpu_shaders.h"
#include "srt_headers/zstdgpu_srt_structs.h"

// LDS = CodeAndSymbol(256) + RankIndex(12) + max(PreInit(322), HuffmanTable(1<<10)) = 1292 dwords.
groupshared uint32_t Lds[1292];
#define ZSTDGPU_LDS Lds
#include "zstdgpu_lds_hlsl.h"

[shader("compute")]
[numthreads(kzstdgpu_TgSizeX_DecompressLiterals, 1, 1)]
void main(StructuredBuffer<uint32_t> inLitGroupEndPerHuffmanTable,
          StructuredBuffer<zstdgpu_Counters> inCounters,
          StructuredBuffer<zstdgpu_LitStreamInfo> inLitRefs,
          ByteAddressBuffer inCompressedData,
          StructuredBuffer<uint32_t> inHufWIdToHufLitId,
          StructuredBuffer<uint32_t> inHufLitIdToLitStreamId,
          RWStructuredBuffer<uint8_t> inoutDecompressedLiterals,
          RWStructuredBuffer<uint32_t> inoutDecompressedLiterals_Dwords,
          StructuredBuffer<uint8_t> inDecompressedHuffmanWeights,
          StructuredBuffer<uint8_t> inDecompressedHuffmanWeightCount,
          StructuredBuffer<uint32_t> inDispatchArgs,
          uint32_t tgOffset, uint32_t workItemCount, uint32_t streamsPerGroup,
          uint32_t litCapacity,
          uint2 groupId2 : SV_GroupID, uint32_t i : SV_GroupThreadID)
{
    zstdgpu_InitHuffmanTableAndDecompressLiterals_SRT srt;
    srt.inLitGroupEndPerHuffmanTable = inLitGroupEndPerHuffmanTable;
    srt.inCounters                   = inCounters;
    srt.inLitRefs                    = inLitRefs;
    srt.inCompressedData             = inCompressedData;
    srt.inHufWIdToHufLitId           = inHufWIdToHufLitId;
    srt.inHufLitIdToLitStreamId      = inHufLitIdToLitStreamId;
    srt.inoutDecompressedLiterals    = inoutDecompressedLiterals;
    srt.inoutDecompressedLiterals_Dwords = inoutDecompressedLiterals_Dwords;
    srt.inDecompressedHuffmanWeights     = inDecompressedHuffmanWeights;
    srt.inDecompressedHuffmanWeightCount = inDecompressedHuffmanWeightCount;
    srt.inDispatchArgs               = inDispatchArgs;
    srt.tgOffset                     = tgOffset;
    srt.workItemCount                = workItemCount;
    srt.streamsPerGroup              = streamsPerGroup;

    // P4 step 2b: the host sizes the literal and sequence buffers from per-context
    // capacities instead of reading the counts back. If this call's real counts
    // exceed them, every kernel touching those buffers does nothing, and the host
    // finds out at sbz_dctx_wait (one readback) and redoes the call exactly.
    const uint32_t groupId = zstdgpu_ConvertTo32BitGroupId(groupId2, tgOffset);
    // the host may launch more groups than DecompressLiteralsGroups (a bound)
    if (groupId >= inCounters[0].DecompressLiteralsGroups ||
        inCounters[0].HUF_Streams_DecodedBytes > litCapacity)
        return; // whole group, before any barrier
    zstdgpu_ShaderEntry_InitHuffmanTable_And_DecompressLiterals(
        srt, groupId, i, kzstdgpu_TgSizeX_DecompressLiterals);
}
