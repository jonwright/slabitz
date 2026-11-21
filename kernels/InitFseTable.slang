// Slang entry for zstdgpu_ShaderEntry_InitFseTable (DirectStorage, MIT).
// Replicates Shaders/ZstdGpuInitFseTable.hlsl but replaces the D3D12
// RootSignature/ConstantBuffer SRT fill with a by-value SRT parameter.
#include "slang_compat.h"

// ZstdGpuInitFseTable.hlsl sets these (PC: WaveSize_Min=4 -> WaveCountMax=32 -> IS_MULTI_WAVE=1).
// Our entry calls zstdgpu_ShaderEntry_InitFseTable directly and does NOT include that
// .hlsl, so IS_MULTI_WAVE would otherwise default to 0 in zstdgpu_shaders.h and compile the
// single-wave path. Define them to reproduce the real PC/CUDA multi-wave behaviour.
#define kzstdgpu_WaveCountMax_InitFseTable (kzstdgpu_TgSizeX_InitFseTable / kzstdgpu_WaveSize_Min)
#define IS_MULTI_WAVE 1

#include "zstdgpu_shaders.h"
#include "srt_headers/zstdgpu_srt_structs.h"

// PC/CUDA: TgSizeX_InitFseTable=128, WaveSize_Min=4, IS_MULTI_WAVE=1.
// Total LDS = DEFAULT( MaxCount_FseProbs + MaxCount_FseElemsAllDigitBits )
//           + MULTI_WAVE( 3*WaveCountMax_InitFseTable + 3 + MaxCount_FseElems )
//           = (64 + 128) + (3*32 + 3 + 512) = 803 dwords.
groupshared uint32_t Lds[803];
#define ZSTDGPU_LDS Lds
#include "zstdgpu_lds_hlsl.h"

[shader("compute")]
[numthreads(kzstdgpu_TgSizeX_InitFseTable, 1, 1)]
void main(StructuredBuffer<int32_t> inFseProbs,
          RWStructuredBuffer<uint32_t> inoutFseElems,
          StructuredBuffer<zstdgpu_FseInfo> inFseInfos,
          StructuredBuffer<zstdgpu_Counters> inCounters,
          StructuredBuffer<uint32_t> inDispatchArgs,
          uint32_t tgOffset, uint32_t workItemCount, uint32_t tableType,
          uint32_t tableStartIndex, uint32_t tableDataStart, uint32_t tableDataCount,
          uint2 groupId2 : SV_GroupID, uint32_t i : SV_GroupThreadID)
{
    zstdgpu_InitFseTable_SRT srt;
    srt.inFseProbs     = inFseProbs;
    srt.inoutFseElems  = inoutFseElems;
    srt.inFseInfos     = inFseInfos;
    srt.inCounters     = inCounters;
    srt.inDispatchArgs = inDispatchArgs;
    srt.tgOffset          = tgOffset;
    srt.workItemCount     = workItemCount;
    srt.tableType         = tableType;
    srt.tableStartIndex   = tableStartIndex;
    srt.tableDataStart    = tableDataStart;
    srt.tableDataCount    = tableDataCount;

    // P4 step 2b: workItemCount ~0u means the table count is on the device (the
    // Counters field of this table type); the host then launches a worst-case grid.
    if (srt.workItemCount == 0xFFFFFFFFu)
    {
        const zstdgpu_Counters c = inCounters[0];
        srt.workItemCount = tableType == 0u ? c.FseHufW
                          : tableType == 1u ? c.FseLLen
                          : tableType == 2u ? c.FseOffs : c.FseMLen;
    }
    const uint32_t groupId = zstdgpu_ConvertTo32BitGroupId(groupId2, tgOffset);
    if (groupId >= srt.workItemCount) // whole group, before any barrier
        return;
    zstdgpu_ShaderEntry_InitFseTable(srt, groupId, i);
}
