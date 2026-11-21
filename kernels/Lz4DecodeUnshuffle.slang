// bitshuffle-LZ4 fused decode + unshuffle (Phase 3, step 2). One 32-lane group per
// bitshuffle block, as Lz4DecodeBlocks. The block is decoded into the file-scope
// groupshared tile (TileTarget, via the shared lzDecodeBlock stage function), then
// unshuffled (the shared bu_unshuffle_group step, with TilePlanes) straight into
// BshufOutput: no round trip through UnCompressedFramesData. Licensed as the
// shared decode stage function it uses (NVIDIA BSD-3), see kernels/lib/lz4_decode.slangh.
//
// Eligible blocks (the host sends only these): full blocks with
// blockElems*elemSize <= LZ4_FUSED_MAX_BLOCK (8192), elemSize in {1,2,4} and
// blockElems % 32 == 0. Every other block (the partial last block, elemSize 8,
// larger blocks) goes through Lz4DecodeBlocks + the separate unshuffle, as before;
// the host splits the block table and both launches can occur in one call.
//
// The fused path never writes UnCompressedFramesData. On a corrupt block
// (lzDecodeBlock returns false) lane 0 writes 1 to inoutStatus[ref.chunkIndex] and
// every lane returns uniformly; nothing is written to BshufOutput.
//
// Barriers: AllMemoryBarrierWithGroupSync inside lzDecodeBlock (match copies) and
// before the group-loop here (all 32 lanes done writing the tile before any lane
// reads it). The unshuffle loop has no wave ops (pure per-lane register transpose
// and stores), so the lanes may take different trip counts without diverging a wave
// synchronised region; keep the decode (which does) the same trip count on every
// lane, as lzDecodeBlock guarantees.
#include "slang_compat.h"
#include "zstdgpu_shaders.h"
#define LZ4_ENABLE_TILE 1
#include "lib/lz4_decode.slangh"
#include "lib/bit_unshuffle.slangh"

struct Lz4BlockRef
{
    uint32_t inputOffs;  // byte offset of the LZ4 data in inCompressedData
    uint32_t compSize;   // compressed size (after the [BE u32 c] record)
    uint32_t decodedSize; // byte size the block must decode to (blockElems * elemSize)
    uint32_t outOffs;    // byte offset of the untransposed block in inoutBshufOutput
    uint32_t chunkIndex;
};

[shader("compute")]
[numthreads(32, 1, 1)]
void main(StructuredBuffer<uint8_t> inCompressedData,
          StructuredBuffer<Lz4BlockRef> inBlockTable,
          RWStructuredBuffer<uint32_t> inoutStatus,
          RWStructuredBuffer<uint8_t> inoutBshufOutput,
          RWStructuredBuffer<uint32_t> inoutBshufOutput32,
          RWStructuredBuffer<uint4> inoutBshufOutput16,
          uint32_t blockCount, uint32_t elemSize, uint32_t tgOffset,
          uint2 groupId2 : SV_GroupID, uint32_t t : SV_GroupThreadID)
{
    const uint32_t blockIdx = zstdgpu_ConvertTo32BitGroupId(groupId2, tgOffset);
    if (blockIdx >= blockCount)
        return;

    const Lz4BlockRef ref = inBlockTable[blockIdx];
    const uint32_t lane = t & (LZ4_LANES - 1u);
    const uint32_t blockElems = ref.decodedSize / elemSize; // a full block, %32==0

    // decode into the groupshared tile (base 0)
    TileTarget tgt;
    tgt.base = 0u;
    const bool ok = lzDecodeBlock(inCompressedData, ref.inputOffs, ref.compSize, tgt, 0u,
                                  ref.decodedSize, lane);
    if (!ok)
    {
        if (lane == 0u)
            inoutStatus[ref.chunkIndex] = 1u;
        return;
    }

    // all 32 lanes have written their part of the tile
    AllMemoryBarrierWithGroupSync();

    // unshuffle the block's 32-element groups straight into BshufOutput. The tile
    // starts at the block's first plane, so the plane source reads from base 0.
    const uint32_t groups = blockElems / 32u;
    TilePlanes planes;
    planes.base = 0u;
    for (uint32_t grp = lane; grp < groups; grp += 32u)
    {
        const uint32_t g = grp;
        const uint32_t oByte = ref.outOffs + g * 32u * elemSize;
        bu_unshuffle_group(planes, 0u, g, blockElems, elemSize, oByte, inoutBshufOutput,
                           inoutBshufOutput32, inoutBshufOutput16);
    }
}
