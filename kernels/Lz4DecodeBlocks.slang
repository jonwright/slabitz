// bitshuffle-LZ4 block decode: unfused entry. The decode loop is the shared
// stage function in kernels/lib/lz4_decode.slangh, instantiated with GlobalTarget
// (the decoded bytes go straight to global inoutDecoded); the existing
// BitUnshuffle kernels run over inoutDecoded afterwards, as for zstd. See the
// header for the licence (NVIDIA BSD-3) and the porting notes.
#include "slang_compat.h"
#include "zstdgpu_shaders.h"
#include "lib/lz4_decode.slangh"

struct Lz4BlockRef
{
    uint32_t inputOffs;  // byte offset of the LZ4 data in inCompressedData
    uint32_t compSize;   // compressed size (after the [BE u32 c] record)
    uint32_t decodedSize; // byte size the block must decode to
    uint32_t outOffs;    // byte offset of the decoded (bit-transposed) block in inoutDecoded
    uint32_t chunkIndex;
};

[shader("compute")]
[numthreads(32, 1, 1)]
void main(StructuredBuffer<uint8_t> inCompressedData,
          StructuredBuffer<Lz4BlockRef> inBlockTable,
          RWStructuredBuffer<uint8_t> inoutDecoded,
          RWStructuredBuffer<uint32_t> inoutStatus,
          uint32_t blockCount, uint32_t tgOffset,
          uint2 groupId2 : SV_GroupID, uint32_t t : SV_GroupThreadID)
{
    const uint32_t blockIdx = zstdgpu_ConvertTo32BitGroupId(groupId2, tgOffset);
    if (blockIdx >= blockCount)
        return;

    const Lz4BlockRef ref = inBlockTable[blockIdx];
    const uint32_t lane = t & (LZ4_LANES - 1u);

    GlobalTarget tgt;
    tgt.buf = inoutDecoded;
    tgt.base = ref.outOffs;

    const bool ok = lzDecodeBlock(inCompressedData, ref.inputOffs, ref.compSize, tgt, 0u,
                                  ref.decodedSize, lane);
    if (!ok)
    {
        if (lane == 0u)
            inoutStatus[ref.chunkIndex] = 1u;
    }
}
