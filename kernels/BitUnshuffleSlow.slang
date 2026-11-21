// Bitshuffle untranspose (bshuf_untrans_bit_elem), general and slow: one thread per
// element, any block size. Slang port of the former kernels/cuda/BitUnshuffleSlow.cu
// (same algorithm). The reference path for P3 and the fallback for blocks whose element
// count is not a multiple of 32.
//
// Phase 3 step 3: takes a block table (uint2 {inByteOffset, outByteOffset} per block)
// so the host launches the kernel once per (kernel, blockElems) group for ALL blocks of
// a call instead of once per chunk part. `fullBlocks` is the table entry count; every
// entry is a block of `blockElems` elements (the group's uniform block size), so a
// thread maps to (block b = j / blockElems, element i = j % blockElems) and reads the
// block's input/output base from the table. `elemSize` is the uniform element size.
//
// Input: the decoded blocks back to back: fullBlocks blocks of blockElems elements.
// Block b of blockElems elements of e bytes is 8e bit planes of blockElems/8 bytes; bit
// k of element i is bit (i % 8) of byte [k * blockElems/8 + i / 8] of that block (LSB
// first, k = 8 * byte + bit of the little-endian element).
// Output: the elements, in order (BshufOutput), starting at each block's outByteOffset.
// The raw tail is not handled here. Same per-block logic as the partial-block path of
// bslz4_to_sparse-dev bslz4_split.cu.
#include "slang_compat.h"

#include "zstdgpu_shaders.h"

[shader("compute")]
[numthreads(256, 1, 1)]
void main(StructuredBuffer<uint8_t> inUnCompressedFramesData,
          RWStructuredBuffer<uint8_t> inoutBshufOutput,
          StructuredBuffer<uint2> inBlockTable,
          uint32_t elemSize, uint32_t blockElems, uint32_t fullBlocks, uint32_t tgOffset,
          uint2 groupId2 : SV_GroupID, uint32_t t : SV_GroupThreadID)
{
    uint32_t inCount, inStride, outCount, outStride;
    inUnCompressedFramesData.GetDimensions(inCount, inStride);
    inoutBshufOutput.GetDimensions(outCount, outStride);

    const uint64_t j = uint64_t(zstdgpu_ConvertTo32BitGroupId(groupId2, tgOffset)) * 256u + t;
    const uint64_t total = uint64_t(fullBlocks) * blockElems;
    if (j >= total)
        return;
    const uint64_t b = j / uint64_t(blockElems);
    const uint64_t i = j - b * uint64_t(blockElems);
    const uint2 ref = inBlockTable[b];
    const uint64_t inByteOffset = ref.x;
    const uint64_t outByteOffset = ref.y;
    if (outByteOffset + uint64_t(blockElems) * elemSize > uint64_t(outCount) ||
        inByteOffset + uint64_t(blockElems) * elemSize > uint64_t(inCount))
        return;
    const uint64_t plane = blockElems / 8u;
    const uint32_t shift = uint32_t(i % 8u);
    const uint64_t p = inByteOffset + i / 8u;
    for (uint32_t m = 0; m < elemSize; ++m)
    {
        uint32_t v = 0;
        for (uint32_t tt = 0; tt < 8; ++tt)
        {
            const uint32_t bit =
                (uint32_t(inUnCompressedFramesData[p + (uint64_t(8u * m + tt)) * plane]) >> shift) &
                1u;
            v |= bit << tt;
        }
        inoutBshufOutput[outByteOffset + i * elemSize + m] = uint8_t(v);
    }
}
