// Bit-unshuffle of bitshuffle blocks, one warp per 32-element group, in registers.
// The transpose core and the per-lane step are the shared stage functions in
// kernels/lib/bit_unshuffle.slangh, instantiated with GlobalPlanes (the decoded
// bit-transposed bytes in global memory). See that header for the generalised-from
// shuffle_warp.cuh notes. Unfused entry: the fused Lz4DecodeUnshuffle entry (Phase 3
// step 2) reuses the same step with TilePlanes.
//
// Phase 3 step 3: takes a block table (uint2 {inByteOffset, outByteOffset} per block)
// so the host launches the kernel once per (kernel, blockElems) group for ALL blocks
// of a call instead of once per chunk part. `fullBlocks` is the table entry count;
// every entry is a block of `blockElems` elements (the group's uniform block size).
//
// Grid: S = ceil(groups/32) slices per block, groups = blockElems/32, so
// fullBlocks*S warps in all; each thread group holds 8 independent warps (256 threads)
// and the grid is ceil(fullBlocks*S / 8). Global warp w -> block b = w / S, slice
// sl = w % S; lane handles group g = sl*32 + lane. Warps past the end, or failing the
// bounds check, simply return (no shared memory, no barrier, no wave ops).
#include "slang_compat.h"
#include "zstdgpu_shaders.h"
#include "lib/bit_unshuffle.slangh"

[shader("compute")]
[numthreads(256, 1, 1)]
void main(StructuredBuffer<uint8_t> inUnCompressedFramesData,
          StructuredBuffer<uint32_t> inUnCompressedFramesData32,
          RWStructuredBuffer<uint8_t> inoutBshufOutput,
          RWStructuredBuffer<uint32_t> inoutBshufOutput32,
          RWStructuredBuffer<uint4> inoutBshufOutput16,
          StructuredBuffer<uint2> inBlockTable,
          uint32_t elemSize, uint32_t blockElems, uint32_t fullBlocks, uint32_t tgOffset,
          uint2 groupId2 : SV_GroupID, uint32_t t : SV_GroupThreadID)
{
    uint32_t inCount, inStride, outCount, outStride;
    inUnCompressedFramesData.GetDimensions(inCount, inStride);
    inoutBshufOutput.GetDimensions(outCount, outStride);

    const uint32_t lane = t & 31u;
    const uint32_t wib = t >> 5u;
    const uint32_t groups = blockElems / 32u;
    const uint32_t nslices = (groups + 31u) / 32u;
    const uint32_t wglob = zstdgpu_ConvertTo32BitGroupId(groupId2, tgOffset) * BU_WARPS + wib;
    const uint32_t b = wglob / nslices;
    const uint32_t sl = wglob % nslices;
    const uint32_t g = sl * 32u + lane;
    const uint32_t ng = (groups - sl * 32u) < 32u ? (groups - sl * 32u) : 32u;
    const uint32_t planeBytes = blockElems / 8u;
    const uint32_t nwords = 8u * elemSize;

    bool ok = b < fullBlocks;
    const uint32_t inByteOffset = ok ? inBlockTable[b].x : 0u;
    const uint32_t outByteOffset = ok ? inBlockTable[b].y : 0u;
    if (ok)
        ok = g < groups &&
             uint64_t(inByteOffset) + uint64_t(blockElems) * elemSize <= uint64_t(inCount) &&
             uint64_t(outByteOffset) + uint64_t(blockElems) * elemSize <= uint64_t(outCount);
    if (!ok)
        return;

    // byte offset of this block's first plane in the input
    const uint32_t wpByte = inByteOffset;
    const bool inAligned = ((wpByte | planeBytes) & 3u) == 0u;
    const uint32_t oByte = outByteOffset + g * 32u * elemSize;

    GlobalPlanes planes;
    planes.buf32 = inUnCompressedFramesData32;
    planes.buf8 = inUnCompressedFramesData;
    planes.aligned = inAligned;

    bu_unshuffle_group(planes, wpByte, g, blockElems, elemSize, oByte, inoutBshufOutput,
                       inoutBshufOutput32, inoutBshufOutput16);
}
