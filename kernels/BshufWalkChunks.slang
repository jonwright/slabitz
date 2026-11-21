// P4 step 2c: the bitshuffle chunk walk on the device (no host pass over the
// compressed bytes). One thread per chunk. The caller states the geometry (element
// size, uncompressed bytes per chunk, bitshuffle block bytes), so the host knows
// every size, output offset and table position; only each LZ4 block's input
// offset and compressed size depend on the bytes, and this kernel finds them by
// walking the chunk's block records ([BE u32 size][data], sequential), with the
// same rules as the host walk (bshuf_walk_chunk in src/sbz_chain.c):
//
//   header: BE u64 total bytes == totalBytes, BE u32 block bytes == blockBytes;
//   each record: 4 bytes inside the chunk, size != 0, data inside the chunk;
//   then tailBytes raw bytes inside the chunk, copied untransposed to the output.
//
// Block k of chunk c goes to the fused table (full blocks, when `fused`) at
// c*nfull + k, outOffs in BshufOutput space; every other block to the unfused
// table, at c (the partial last block when fused) or c*nblk + k, outOffs in
// UnCompressedFramesData space. On any mismatch the chunk's status is set (2)
// and its remaining entries get size 0 at the chunk start, which keeps the
// decoders in bounds and makes them report the block corrupt; the host then
// replays the call exactly from a host copy (the slow path).
//
// Cost: the records form a chain (each position depends on the previous size),
// so a walk pays about one memory latency per record: 2026-11-20, L40S, 16 chunks
// per call: 412 us (silicon, ~1100 records of ~130 B per chunk) and 1255 us (CeO2,
// ~2200 records of ~1.8 kB). It is per call, not per chunk (chunks walk in
// parallel), and uses one warp, so it overlaps another context's decode. Tried:
// a 32-lane group per chunk staging 4 kB windows in groupshared (CeO2 2.3x
// slower), and every lane following the chain while touching the lines ahead (no
// gain: the touch is consumed at once, so the warp waits for it too). A real
// prefetch (PTX prefetch.global.L2, CUDA only) is the remaining idea.
#include "slang_compat.h"
#include "zstdgpu_shaders.h"

struct Lz4BlockRef
{
    uint32_t inputOffs;
    uint32_t compSize;
    uint32_t decodedSize;
    uint32_t outOffs;
    uint32_t chunkIndex;
};

uint32_t bswBe32(StructuredBuffer<uint8_t> b, uint32_t at)
{
    return (uint32_t(b[at]) << 24u) | (uint32_t(b[at + 1u]) << 16u) |
           (uint32_t(b[at + 2u]) << 8u) | uint32_t(b[at + 3u]);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void main(StructuredBuffer<uint8_t> inCompressedData,
          StructuredBuffer<uint2> inChunkSpans,
          RWStructuredBuffer<Lz4BlockRef> inoutBlockTable,
          RWStructuredBuffer<Lz4BlockRef> inoutBlockTableFused,
          RWStructuredBuffer<uint32_t> inoutStatus,
          RWStructuredBuffer<uint8_t> inoutBshufOutput,
          uint32_t chunkCount, uint32_t totalBytes, uint32_t blockBytes, uint32_t nfull,
          uint32_t lastBytes, uint32_t tailBytes, uint32_t fused, uint32_t chunkOutBytes,
          uint32_t frameBytes, uint32_t tgOffset,
          uint2 groupId2 : SV_GroupID, uint32_t t : SV_GroupThreadID)
{
    const uint32_t c = zstdgpu_ConvertTo32BitGroupId(groupId2, tgOffset) * 32u + t;
    if (c >= chunkCount)
        return;
    const uint2 sp = inChunkSpans[c];
    const uint32_t base = sp.x;
    const uint32_t clen = sp.y;
    const uint32_t nblk = nfull + (lastBytes != 0u ? 1u : 0u);

    bool ok = clen >= 12u;
    if (ok)
        ok = bswBe32(inCompressedData, base) == 0u &&
             bswBe32(inCompressedData, base + 4u) == totalBytes &&
             bswBe32(inCompressedData, base + 8u) == blockBytes;

    uint32_t pos = 12u;
    for (uint32_t k = 0u; k < nblk; ++k)
    {
        uint32_t cs = 0u;
        uint32_t inOffs = base;
        if (ok)
        {
            if (clen - pos < 4u)
                ok = false;
            else
            {
                cs = bswBe32(inCompressedData, base + pos);
                if (cs == 0u || clen - pos - 4u < cs)
                    ok = false;
                else
                {
                    inOffs = base + pos + 4u;
                    pos += 4u + cs;
                }
            }
        }
        if (!ok)
        {
            cs = 0u;
            inOffs = base;
        }
        const bool isFull = k < nfull;
        Lz4BlockRef ref;
        ref.inputOffs = inOffs;
        ref.compSize = cs;
        ref.decodedSize = isFull ? blockBytes : lastBytes;
        ref.chunkIndex = c;
        if (fused != 0u && isFull)
        {
            ref.outOffs = c * chunkOutBytes + k * blockBytes;
            inoutBlockTableFused[c * nfull + k] = ref;
        }
        else
        {
            ref.outOffs = c * frameBytes + k * blockBytes;
            inoutBlockTable[fused != 0u ? c : c * nblk + k] = ref;
        }
    }
    if (ok && clen - pos < tailBytes)
        ok = false;
    if (ok)
    {
        const uint32_t o = c * chunkOutBytes + frameBytes;
        for (uint32_t i = 0u; i < tailBytes; ++i)
            inoutBshufOutput[o + i] = inCompressedData[base + pos + i];
    }
    if (!ok)
        inoutStatus[c] = 2u;
}
