#!/usr/bin/env python3
"""Transform Microsoft DirectStorage HLSL headers so Slang accepts them.

Two mechanical fixes are needed (everything else is plain HLSL Slang already
accepts):

  1. C `typedef struct X { ... } X;` is not folded by Slang (the struct tag and
     the typedef name collide). Rewrite to `struct X { ... };` — Slang lets you
     use `X` directly as the type name.

  2. `zstdgpu_lds.h` uses `#pragma dxc push/pop`, which Slang does not have;
     drop those lines.

  3. HLSL masks shift counts to the operand width (x >> 32 == x >> 0); C++/CUDA
     does not (UB; on NVIDIA `~0u >> 32` == 0). Slang emits shifts verbatim, so
     DS code relying on HLSL semantics silently breaks on CUDA. Patch the known
     sites to mask the count explicitly (identical result under HLSL).
     See docs/history.md (DecompressHuffmanWeights divergence).

Usage:
    transform_ds_for_slang.py [--in DIR] [--out DIR]

The script copies the DirectStorage shader headers from --in (the zstdgpu/
directory) to --out (a scratch dir), then applies the transforms in place.
By default it transforms the live tree if --in is omitted (write to --out).

Run before feeding the tree to slangc with -I<--out> and -D__hlsl_dx_compiler
(or, better, #include tools/slang_compat.h first).
"""

import argparse, os, pathlib, re, shutil, sys

DS_VENDORED = pathlib.Path(__file__).resolve().parent.parent / "third_party" / "directstorage_zstdgpu"

# Headers needed to build a kernel entry (everything zstdgpu_shaders.h includes).
DS_HEADERS = [
    "zstdgpu_assert.h",
    "zstdgpu_shared_structs.h",
    "zstdgpu_structs.h",
    "zstdgpu_shaders.h",
    "zstdgpu_lds.h",
    "zstdgpu_lds_hlsl.h",
    "zstdgpu_lds_decl_base.h",
    "zstdgpu_lds_decl_size.h",
    "zstdgpu_lds_decl_undef.h",
    "srt_headers/zstdgpu_srt_structs.h",
]


def transform_struct_typedefs(text):
    names = re.findall(r"typedef\s+struct\s+(\w+)", text)
    text = re.sub(r"typedef\s+struct\s+(\w+)", r"struct \1", text, flags=re.M)
    for n in set(names):
        text = re.sub(r"\}\s*" + re.escape(n) + r"\s*;", "};", text)
    return text


# DS `ZSTDGPU_*_TYPED_BUFFER(ShaderType, StorageType)` declares a typed buffer whose element type
# is the StorageType on D3D12 (true byte/halfword loads+stores, no read-modify-write). Remapping it
# to `RWStructuredBuffer<ShaderType>` (uint32) and doing masked RMW stores races when several lanes
# write consecutive bytes of the same 32-bit word. Fix: give byte/halfword buffers their real element
# type so `buf[i]` is a true byte/halfword store. FseProbs is `(int32_t,int16_t)` but is int32-element
# indexed on GPU (validated), so it must stay int32. These exact instantiations are all that exist.
TYPED_BUFFER_TYPE_PATCHES = [
    ("ZSTDGPU_RW_TYPED_BUFFER(uint32_t, uint8_t)", "RWStructuredBuffer<uint8_t>"),
    ("ZSTDGPU_RO_TYPED_BUFFER(uint32_t, uint8_t)", "StructuredBuffer<uint8_t>"),
    ("ZSTDGPU_RW_TYPED_BUFFER(uint32_t, uint16_t)", "RWStructuredBuffer<uint16_t>"),
    ("ZSTDGPU_RO_TYPED_BUFFER(uint32_t, uint16_t)", "StructuredBuffer<uint16_t>"),
    # FseProbs: storage types says int16 but the DS indexes it as int32 elements (validated).
    ("ZSTDGPU_RW_TYPED_BUFFER(int32_t, int16_t)", "RWStructuredBuffer<int32_t>"),
    ("ZSTDGPU_RO_TYPED_BUFFER(int32_t, int16_t)", "StructuredBuffer<int32_t>"),
]


def patch_typed_buffer_types(text):
    for old, new in TYPED_BUFFER_TYPE_PATCHES:
        text = text.replace(old, new)
    return text


# (exact source text, replacement) -- each must match exactly once or we fail loudly.
HLSL_SHIFT_PATCHES = [
    # Backward_BitBuffer_V0_InitWithSegment: bitcnt == 0 when the segment end is dword-aligned;
    # HLSL gives ~0u, CUDA gave 0 -> garbled start of every backward stream.
    ("uint32_t bitmsk = ~0u >> (32u - bitcnt);",
     "uint32_t bitmsk = ~0u >> ((32u - bitcnt) & 31u);"),
]


def patch_hlsl_shift_semantics(text):
    for old, new in HLSL_SHIFT_PATCHES:
        n = text.count(old)
        if n != 1:
            raise SystemExit("HLSL shift patch: expected 1 match of %r, found %d" % (old, n))
        text = text.replace(old, new)
    return text


# Slang->CUDA lowering bug: a `ZSTDGPU_FOR_WORK_ITEMS` loop whose trip count is not a multiple of
# the thread count makes lanes leave the loop on different iterations. Slang then emits different
# sequences of `__ballot_sync`/`__syncthreads` for the lanes that stay vs. the lanes that break,
# so the warp barrier never matches and the warp deadlocks (see docs/history.md, literals kernel hang, and
# tests/repro/wave_loop_divergence.slang). Fix: give every lane the same trip count and guard the
# body with `if (<id> < <count>)`. On HLSL this is identical when <count> is wave-uniform (it is
# for every DS use; the PreInit counts come from tableId/LDS, plus thisGroupLiteralRemain per group).
#   NOTE for wave-op bodies: the `if` guard keeps lanes converged at the loop level so the wave
#   intrinsics never see a mismatched barrier; the reduction over the *active* lanes is still the
#   DS semantics because the guard is uniform within the last iteration that has any active lane.
FOR_WORK_ITEMS_PATCH = (
    "#   define ZSTDGPU_FOR_WORK_ITEMS(workItemId, workItemCount, groupThreadId, groupThreadCount)   \\\n"
    "        for (ZSTDGPU_WARN_DISABLE_DXC(-Wfor-redefinition, uint32_t workItemId = groupThreadId); workItemId < workItemCount; workItemId += groupThreadCount)",
    "#   define ZSTDGPU_FOR_WORK_ITEMS(workItemId, workItemCount, groupThreadId, groupThreadCount)   \\\n"
    "        for (uint32_t workItemId = groupThreadId, workItemId##force = ((uint32_t)(workItemCount) + (uint32_t)(groupThreadCount) - 1u) / (uint32_t)(groupThreadCount) * (uint32_t)(groupThreadCount); workItemId < workItemId##force; workItemId += groupThreadCount) \\\n"
    "            if (workItemId < (workItemCount))",
)


def patch_for_work_items(text):
    old, new = FOR_WORK_ITEMS_PATCH
    n = text.count(old)
    if n != 1:
        raise SystemExit("FOR_WORK_ITEMS patch: expected 1 match of the macro, found %d" % n)
    return text.replace(old, new)


MATCH_COPY_OLD = """static void zstdgpu_MatchCopy(RWStructuredBuffer<uint8_t> dstData,
                              ZSTDGPU_PARAM_INOUT(uint32_t) dstOfs,
                              zstdgpu_Sequence seq,
                              uint32_t dstEnd  //< NOTE(pamartis): this one is passed for safety clamp, could be ignored
                              )
{
    const uint32_t laneCount = WaveGetLaneCount();
    const uint32_t laneId = WaveGetLaneIndex();

    // When either path could handle the match copy, prefer this one since there is no modulo.
    ZSTDGPU_BRANCH if (seq.offs >= zstdgpu_MinU32(laneCount, seq.mlen))
    {
        uint32_t dstI = dstOfs + laneId;
        dstOfs += seq.mlen;
        dstEnd = zstdgpu_MinU32(dstEnd, dstOfs); //< NOTE(pamartis): could skip min

        ZSTDGPU_LOOP for (; dstI < dstEnd; dstI += laneCount)
        {
            zstdgpu_TypedStoreU8(dstData, dstI, dstData[dstI - seq.offs]);
        }
    }
    else // 'seq.offs < seq.mlen && seq.offs < laneCount':
    {
        uint32_t srcSym = 0;
        ZSTDGPU_BRANCH if (laneId < seq.offs)
        {
            srcSym = dstData[dstOfs + laneId - seq.offs];
        }

        const uint32_t copyLen = zstdgpu_MinU32(dstEnd - dstOfs, seq.mlen); //< NOTE(pamartis): could skip min

        // NOTE(jweinste): It is undefined what is returned upon reading from an inactive lane via WaveReadLaneAt()
        // (most current drivers seem to yield 0 for this case).
        // Hence, we must place the loop exit/break carefully.
        //
        // Consider this example:
        //      "ABC", offs=3, mlen=9 // initial data
        //      "ABCABCABCABC"        // expected output
        //
        //      Say WaveSize = 4:
        //          Store 0: stores ABCA = swizzle(ABC, {0,1,2,0}).
        //          Store 1: stores BCAB = swizzle(ABC, {1,2,0,1}).
        //          Store 2: only lane 0 can be active for the store, but the element it wants (C) is in lane 2.
        for (uint32_t copyId = laneId; /* mid-break in the loop */; copyId += laneCount)
        {
            //< NOTE(pamartis): We could replace 'mod' operation by Lemire'19 approach and precomputing up to 128 constants..
            const uint32_t swizzled = WaveReadLaneAt(srcSym, copyId % seq.offs);
            // Deactivate lanes only after doing WaveReadLaneAt.
            if (copyId >= copyLen)
            {
                break;
            }
            zstdgpu_TypedStoreU8(dstData, dstOfs + copyId, swizzled);
        }
        dstOfs += copyLen;
    }
}"""

MATCH_COPY_NEW = """static void zstdgpu_MatchCopy(RWStructuredBuffer<uint8_t> dstData,
                              RWStructuredBuffer<uint32_t> dstData32,
                              RWStructuredBuffer<uint4> dstData16,
                              ZSTDGPU_PARAM_INOUT(uint32_t) dstOfs,
                              zstdgpu_Sequence seq,
                              uint32_t dstEnd  //< NOTE(pamartis): this one is passed for safety clamp, could be ignored
                              )
{
    const uint32_t laneCount = WaveGetLaneCount();
    const uint32_t laneId = WaveGetLaneIndex();
    const uint32_t copyLen = (dstOfs < dstEnd) ? zstdgpu_MinU32(dstEnd - dstOfs, seq.mlen) : 0u;

    // Generic vectorized fill for offset-1 runs (e.g. empty bit-planes in bitshuffle):
    // one byte load on lane 0, then 16-byte aligned uint4 stores across the interior.
    ZSTDGPU_BRANCH if (seq.offs == 1u && copyLen > 0u && dstOfs >= 1u)
    {
        uint32_t s0 = 0u;
        if (laneId == 0u)
        {
            s0 = dstData[dstOfs - 1u];
        }
        const uint32_t sym = WaveReadLaneFirst(s0);
        ZSTDGPU_BRANCH if (copyLen >= 32u)
        {
            const uint32_t w = (sym & 0xFFu) * 0x01010101u;
            const uint4 v4 = uint4(w, w, w, w);
            const uint32_t head16 = (16u - (dstOfs & 15u)) & 15u;
            const uint32_t nVecs = (copyLen - head16) >> 4u;
            const uint32_t vecBase = (dstOfs + head16) >> 4u;
            const uint32_t tailStart = head16 + (nVecs << 4u);
            const uint32_t tail16 = copyLen - tailStart;
            if (laneId < head16)
            {
                zstdgpu_TypedStoreU8(dstData, dstOfs + laneId, sym);
            }
            ZSTDGPU_LOOP for (uint32_t v = laneId; v < nVecs; v += laneCount)
            {
                dstData16[vecBase + v] = v4;
            }
            if (laneId < tail16)
            {
                zstdgpu_TypedStoreU8(dstData, dstOfs + tailStart + laneId, sym);
            }
        }
        else
        {
            if (laneId < copyLen)
            {
                zstdgpu_TypedStoreU8(dstData, dstOfs + laneId, sym);
            }
        }
        dstOfs += copyLen;
    }
    // When either path could handle the match copy, prefer this one since there is no modulo.
    else ZSTDGPU_BRANCH if (seq.offs >= zstdgpu_MinU32(laneCount, seq.mlen))
    {
        // Generic vectorized non-overlapping aligned match copy (e.g. plane-to-plane
        // repeats where offset P is a multiple of 16 or 4): disjoint src and dst ranges.
        ZSTDGPU_BRANCH if (seq.offs >= copyLen && dstOfs >= seq.offs && copyLen >= 64u && (seq.offs & 15u) == 0u)
        {
            const uint32_t srcOfs = dstOfs - seq.offs;
            const uint32_t head16 = (16u - (dstOfs & 15u)) & 15u;
            const uint32_t nVecs = (copyLen - head16) >> 4u;
            const uint32_t dstVecBase = (dstOfs + head16) >> 4u;
            const uint32_t srcVecBase = (srcOfs + head16) >> 4u;
            const uint32_t tailStart = head16 + (nVecs << 4u);
            const uint32_t tail16 = copyLen - tailStart;
            if (laneId < head16)
            {
                zstdgpu_TypedStoreU8(dstData, dstOfs + laneId, dstData[srcOfs + laneId]);
            }
            ZSTDGPU_LOOP for (uint32_t v = laneId; v < nVecs; v += laneCount)
            {
                dstData16[dstVecBase + v] = dstData16[srcVecBase + v];
            }
            if (laneId < tail16)
            {
                zstdgpu_TypedStoreU8(dstData, dstOfs + tailStart + laneId, dstData[srcOfs + tailStart + laneId]);
            }
            dstOfs += seq.mlen;
        }
        else ZSTDGPU_BRANCH if (seq.offs >= copyLen && dstOfs >= seq.offs && copyLen >= 32u && (seq.offs & 3u) == 0u)
        {
            const uint32_t srcOfs = dstOfs - seq.offs;
            const uint32_t head4 = (4u - (dstOfs & 3u)) & 3u;
            const uint32_t nWords = (copyLen - head4) >> 2u;
            const uint32_t dstWordBase = (dstOfs + head4) >> 2u;
            const uint32_t srcWordBase = (srcOfs + head4) >> 2u;
            const uint32_t tailStart = head4 + (nWords << 2u);
            const uint32_t tail4 = copyLen - tailStart;
            if (laneId < head4)
            {
                zstdgpu_TypedStoreU8(dstData, dstOfs + laneId, dstData[srcOfs + laneId]);
            }
            ZSTDGPU_LOOP for (uint32_t w = laneId; w < nWords; w += laneCount)
            {
                dstData32[dstWordBase + w] = dstData32[srcWordBase + w];
            }
            if (laneId < tail4)
            {
                zstdgpu_TypedStoreU8(dstData, dstOfs + tailStart + laneId, dstData[srcOfs + tailStart + laneId]);
            }
            dstOfs += seq.mlen;
        }
        else
        {
            uint32_t dstI = dstOfs + laneId;
            dstOfs += seq.mlen;
            dstEnd = zstdgpu_MinU32(dstEnd, dstOfs); //< NOTE(pamartis): could skip min

            ZSTDGPU_LOOP for (; dstI < dstEnd; dstI += laneCount)
            {
                zstdgpu_TypedStoreU8(dstData, dstI, dstData[dstI - seq.offs]);
            }
        }
    }
    else // '1 < seq.offs < seq.mlen && seq.offs < laneCount':
    {
        uint32_t srcSym = 0;
        ZSTDGPU_BRANCH if (laneId < seq.offs)
        {
            srcSym = dstData[dstOfs + laneId - seq.offs];
        }

        // Pad the loop to a uniform trip count so WaveReadLaneAt always runs
        // with the whole wave converged (avoids pre-Volta/CUDA mask deadlock).
        const uint32_t copyEnd = ((copyLen + laneCount - 1) / laneCount) * laneCount;
        for (uint32_t copyId = laneId; copyId < copyEnd; copyId += laneCount)
        {
            const uint32_t swizzled = WaveReadLaneAt(srcSym, copyId % seq.offs);
            if (copyId < copyLen)
            {
                zstdgpu_TypedStoreU8(dstData, dstOfs + copyId, swizzled);
            }
        }
        dstOfs += copyLen;
    }
}"""

# Exact-once patches to wire inoutUnCompressedFramesData32/16 through ExecuteSequences into MatchCopy.
EXEC_VEC_PATCHES = [
    (MATCH_COPY_OLD, MATCH_COPY_NEW),
    ("static void zstdgpu_ExecuteSequence_FusedCopy(RWStructuredBuffer<uint8_t> dstData,\n"
     "                                              ZSTDGPU_PARAM_INOUT(uint32_t) dstOfs,",
     "static void zstdgpu_ExecuteSequence_FusedCopy(RWStructuredBuffer<uint8_t> dstData,\n"
     "                                              RWStructuredBuffer<uint32_t> dstData32,\n"
     "                                              RWStructuredBuffer<uint4> dstData16,\n"
     "                                              ZSTDGPU_PARAM_INOUT(uint32_t) dstOfs,"),
    ("        zstdgpu_MatchCopy(dstData, dstOfs, seq, dstEnd);",
     "        zstdgpu_MatchCopy(dstData, dstData32, dstData16, dstOfs, seq, dstEnd);"),
    ("static void zstdgpu_ExecuteSequencePair_FusedCopy(RWStructuredBuffer<uint8_t> dstData,\n"
     "                                                  ZSTDGPU_PARAM_INOUT(uint32_t) dstOfs,",
     "static void zstdgpu_ExecuteSequencePair_FusedCopy(RWStructuredBuffer<uint8_t> dstData,\n"
     "                                                  RWStructuredBuffer<uint32_t> dstData32,\n"
     "                                                  RWStructuredBuffer<uint4> dstData16,\n"
     "                                                  ZSTDGPU_PARAM_INOUT(uint32_t) dstOfs,"),
    ("        zstdgpu_ExecuteSequence_FusedCopy(dstData, dstOfs, litBuf, litOfs, seq0, dstEnd);\n"
     "        zstdgpu_ExecuteSequence_FusedCopy(dstData, dstOfs, litBuf, litOfs, seq1, dstEnd);",
     "        zstdgpu_ExecuteSequence_FusedCopy(dstData, dstData32, dstData16, dstOfs, litBuf, litOfs, seq0, dstEnd);\n"
     "        zstdgpu_ExecuteSequence_FusedCopy(dstData, dstData32, dstData16, dstOfs, litBuf, litOfs, seq1, dstEnd);"),
    ("            zstdgpu_ExecuteSequencePair_FusedCopy(srt.inoutUnCompressedFramesData, dstOfs, litBuf, litOfs, seq, seq1, dstEnd);",
     "            zstdgpu_ExecuteSequencePair_FusedCopy(srt.inoutUnCompressedFramesData, srt.inoutUnCompressedFramesData32, srt.inoutUnCompressedFramesData16, dstOfs, litBuf, litOfs, seq, seq1, dstEnd);"),
    ("            zstdgpu_ExecuteSequence_FusedCopy(srt.inoutUnCompressedFramesData, dstOfs, litBuf, litOfs, seq, dstEnd);",
     "            zstdgpu_ExecuteSequence_FusedCopy(srt.inoutUnCompressedFramesData, srt.inoutUnCompressedFramesData32, srt.inoutUnCompressedFramesData16, dstOfs, litBuf, litOfs, seq, dstEnd);"),
    ("                zstdgpu_MatchCopy(srt.inoutUnCompressedFramesData, blockByteCur, seq, blockByteEnd);",
     "                zstdgpu_MatchCopy(srt.inoutUnCompressedFramesData, srt.inoutUnCompressedFramesData32, srt.inoutUnCompressedFramesData16, blockByteCur, seq, blockByteEnd);"),
]


def patch_match_copy_swizzle(text):
    """Uniform the swizzle loop and add generic vectorized fills/copies to zstdgpu_MatchCopy."""
    for idx, (old, new) in enumerate(EXEC_VEC_PATCHES):
        n = text.count(old)
        if n != 1:
            raise SystemExit("EXEC_VEC patch %d: expected 1 match, found %d" % (idx, n))
        text = text.replace(old, new)
    return text


def patch_execute_sequences_srt(text):
    """Add 32-bit and 128-bit views of inoutUnCompressedFramesData to ExecuteSequences_SRT."""
    old = (
        "    StructuredBuffer<uint8_t>              inDecompressedLiterals;\n"
        "    RWStructuredBuffer<uint8_t>              inoutUnCompressedFramesData;\n"
        "    ZSTDGPU_RO_BUFFER(zstdgpu_Counters)                     inCounters;"
    )
    new = (
        "    StructuredBuffer<uint8_t>              inDecompressedLiterals;\n"
        "    RWStructuredBuffer<uint8_t>              inoutUnCompressedFramesData;\n"
        "    RWStructuredBuffer<uint32_t>             inoutUnCompressedFramesData32;\n"
        "    RWStructuredBuffer<uint4>                inoutUnCompressedFramesData16;\n"
        "    ZSTDGPU_RO_BUFFER(zstdgpu_Counters)                     inCounters;"
    )
    n = text.count(old)
    if n != 1:
        raise SystemExit("ExecuteSequences_SRT patch: expected 1 match, found %d" % n)
    return text.replace(old, new)


def strip_dxc_pragmas(text):
    return re.sub(r"^#pragma dxc.*$", "", text, flags=re.M)


def patch_byte_buffers(text):
    """Rewrite byte/halfword typed-buffer accesses for the StructuredBuffer<uint32>
    remap, where a bare `buffer[i] = v` would store a full dword (wrong semantics).

    The typed buffers are declared `ZSTDGPU_RW_TYPED_BUFFER(uint32_t, uint8_t|uint16_t)` and
    remapped (in tools/slang_compat.h) to `RWStructuredBuffer<uint32_t>`. The DS HLSL reaches
    them only through two helpers: the stores in `zstdgpu_TypedStoreU8` / `_U16`, and direct
    reads. We patch the store bodies to do masked byte/halfword stores and inject masked
    `zstdgpu_TypedLoadU8`/`_U16` helpers (direct-read rewrite is a separate step).
    """
    masked_u8 = """static void zstdgpu_TypedStoreU8(ZSTDGPU_RW_TYPED_BUFFER(uint32_t, uint8_t) inoutBuffer, uint32_t index, uint32_t value)
{
#ifdef __hlsl_dx_compiler
    uint32_t dw = index >> 2; uint32_t sh = (index & 3) * 8;
    inoutBuffer[dw] = (inoutBuffer[dw] & ~(0xFFu << sh)) | ((value & 0xFFu) << sh);
#else
    ZSTDGPU_ASSERT(value < (1u << 8u));
    inoutBuffer[index] = (uint8_t)value;
#endif
}"""
    masked_u16 = """static void zstdgpu_TypedStoreU16(ZSTDGPU_RW_TYPED_BUFFER(uint32_t, uint16_t) inoutBuffer, uint32_t index, uint32_t value)
{
#ifdef __hlsl_dx_compiler
    uint32_t dw = index >> 2; uint32_t sh = (index & 3) * 8;
    inoutBuffer[dw] = (inoutBuffer[dw] & ~(0xFFFFu << sh)) | ((value & 0xFFFFu) << sh);
#else
    ZSTDGPU_ASSERT(value < (1u << 16u));
    inoutBuffer[index] = (uint16_t)value;
#endif
}"""

    def find_func(txt, fn):
        start = txt.find("static void %s(" % fn)
        if start < 0:
            return None
        end = txt.find("\n}\n", start) + 3
        return start, end

    for fn, masked in (("zstdgpu_TypedStoreU8", masked_u8), ("zstdgpu_TypedStoreU16", masked_u16)):
        span = find_func(text, fn)
        if span:
            s, e = span
            text = text[:s] + masked + text[e:]

    loads = """
static uint32_t zstdgpu_TypedLoadU8(RWStructuredBuffer<uint32_t> inBuffer, uint32_t index)
{
    return (inBuffer[index >> 2] >> ((index & 3) * 8)) & 0xFFu;
}
static uint32_t zstdgpu_TypedLoadU16(RWStructuredBuffer<uint32_t> inBuffer, uint32_t index)
{
    return (inBuffer[index >> 2] >> ((index & 3) * 8)) & 0xFFFFu;
}
static int32_t zstdgpu_TypedLoadI16(RWStructuredBuffer<int32_t> inBuffer, uint32_t index)
{
    uint32_t v = (uint32_t)(inBuffer[index >> 1] >> ((index & 1) * 16)) & 0xFFFFu;
    return (int32_t)(uint32_t)((uint32_t)(v << 16) >> 16);
}
static void zstdgpu_TypedStoreI16(RWStructuredBuffer<int32_t> inoutBuffer, uint32_t index, int32_t value)
{
    uint32_t dw = index >> 1; uint32_t sh = (index & 1) * 16;
    uint32_t v = (uint32_t)value & 0xFFFFu;
    inoutBuffer[dw] = (inoutBuffer[dw] & ~(0xFFFFu << sh)) | (v << sh);
}
"""
    anchor = "static void zstdgpu_TypedStoreU8("
    if anchor in text:
        text = text.replace(anchor, loads.strip() + "\n\n" + anchor, 1)

    # Route direct read/write on the byte/halfword-typed buffers through the masked helpers.
    # Note: DS's `ZSTDGPU_*_TYPED_BUFFER(ShaderType, StorageType)` is `Buffer<ShaderType>` =
    # `RWBuffer<int32_t>` on GPU (see zstdgpu_structs.h), i.e. **int32 element-indexed**; the
    # StorageType (int8/16) is only the CPU-side packed representation. So int16 FseProbs is
    # NOT halfword-indexed on GPU (validated: InitFseTable C1/C2/C3 match a dword-element CPU
    # reference). Only the byte-*stepped* output buffers need masked access.
    I16 = []  # no masked store/load (kept as int32-element access)
    U8 = ["inoutDecompressedHuffmanWeightCount", "inoutDecompressedHuffmanWeights",
          "DecompressedLiterals", "dstData", "srcData", "litBuf",
          "HuffmanWeights", "HuffmanWeightCount"]
    name = re.compile("|".join(sorted((I16 + U8), key=len, reverse=True)))

    # Pass 1: write sites  `[q]name[IDX] = RHS;`  ->  TypedStore(width)([q]name, IDX, RHS);
    # Capture any leading qualifier (e.g. `srt.`) so the store/load sees the whole reference.
    def store_repl(m):
        q, buf, idx, rhs = m.group("q"), m.group("buf"), m.group("idx"), m.group("rhs")
        fn = ("zstdgpu_TypedStoreI16" if buf in I16 else "zstdgpu_TypedStoreU8")
        return "%s(%s%s, %s, %s);" % (fn, q or "", buf, idx.strip(), rhs.strip())
    text = re.sub(r"(?P<q>\b\w+\.)?(?P<buf>\b%s\b)\s*\[\s*(?P<idx>[^\]]+)\s*\]\s*=(?!=)\s*(?P<rhs>[^;]*);"
                  % name.pattern, store_repl, text)

    # Pass 2: all remaining `name[IDX]` are reads  ->  TypedLoad(width)(name, IDX);
    def load_repl(m):
        q, buf, idx = m.group("q"), m.group("buf"), m.group("idx")
        fn = ("zstdgpu_TypedLoadI16" if buf in I16 else "zstdgpu_TypedLoadU8")
        return "%s(%s%s, %s)" % (fn, q or "", buf, idx.strip())
    # skip function-call positions already produced: they read `name,` handled below; we only
    # rewrite bare `name[IDX]` (not preceded by a comma-arg where we already wrapped) -- safe.
    text = re.sub(r"(?P<q>\b\w+\.)?(?P<buf>\b%s\b)\s*\[\s*(?P<idx>[^\]]+)\s*\]" % name.pattern,
                  load_repl, text)

    # The store wrappers above produced TypedStore*(name, IDX, RHS) where a read `name[IDX]` in
    # RHS was already a load by pass 2 (pass 2 ran after). But pass 2 also rewrote the `name` of
    # the store call itself? No -- store call first arg is a bare `name` (no brackets), untouched.

    return text, 2



def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="src", default=None,
                    help="DirectStorage zstdgpu/ dir (default: $SLABITZ_DS_DIR, else the "
                         "vendored third_party/directstorage_zstdgpu)")
    ap.add_argument("--out", dest="dst", required=True, help="scratch output dir")
    a = ap.parse_args(argv)

    src = pathlib.Path(a.src or os.environ.get("SLABITZ_DS_DIR") or DS_VENDORED)
    dst = pathlib.Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "srt_headers").mkdir(exist_ok=True)

    for rel in DS_HEADERS:
        s = src / rel
        if not s.exists():
            raise SystemExit("missing DirectStorage header: %s (source dir %s)" % (rel, src))
        d = dst / rel
        text = s.read_text()
        if "typedef struct" in text:
            before = len(text)
            text = transform_struct_typedefs(text)
            print("  typedef->struct:", rel, "(%d typedef blobs)" % len(re.findall(r"struct \w+\s*\{", text)))
        if rel == "zstdgpu_lds.h":
            text = strip_dxc_pragmas(text)
            print("  stripped #pragma dxc:", rel)
        if rel == "zstdgpu_structs.h":
            text = patch_hlsl_shift_semantics(text)
            print("  patched HLSL shift-count semantics:", rel, "(%d sites)" % len(HLSL_SHIFT_PATCHES))
            text = patch_for_work_items(text)
            print("  patched FOR_WORK_ITEMS uniform trip count:", rel)
        if rel == "zstdgpu_shaders.h":
            text = patch_typed_buffer_types(text)
            print("  typed byte/halfword buffers as native elements:", rel)
            text = patch_match_copy_swizzle(text)
            print("  patched MATCH_COPY swizzle uniform trip count:", rel)
        if rel == "zstdgpu_shared_structs.h" or rel.startswith("srt_headers/"):
            text = patch_typed_buffer_types(text)
            print("  typed byte/halfword buffers as native elements:", rel)
            if rel == "srt_headers/zstdgpu_srt_structs.h":
                text = patch_execute_sequences_srt(text)
                print("  added u32/uint4 output views to ExecuteSequences_SRT:", rel)
        d.write_text(text)

    print("Wrote ported tree to", dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
