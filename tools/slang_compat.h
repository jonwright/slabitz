// Slang compatibility prelude for the Microsoft DirectStorage zstd shaders.
//
// The DirectStorage HLSL was written for Microsoft's DXC compiler. Most of the
// source is plain HLSL that Slang accepts; the only friction is a set of
// DXC/MSVC-only warning-pragma and `globallycoherent` macros. Define those to
// benign forms BEFORE including the DirectStorage headers (they are all guarded
// by `#ifndef`, so these definitions win).
//
// Define __hlsl_dx_compiler so the shader GPU path (real HLSL resource types and
// intrinsics) is taken instead of the CPU C++ fallback shims, and set the LDS
// atomics/accessor names Slang resolves.
//
// Include this file before zstdgpu_structs.h / zstdgpu_shaders.h:
//     #include "slang_compat.h"
//     #include ".../zstdgpu_shaders.h"

#ifndef ZSTDGPU_SLANG_COMPAT_H
#define ZSTDGPU_SLANG_COMPAT_H

#ifndef __hlsl_dx_compiler
#   define __hlsl_dx_compiler 1
#endif

// DXC / MSVC warning-pragma plumbing -> no-ops under Slang.
#define ZSTDGPU_PRAGMA_GNUC(x)
#define ZSTDGPU_PRAGMA_MSVC(x)
#define ZSTDGPU_WARN_PUSH_DXC()
#define ZSTDGPU_WARN_STOP_DXC(w)
#define ZSTDGPU_WARN_POP_DXC()
#define ZSTDGPU_WARN_PUSH_MSVC()
#define ZSTDGPU_WARN_STOP_MSVC(w)
#define ZSTDGPU_WARN_POP_MSVC()
#define ZSTDGPU_WARN_DISABLE_DXC(w, stmt) stmt
#define ZSTDGPU_WARN_DISABLE_MSVC(w, stmt) stmt
#define ZSTDGPU_GLC
// Slang 2026.22 lowers a `(type)0` cast as "the default value of `type`"; for
// `void` there is none, so `ZSTDGPU_UNUSED(x)` == `(void)0` aborts with
// internal error E99997 ("missing case for getting IR default value"). Make it
// empty to drop the expression. (DirectStorage defines this only under
// `__hlsl_dx_compiler`, so it is safe to pre-define empty.)
#define ZSTDGPU_UNUSED(x)
#define ZSTDGPU_ENUM(e)             zstdgpu_##e
#define ZSTDGPU_ENUM_CONST(ec)      kzstdgpu_##ec
#define ZSTDGPU_INTENDED_SHIFT(e)   (e)
#define ZSTDGPU_INTENDED_MUL32(e)   (e)

// Typed Buffer<T>/RWBuffer<T> members are NOT available as readable/writable
// struct members on the CUDA target (error E36107, "using of '__ShapeBuffer'").
// StructuredBuffer/RWStructuredBuffer have the same [i] indexing here, so remap.
// All typed buffers become RWStructuredBuffer (even RO_TYPED ones). DS treats the
// RO ones as read-only anyway, and on CUDA the {data,count} ABI is identical; making
// them RWStructuredBuffer lets a single masked-load helper take any typed buffer
// (Slang does not implicitly convert RWStructuredBuffer<->StructuredBuffer).
#define ZSTDGPU_RO_TYPED_BUFFER(ShaderType, StorageType) RWStructuredBuffer<ShaderType>
#define ZSTDGPU_RW_TYPED_BUFFER(ShaderType, StorageType) RWStructuredBuffer<ShaderType>

// Reserved for the LDS accessor block; the shaders declare their own
// `groupshared` region and `#define ZSTDGPU_LDS <region>` before including
// zstdgpu_lds_hlsl.h (which Slang handles as-is).

// WaveIsFirstLane() for pre-Volta CUDA (sm_3x..sm_6x), one source for all GPUs.
// Slang's CUDA WaveIsFirstLane tests the lowest bit of the lane mask it tracks
// through branches. Those masks rely on Volta *_sync semantics: the "merge"
// ballot inside each side of an if/else, __ballot_sync(parentMask, true), waits
// for the whole parent mask on Volta, but on a lockstep pre-Volta warp it sees
// only that side's lanes, so each side gets its own "first lane" and per-wave
// atomics run twice (seen: Huffman weight sum doubled -> bitsMax + 1).
// tools/slang_cuda_prevolta.h makes every *_sync/_wave* call use __activemask()
// before Volta; this does the same for the one helper that reads the mask
// directly. On Volta+ the expression is Slang's own (tracked mask).
bool sbz_waveIsFirstLaneOf(uint mask)
{
    __target_switch
    {
    case cuda:
        __intrinsic_asm "((((__CUDA_ARCH__ < 700) ? __activemask() : (unsigned)($0)) & (0u - ((__CUDA_ARCH__ < 700) ? __activemask() : (unsigned)($0)))) == (1u << _getLaneId()))";
    default:
        return WaveIsFirstLane();
    }
}
bool sbz_WaveIsFirstLane() { return sbz_waveIsFirstLaneOf(WaveGetActiveMask()); }
#define WaveIsFirstLane() sbz_WaveIsFirstLane()

#endif // ZSTDGPU_SLANG_COMPAT_H
