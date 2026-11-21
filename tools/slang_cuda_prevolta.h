// Pre-Volta (sm_3x..sm_6x) wave semantics for Slang-generated CUDA.
//
// Included ahead of every Slang-generated kernel by the build (the prelude is
// include-guarded, so the generated file's own #include of it is then a no-op
// and the macros below apply to all generated code). Does nothing on Volta+.
//
// Slang emulates HLSL's "active lanes" by tracking a lane mask through every
// branch and passing it to __ballot_sync/__shfl_*_sync and the prelude's
// _wave* helpers. That tracking assumes Volta semantics: a *_sync call waits
// for every lane in its mask. Pre-Volta warps run in lockstep and a *_sync
// call whose mask names a lane that is not executing it is undefined; Slang's
// tracked mask often names such lanes (per-lane loops, early exits, the
// "reconverge" ballots after a break). On lockstep hardware the HLSL active
// set is exactly the set of lanes executing the instruction, i.e.
// __activemask(), so use that for every mask.
#ifndef SBZ_SLANG_CUDA_PREVOLTA_H
#define SBZ_SLANG_CUDA_PREVOLTA_H

#include "slang-cuda-prelude.h"

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)

#define __ballot_sync(m, p)        __ballot_sync(__activemask(), (p))
#define __any_sync(m, p)           __any_sync(__activemask(), (p))
#define __all_sync(m, p)           __all_sync(__activemask(), (p))
#define __shfl_sync(m, v, l)       __shfl_sync(__activemask(), (v), (l))
#define __shfl_up_sync(m, v, d)    __shfl_up_sync(__activemask(), (v), (d))
#define __shfl_down_sync(m, v, d)  __shfl_down_sync(__activemask(), (v), (d))
#define __shfl_xor_sync(m, v, l)   __shfl_xor_sync(__activemask(), (v), (l))

#define _waveSum(m, v)                  _waveSum(__activemask(), (v))
#define _waveMax(m, v)                  _waveMax(__activemask(), (v))
#define _wavePrefixSum(m, v)            _wavePrefixSum(__activemask(), (v))
#define _waveReadFirst(m, v)            _waveReadFirst(__activemask(), (v))
#define _waveShuffleMultiple(m, v, l)   _waveShuffleMultiple(__activemask(), (v), (l))

#endif
#endif
