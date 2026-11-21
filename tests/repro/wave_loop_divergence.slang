// Minimal repro of the Slang->CUDA partial-iteration wave-intrinsic deadlock.
//
// Two entries:
//   main_orig    -- DS-style for(j=tid; j<n; j+=32) with a WaveActiveSum body.
//   main_uniform -- uniform trip count (end=align_up(n,32)) with an `if(j<n)` body guard.
//
// Both write sh[0] = sum over j of (w[j]>0 ? 1<<(w[j]-1) : 0), and outDbg[0] = sh[0].
// main_orig hangs when n is not a multiple of 32; main_uniform must finish for all n
// and match the reference.
groupshared uint32_t sh[1];

[shader("compute")]
[numthreads(32, 1, 1)]
void main_orig(RWStructuredBuffer<uint32_t> inW, RWStructuredBuffer<uint32_t> inN,
               RWStructuredBuffer<uint32_t> outDbg, uint32_t i : SV_GroupThreadID)
{
    uint n = inN[0];
    sh[0] = 0;
    GroupMemoryBarrierWithGroupSync();
    for (uint32_t j = i; j < n; j += 32) {
        uint w = inW[j];
        uint s = WaveActiveSum(w > 0 ? (1u << (w - 1)) : 0u);
        if (WaveIsFirstLane()) { sh[0] += s; }
    }
    GroupMemoryBarrierWithGroupSync();
    if (i == 0) { outDbg[0] = sh[0]; }
}

[shader("compute")]
[numthreads(32, 1, 1)]
void main_uniform(RWStructuredBuffer<uint32_t> inW, RWStructuredBuffer<uint32_t> inN,
                  RWStructuredBuffer<uint32_t> outDbg, uint32_t i : SV_GroupThreadID)
{
    uint n = inN[0];
    uint end = ((n + 31u) / 32u) * 32u;
    sh[0] = 0;
    GroupMemoryBarrierWithGroupSync();
    for (uint32_t j = i; j < end; j += 32) {
        if (j < n) {
            uint w = inW[j];
            uint s = WaveActiveSum(w > 0 ? (1u << (w - 1)) : 0u);
            if (WaveIsFirstLane()) { sh[0] += s; }
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (i == 0) { outDbg[0] = sh[0]; }
}

// Correct pattern: uniform bound, but the WAVE ops are NOT guarded -- all 32 lanes run them and
// stay converged. Only the data read is guarded (padded lanes read w=0), so a padded lane
// contributes 0 to the reduction.
[shader("compute")]
[numthreads(32, 1, 1)]
void main_correct(RWStructuredBuffer<uint32_t> inW, RWStructuredBuffer<uint32_t> inN,
                  RWStructuredBuffer<uint32_t> outDbg, uint32_t i : SV_GroupThreadID)
{
    uint n = inN[0];
    uint end = ((n + 31u) / 32u) * 32u;
    sh[0] = 0;
    GroupMemoryBarrierWithGroupSync();
    for (uint32_t j = i; j < end; j += 32) {
        uint w = (j < n) ? inW[j] : 0u;
        uint s = WaveActiveSum(w > 0 ? (1u << (w - 1)) : 0u);
        if (WaveIsFirstLane()) { sh[0] += s; }
    }
    GroupMemoryBarrierWithGroupSync();
    if (i == 0) { outDbg[0] = sh[0]; }
}
