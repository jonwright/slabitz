# T22 Slang target survey (compile only)

- slangc: 2026.22
- worktree HEAD: `683fa3d7c252c4649255a57baabe452d1ba4d7bc`
- targets: spirv, metal, hlsl, cpp
- flags: `-profile sm_6_0 -I tools -I <transformed DS tree> -entry main -stage compute` (the CUDA build's flags, only `-target` changed)
- rows are the entries of `src/sbz_kernels_gen.h` in order; cell is `ok` or the first error line (truncated to 160 chars).

| entry | spirv | metal | hlsl | cpp |
|---|---|---|---|---|
| MemsetMemcpy | ok | ok | ok | ok |
| ComputeDestBlockOffsets | ok | ok | ok | ok |
| InitResources | ok | ok | ok | ok |
| PropagateFseIndex | ok | ok | ok | error[E36107]: unavailable features in entry point |
| DecodeHuffmanWeights | ok | ok | ok | ok |
| PrefixSum | ok | error[E36107]: unavailable features in entry point | ok | error[E36107]: unavailable features in entry point |
| ComputePrefixSum | ok | error[E36107]: unavailable features in entry point | ok | error[E36107]: unavailable features in entry point |
| PrefixSequenceOffsets | ok | ok | ok | error[E36107]: unavailable features in entry point |
| ParseFrames | ok | ok | ok | error[E36107]: unavailable features in entry point |
| ParseCompressedBlocks | ok | error[E36107]: unavailable features in entry point | ok | error[E36107]: unavailable features in entry point |
| InitFseTable | ok | error[E36107]: unavailable features in entry point | ok | error[E36107]: unavailable features in entry point |
| DecompressHuffmanWeights | ok | ok | ok | ok |
| InitHuffmanTableAndDecompressLiterals | ok | ok | ok | error[E36107]: unavailable features in entry point |
| DecompressSequences | ok | ok | ok | ok |
| DecompressSequencesMultiStream | ok | ok | ok | ok |
| FinaliseSequenceOffsets | ok | ok | ok | ok |
| CheckSequenceOffsets | ok | ok | ok | error[E36107]: unavailable features in entry point |
| ExecuteSequences | ok | ok | ok | error[E36107]: unavailable features in entry point |
| BitUnshuffleSlow | ok | error[E36107]: unavailable features in entry point | ok | ok |
| BitUnshuffleWarp | ok | error[E36107]: unavailable features in entry point | ok | ok |
| Lz4DecodeBlocks | ok | ok | ok | error[E36107]: unavailable features in entry point |
| Lz4DecodeUnshuffle | ok | ok | ok | error[E36107]: unavailable features in entry point |
| BshufWalkChunks | ok | ok | ok | ok |
| **totals (ok / 23)** | 23/23 | 17/23 | 23/23 | 11/23 |

