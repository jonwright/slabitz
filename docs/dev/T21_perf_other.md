# T21: Other decoder performance items

Status: todo
Executor: Claude
Depends on: T03 (gate); best after T20
Branch: one per item, `t21a-...`, `t21b-...`

## Goal

Work through the remaining decoder costs one at a time. Each item is its own
PR, with a gate `--compare` table. An item that does not pay is recorded in
`docs/history.md` and dropped, and nothing of it is kept.

## Items (in the order to try)

1. **Small kernels.** About 13 small zstd kernels cost about 15 us per
   frame together (synced numbers, so an overestimate). Batch their
   launches, or make them read their counts on the device (DS's indirect
   design), so the launch grid is one worst-case launch.
   Target: at least 30% less total device time across them at 16 frames.
2. **DecompressSequences** runs one thread per compressed block. Measure
   the multi-stream variant (DecompressSequencesMultiStream, already
   present) against block counts, and pick the variant per call.
   Target: the gate's sequences kernel time at least 20% lower on CeO2.
3. **Unshuffle for zstd** (34 / 63 us per frame silicon / CeO2). Check it
   against the LZ4 fused-path unshuffle throughput, and close the gap if
   the zstd path uses the slower variant.
4. **Literals on CeO2** (71 us/frame). Profile Huffman decode
   occupancy, and the LDS that a0338bf left at 1292 dwords.
5. **Upload/decode overlap helper** (former AGENTS.md P6 and the P4
   trace). The pipelining guide in T13 shows one upload stream with
   events. Add `bench/stream_pipeline.py`, which reaches max(upload,
   device) per frame. Target: within 10% of max(upload, device) for CeO2
   LZ4 and zstd on the L40S.
6. **CUDA graphs** (lowest priority).
   - **Expected gain:** phase 3 measured the fixed cost per call as device
     time, not launches, so graphs would save at most about 0.2 ms per
     zstd call.
   - **Preconditions:**
     - no mid-call readback in exact mode (indirect-dispatch pattern);
     - stable buffer addresses;
     - graph calls optional, through dlopen (driver 410+);
     - plain launches on driver 340.
   - **Capture:** use stream capture around the existing launch code.

## Acceptance (per item)

1. The report `docs/dev/reports/T21<letter>.md` has a gate `--compare`
   table before and after.
2. The item's target is met with no other cell regressing beyond the gate
   thresholds. If the target is missed, the item is not merged; history
   records why.
3. `sbz test`, `sbz test-kernels` and the T06/T07 subsets pass.
   The kernel sets are rebuilt if kernels changed.

## Reviewer checks

Rerun the gate compare on an idle GPU. Check that the "after" numbers come
from the merged commit, not from an intermediate one.
