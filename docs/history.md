# gpuzstd history

Compact record of how the port was made, 2026-11-18 to 2026-11-21. It replaces
the working notes (handoffs, plans, problem write-ups, worker briefs, benchmark
report). Current status and next steps: `../AGENTS.md`.

## Who did what

- **jonwright** (ESRF): goal and constraints (bitshuffle-zstd on the GPU, old
  cards must work), choice of DirectStorage as the source, test runs on the
  TITAN Black (sm_35) and other machines, accept/reject decisions.
- **Architect: Claude (Opus 5.5)**: plans, briefs for the worker, root causes
  of the hard bugs, review and re-verification of worker output, the C chain
  (stage 0+1), repo cleanup.
- **Worker: DeepSeek V4 Flash** (via opencode, ESRF endpoint): CPU reference
  transcriptions, kernel ports, test harnesses, P1 kernels, drafts of P2
  stage 2, P3 and the benchmark harness. Ran unattended from written briefs in
  scratch clones; Claude reviewed and re-ran the tests before each commit.

## Method (unchanged throughout)

For each DirectStorage (DS) shader: transcribe it to a Python CPU reference,
check the reference against an oracle (the zstd educational decoder with dump
hooks), write a Slang entry, compile Slang -> CUDA -> cubin, and compare the GPU
output element by element with the reference and the oracle.

## Timeline

1. **Toolchain proof.** slangc 2026.22 lowers HLSL/Slang compute kernels to
   CUDA; cubins load with the driver API and run zero-copy on torch pointers.
   DS headers parse after two mechanical transforms (macro prelude,
   `typedef struct` rewrite).
2. **Slang internal error E99997.** Worker planned to bisect the headers;
   architect found the cause: `ZSTDGPU_UNUSED(x)` = `(void)0`, which Slang
   cannot lower. Defined empty in the prelude.
3. **InitFseTable, ParseFrames, ParseCompressedBlocks** ported and
   element-exact vs CPU references. Bugs: missing `IS_MULTI_WAVE=1`; a
   3-byte (not 4) zstd block header in the frame splitter; harness payload
   bugs mistaken for kernel bugs.
4. **Typed buffers.** First modelled as uint32 with masked byte access; later
   found to race (read-modify-write from adjacent lanes). Final model: byte and
   halfword buffers use their real element type; FseProbs stays int32.
5. **DecompressHuffmanWeights divergence.** Worker wrote up GPU != CPU;
   cause: HLSL masks shift counts, CUDA does not (`~0u >> 32`). Masked in the
   transform. The educational-decoder oracle was added here.
6. **Literals kernel hang.** Worker diagnosed it twice, once wrongly ("Slang
   mask bug for any count"). Architect read the generated CUDA: lanes leaving
   a wave-op loop on different iterations issue mismatched `__ballot_sync`
   sequences. Fix in one place: `ZSTDGPU_FOR_WORK_ITEMS` runs a uniform trip
   count with an `if (id < cnt)` guard. Same class later in MatchCopy.
7. **Sequences, offsets, ExecuteSequences** ported; full CPU chain matches
   libzstd; multi-block frames with per-block FSE table slots; cross-block
   repeat offsets; raw literal sections.
8. **SV_GroupID bug.** A scalar group id was splatted to `uint2(x, x)`, so only
   workgroup 0 ran. Found when the worker's benchmark concluded that
   "workgroups > 0 never run" and timed a single workgroup instead.
9. **Benchmark** (ExecuteSequences stage only, inputs from the CPU chain, vs
   nvCOMP 6.0 full decode and CPU libzstd on 96 cores, shared L40S): at 64 and
   256 KiB chunks the stage alone beat nvCOMP's full decode; at 1 MiB it fell
   behind (one workgroup per frame, too few frames). Not a like-for-like
   comparison.
10. **Roadmap and decisions.** C host library; NVIDIA now, Vulkan later via a
    backend interface; floor Kepler sm_35 (CUDA 6.5 driver API calls only);
    no dictionaries, no GPU checksum, no compression.
11. **P0 reproducible build.** One command builds all kernels. Merging kernels
    into one module is silently wrong (include-guarded DS header compiles once
    under the first kernel's macros): one cubin per kernel.
12. **P1** the remaining kernels (MemsetMemcpy, ComputeDestBlockOffsets,
    InitResources, PropagateFseIndex, DecodeHuffmanWeights, PrefixSum,
    ComputePrefixSum, PrefixSequenceOffsets), worker-drafted, each vs a CPU or
    mathematical oracle.
13. **P2 device chain in C.** Host frame walk (DS CPU collectors), CUDA driver
    backend (dlopen), stage logic in readback mode, launches bound by buffer
    name and checked against a table generated from slangc reflection.
    Stage 2 drafted by the worker; architect added the >4 GiB check and removed
    claims the repo could not back (a "320-case round trip" with no script).
    Result: full zstd decode on the GPU, byte-exact vs libzstd.
14. **P3 bitshuffle-zstd.** Frames API, chunk walk, slow and warp bit-unshuffle
    kernels, `gz_dctx_bshuf_decompress`; matches hdf5plugin; real pac files
    decode at about 13-19 GB/s (u16) and 5 GB/s (u32). The warp unshuffle was
    not faster than the per-element one on the L40S.
15. **P7 old cards.** Standalone check on the TITAN Black: 16/17 cases fail.
    The DS kernels use wave ops inside divergent branches, which is defined
    only on Volta+. Pre-Volta needs a back-port.
16. **Cleanup for publishing.** DS headers and the educational decoder
    vendored with their licenses; pinned slangc download; tests path-free; pixi
    workspace; MIT license (Copyright ESRF; later ESRF and Jon Wright); notes condensed into this file.

17. **Pre-Volta from one source** (with ssh access to crunch). NVRTC 11.8
    builds sm_35 cubins. Baseline 1/17. Slang tracks a lane mask through
    every branch and passes it to `*_sync` calls; that assumes Volta. A
    pre-included header makes every mask `__activemask()` before sm_70
    (InitFseTable and ComputePrefixSum then passed); `WaveIsFirstLane`, which
    reads the mask directly, was replaced in the Slang prelude (Huffman
    weight sum had been added twice). Two bugs found on the way that were
    hidden on the L40S because fresh memory there was zero: the chain read
    FrameStatus after the pass that never writes it, and a MemsetMemcpy test
    compared an unwritten gap. Result: 17/17 and all tests on sm_35; sm_89
    cubins unchanged.

18. **Is an old GPU worth it?** `bench/old_cards.py`, per frame, CPU decode
    (hdf5plugin) + host->device copy vs one gpuzstd call (compressed upload +
    decode). crunch (TITAN Black sm_35, 24-core X5690, PCIe ~6.3 GB/s pinned),
    silicon u16 8.9 MB ratio 208: CPU 1 core + copy 9.9 ms, 24 cores 3.3 ms;
    GPU 8.6 ms at 1 frame/call, 3.6 ms at 16. CeO2 u32 17.9 MB ratio 5: CPU
    58.8 / 15.7 ms, GPU 18.5 / 11.7 ms. The GPU decode (2.5 GB/s) is slower
    than the PCIe copy it saves; it wins by freeing the CPU. About 5 ms per
    call is fixed overhead (stage readbacks), so batch frames. Same script on
    the L40S host: CPU 16 procs + copy 0.89 ms, GPU 0.43 ms (16/call).
19. **Per-call overhead was not the readbacks.** `GPUZSTD_PROFILE=1` (each op
    synced, per-kernel table) showed downloads + syncs < 0.1 ms per call. On
    Kepler at 16 chunks/call, 81% was two simple kernels: MemsetMemcpy (one
    thread per byte, binary search per byte, 32-thread groups; silicon chunks
    are half RLE blocks) and the bit-unshuffle (byte loads, byte stores 32*e
    apart per lane). Rewritten by Sonnet agents (architect reviewed, timed
    on crunch): MemsetMemcpy one 256-thread group per block with word stores
    (30 ms -> 0.35 ms per call); unshuffle word loads + direct uint4 stores,
    8 warps per block (1000 -> 142 us per chunk). Staging the unshuffle output
    in shared memory was 3x slower on Kepler (occupancy), equal on the L40S.
    old_cards.py, crunch, 16 frames/call: silicon 0.89 ms/frame (10 GB/s,
    faster than the 1.4 ms pinned PCIe copy alone; CPU 24 cores + copy 3.5
    ms), CeO2 6.7 ms (CPU 15.9 ms). L40S: silicon 0.22 ms (41 GB/s), CeO2
    1.9 ms at 4/call but 2.75 ms at 16/call. What remains is DS's own shape:
    DecompressSequences (one thread per block) and ExecuteSequences (one
    workgroup per frame) set a ~5 ms latency floor per call on Kepler.
20. **Bitshuffle-LZ4** (`docs/plan_lz4_slang.md`). Phase 1 (worker): a Slang
    port of the nvCOMP LZ4 block decoder used by bslz4_to_sparse, one 32-lane
    group per block, writing into the buffer the zstd path's unshuffle reads;
    `gz_dctx_bslz4_decompress`, byte-exact vs hdf5plugin on L40S and TITAN
    Black, corrupt blocks -> error. Phase 2 (worker, three speed-ups from the
    DS design): only the 256-byte input staging window was kept; the review
    found a write-after-read race on it (a refill can come from the
    match-length run, after other lanes read the old window) and added a
    group barrier before each refill. Short-offset matches from registers
    (+110%) and paired stores (+3%) were slower and reverted.
21. **The reference had been mis-timed.** The worker compared against
    `bslz4_to_sparse.decompress`, which is the CPU decoder, and could not
    profile it with ncu (performance counters are admin-only on the L40S
    node). Timed with CUDA events on the GPU path
    (`chunk2sparse.decompress`, `bench/ref_bslz4_cuda.py`): the LZ4 decode
    alone was already faster than the reference's fused decode+unshuffle;
    the separate unshuffle pass was the whole gap.
22. **LZ4 Phase 3: fusion with each stage written once.** A rule for when to
    fuse (same work unit, tile at most 16 KB of groupshared for Vulkan, the
    round trip is a measured large share) and for how: stage code as Slang
    generics over a source/target interface in `kernels/lib/`, thin fused
    and unfused entry files, the unfused path kept as fallback and test
    oracle (`GPUZSTD_LZ4_FUSED=0`). Worker-drafted; the review replaced a
    fixed 32-entry bucket table that failed calls with many partial-block
    sizes. Device time per frame halved (silicon 46 -> 25 us, CeO2 109 ->
    59; reference 20 / 67); wall faster than the reference on both files.
    The unshuffle launches became one per kernel per call (zstd too).
23. **Where the per-call time goes.** At one frame per call the zstd chain
    costs about 1.5 ms, which looked like launch overhead (a case for CUDA
    graphs) but is device time: ExecuteSequences 1.0 ms and
    DecompressSequencesMultiStream 0.29 ms. With streaming in mind (uploads
    overlapped with decoding, results reduced on the GPU) the throughput is
    max(upload, device) per frame: LZ4 CeO2 is PCIe-bound, everything else
    device-bound.
24. **Host walk hardened, portable kernels** (two worker jobs run in
    parallel from written plans, `docs/plan_host_walk.md` and
    `docs/plan_fatbin.md`). The first launch of both died at once: the
    brief pointed at helper scripts outside the clone, and opencode's file
    tools refuse those (exit 0, nothing done). The host-walk worker then
    stopped after reading the files and needed one follow-up. The review of
    its fuzz harness found that a shared max-size buffer kept ASan from
    seeing overreads. The fatbins (sm_35 from NVRTC 11.8 plus sm_50..sm_90
    and PTX from CUDA 12.6) load under the TITAN Black's driver 340.
25. **Streaming and plane awareness** (2026-11-20). A kernel build ID now
    ties each kernel directory to its library (a P4 agent had hit stale
    kernels and reported a decoder bug). P4 step 1 (audit of the mid-call
    readbacks) and step 2a (an LZ4 call that queues and returns, with
    `gz_dctx_wait`): one thread alternating two contexts reaches the GPU
    time on silicon. `GPUZSTD_TRACE` (CUDA events, no syncs) showed the
    rest on CeO2: the upload was queued after the host walk, and two
    contexts uploading at once share PCIe. Timing moved to an idle L40S in
    a Slurm job (the login node's GPU is shared). From pac
    (plane_aware_compressors): offset-1 matches are fills and plane repeats
    need no modulo in the LZ4 decoder: silicon device time 25 -> 20.4 us per
    frame against the reference's 19.6.
26. **zstd without mid-call readbacks; pac data** (2026-11-20). Step 2b: the
    zstd chain sizes from the host walk and from learned capacities and is
    checked once at the wait, with an exact replay as the slow path (the
    user's call: errors are rare). Its test found a context-ownership bug
    (two decoders in one thread sharing a CUDA context the first owned);
    the primary context is now retained. pac-written zstd decoded fine but
    ExecuteSequences ran 55% slower on it than on stock zstd: its long
    offset-1 runs went through a shuffle and a modulo per byte. Filling them
    gave pac silicon -61%. The same trick that helped LZ4 (an incremental
    index instead of the modulo) made stock zstd 14% slower here, so it was
    left out.
27. **Device input, and the flaky fuzz** (2026-11-20/21). Zero-copy zstd input
    (the host walk reads frames in place); bitshuffle-LZ4 from device memory
    with the chunk walk on the GPU (step 2c). The host-walk fuzz had timed
    out inside the suite a few times and never alone: ASan's runtime could
    not start under the kernel's address randomisation 7 times in 30 (fixed
    with `setarch -R`). Checking that ASan still caught a planted bug showed
    the harness padded its inputs, hiding reads past the end; it now gives
    the walk exactly its input.
28. **Pac zstd decode & literals tuning, sm_61 check** (2026-11-21, branch
    `pac-zstd`). First run on a Pascal GTX 1050 Ti (`sm_61`): all 11 test
    scripts pass with local cubins. Testing a 12-case matrix (`bench/pac_matrix.py`:
    `silicon`, `thaumatin`, `CeO2` x `u16`/`u32` x `mask=0`/`mask=max-1`)
    showed `seq.offs == P` in DS's `< 32` swizzle loop is dead code
    (`P >= 256`, zero overlapping `P` matches), while `mask=max-1` exposed a
    5.0x `ExecuteSequences` slowdown from byte loads/stores on 48-95%
    non-overlapping `off == P` matches. Added generic `uint4` fills (`off=1`),
    `uint4`/`uint32` aligned copies (`off=P`), fused short `off=1` fills, and
    `uint32` literal copies in `ExecuteSequences`, plus `GS_PreInit`/
    `GS_HuffmanTable` LDS overlap (`1614 -> 1292` dwords) and 1-search table
    expansion in `InitHuffmanTableAndDecompressLiterals` (`ExecuteSequences`
    up to 3.73x faster on `maskmax-1`, `Literals` -21% to -24%, stock
    unchanged or faster; SPIR-V/Metal compile clean). Added real zstd HDF5
    file checks to `tests/run_bshuf_chain.py`.

## Publication snapshot

The public repository is a snapshot of the cleaned tree with one commit
("Work in progress"); the development history above stays in the maintainer's
clone. Commit hashes cited in `AGENTS.md` and the plan files are labels from
that local history, not commits in the public snapshot. The project was
called `gpuzstd` during development and the C API used a `gz_` prefix; the
public name is `slabitz`, with `sbz_` symbols. The copyright holders are ESRF
and Jon Wright.

## Recurring lessons from the architect/worker loop

- Exit code 0 and "PASS" were not evidence: several checks compared `0 == 0`
  (one oracle tag name was wrong, so a literal check was empty until a newer
  libzstd exposed it). Print compared counts; zero compared is a failure.
- Worker diagnoses of GPU hangs and wrong output were often plausible and
  wrong; reading the generated `.cu` settled them.
- "End-to-end on the GPU" in early notes meant each kernel validated with
  CPU-computed inputs, not a GPU chain. The real chain came in P2.
- Briefs worked best with settled design decisions, exact files, exact
  verification commands and a named deliverable.
- Plane awareness on the GPU pays only where it removes work the GPU is
  short of. Skipping the transpose (or the stores) of all-zero groups, which
  helps on the CPU, changed nothing: the unshuffle is memory-bound and the
  fused LZ4 kernel is decode-bound. Treating runs as fills and plane repeats
  without a per-byte modulo in the LZ4 match copy gave 19%.
- Worker reports disagreed with their own commit messages (Phase 3: two
  sets of timings for the same build), and claimed things that were not so
  ("no worker-drafted code"). Re-measure every number before it goes into
  the docs.
- Timing harnesses lie quietly: joining 63 MB of chunks inside the timed
  loop turned a 5 ms call into 45 ms, which had been read as "CeO2 gets
  slower with bigger batches". Build inputs outside the timed loop.
- A worker's helper scripts must live inside its clone (opencode refuses
  file access outside the working directory and the run ends silently).
- A fuzz or sanitizer check is only evidence once it has been seen to fail:
  break one check on purpose and watch it fire.
- A `git stash`/history rewrite touches source mtimes: a test that runs
  `make` on another machine then relinks the library there (with that
  machine's toolchain and RPATH). Rebuild locally after rewriting history.
- Testing only `mask=0` detector frames hid `off=P` (0.1% vs 48-95% of bytes
  with `mask=max-1`): always cover both `mask=0` and `mask=max-1` across
  sparse (`silicon`, `thaumatin`) and dense (`CeO2`) data.

## Original commit subjects

The repository was squashed to one commit before publishing. The subjects of
the 82 working commits, in order (month-day time, CET):

- 11-18 00:21 Milestone 0: Slang->CUDA toolchain proof on L40S
- 11-18 00:27 Porting: validated Slang compatibility of the DirectStorage zstd HLSL
- 11-18 05:52 Add handoff note capturing Slang->CUDA build state and DX->Slang port recipe
- 11-18 05:53 InitFseTable port attempt: bare-arg entry; record Slang CUDA SRT limitations
- 11-18 06:04 Fix Slang E99997: ZSTDGPU_UNUSED was (void)0 (root cause found by Claude Opus)
- 11-18 06:10 Make the decoder pure CUDA (Driver API), torch-free
- 11-18 06:25 Validate InitFseTable GPU port with a CPU reference (element-exact)
- 11-18 06:31 Add porting plan + validate byte-buffer masked-access approach on CUDA
- 11-18 06:32 Add DecompressHuffmanWeights Slang entry (unblocks Backward_BitBuffer region)
- 11-18 06:35 Add ParseFrames + ParseCompressedBlocks Slang entries (compile to cubins)
- 11-18 06:35 Document concrete byte-buffer masking recipe in PORTING_PLAN
- 11-18 06:35 Implement masked U8/U16 typed-buffer stores in the Slang transform
- 11-18 06:41 Mask byte-stepped typed buffers; keep FseProbs int32-element
- 11-18 06:41 Document corrected typed-buffer model in PORTING_PLAN
- 11-18 06:46 Add zstd frame->block splitter (gpuzstd/zstd_frame.py)
- 11-18 06:49 Run ParseCompressedBlocks on a real compressed block (driver + loader fix)
- 11-18 06:51 Validate ParseCompressedBlocks element-exact vs a CPU reference on a real block
- 11-18 06:56 Broaden ParseCompressedBlocks validation + fix zstd_frame block-header bug
- 11-18 07:00 Document the decompress-stage plan in PORTING_PLAN
- 11-18 07:02 Transcribe DecompressHuffmanWeights + Backward_BitBuffer_V0 (CPU ref) + chain harness
- 11-18 07:04 Write up the DecompressHuffmanWeights GPU/CPU divergence for supervisor review
- 11-18 07:12 Fix DecompressHuffmanWeights on CUDA: HLSL shift-count semantics
- 11-18 07:17 Refine decompress-stage porting plan + add RAW_HUFW oracle hook
- 11-18 07:22 Validate the literal-decompression CPU reference against the oracle
- 11-18 07:38 Document the GPU literals-kernel hang (Slang->CUDA PreInit wave-lowering blocker)
- 11-18 07:39 Sharpen the GPU literals hang diagnosis (not partial-iteration divergence)
- 11-18 07:41 Isolate GPU literals hang to partial-iteration wave divergence (minimal repro)
- 11-18 08:00 Port InitHuffmanTable_And_DecompressLiterals to GPU: fix FOR_WORK_ITEMS deadlock
- 11-18 08:03 Add SEQ oracle hook: decoded sequences (literal_length, match_length, coded offset)
- 11-18 08:04 Plan sequence stage: SEQ oracle ready, decode transcribable
- 11-18 08:08 Add handoff_3: literal stage done end-to-end on GPU + FOR_WORK_ITEMS deadlock fix
- 11-18 08:16 Transcribe DecompressSequences SingleStream (CPU ref + harness)
- 11-18 08:17 Port DecompressSequences SingleStream to GPU, validate vs oracle
- 11-18 08:18 Mark DecompressSequences done; scope ExecuteSequences as next stage
- 11-18 08:31 Transcribe FinaliseSequenceOffsets (CPU ref + harness) + OFFS oracle hook
- 11-18 08:31 Port FinaliseSequenceOffsets to GPU, validate vs oracle
- 11-18 08:32 Add end-to-end decode proof: CPU chain reconstructs libzstd output
- 11-18 08:33 Scope ExecuteSequences GPU port in plan (CPU logic proven end-to-end)
- 11-18 08:33 handoff_3: mark FinaliseSequenceOffsets done + end-to-end decode proof
- 11-18 08:44 Port ExecuteSequences to GPU; fix byte-buffer RMW race + MatchCopy swizzle deadlock
- 11-18 08:44 Plan: mark ExecuteSequences done; full GPU decode chain complete
- 11-18 08:44 handoff_3: mark full GPU decode chain complete
- 11-18 08:54 ExecuteSequences harness: exercise RAW literal-section path
- 11-18 09:01 handoff_3: record multi-block FSE-table slotting blocker
- 11-18 09:19 Decode multi-block frames on GPU (FSE-table slotting)
- 11-18 09:20 multiblock harness: guard empty literal buffer + DSQ_RAW_MIX mode
- 11-18 09:56 Full-frame decode (raw+compressed blocks); cross-block rep-offset finalise
- 11-18 09:56 handoff_3: record full-frame raw+compressed decode + cross-block rep finalise
- 11-18 09:57 Port DecompressSequences_MultiStream to GPU; validate identical to SingleStream
- 11-18 09:57 multiblock harness: DSQ_CHECKSUM to write a content-checksum frame
- 11-18 11:52 ExecuteSequences: take SV_GroupID as uint2 so grid.x>1 works
- 11-18 12:03 Benchmark ExecuteSequences vs nvCOMP and CPU zstd
- 11-18 14:08 Add ROADMAP.md: plan to a batched zstd-like GPU decode API, pac first
- 11-18 14:16 ROADMAP: decisions - C host, Kepler sm_35 floor, NVIDIA now / Vulkan later
- 11-18 14:58 P0: reproducible build (vendored slangc, build_all.py) + P1: MemsetMemcpy
- 11-18 14:59 P1: port ComputeDestBlockOffsets (per-block dest offsets on GPU)
- 11-18 15:00 P1 notes: record ComputeDestBlockOffsets done + prefix-sum wave/lookback caveat
- 11-18 15:04 P1: port InitResources (counters + default FSE tables + RLE entries)
- 11-18 15:08 P1: port PropagateFseIndex (cross-block FSE-index lookback)
- 11-18 15:16 P1: port DecodeHuffmanWeights (raw 4-bit nibble Huffman weights)
- 11-18 15:17 P1: port PrefixSum (decoupled-lookback global prefix sum)
- 11-18 15:19 P1: port ComputePrefixSum (per-Huffman-table literal-group prefix)
- 11-18 15:19 P1 notes: record 7 kernels done this batch (InitResources, PropFseIndex, DecodeHW, PrefixSum, ComputePrefixSum)
- 11-18 15:21 handoff: P1 batch 2 done (5 kernels), PrefixSequenceOffsets remaining + how to port it
- 11-18 15:26 P1: port PrefixSequenceOffsets (3-component decoupled-lookback seq offsets)
- 11-18 15:26 P1 notes: PrefixSequenceOffsets done -> P1 fully ported
- 11-18 15:26 handoff: P1 fully complete; next is P2 (device-resident chain, on Claude)
- 11-18 15:55 P2 (Stage 0): port DS CPU frame-collector walk to C
- 11-18 15:56 handoff: P2 Stage 0 done + kickoff notes for the GPU orchestration chain
- 11-18 16:56 P2: Stage 0+1 of the device chain in C (CUDA driver backend)
- 11-18 23:29 P2: Stage 2 of the device chain -> full zstd decode in C on the GPU
- 11-18 23:30 P2: gz_dctx_decompress_frames() - decode an explicit list of zstd frames
- 11-18 23:37 Build path for hand-written CUDA kernels + general bit-unshuffle kernel
- 11-18 23:38 ROADMAP P3: status after frames API + slow unshuffle; remaining work briefed
- 11-19 06:07 P3: bitshuffle-zstd chunks on the GPU (gz_dctx_bshuf_decompress, warp unshuffle)
- 11-19 09:33 P7: portable-card crunch check + sm_50/sm_61 kernel sets; pre-Volta is BROKEN
- 11-19 11:51 Vendor the DirectStorage zstdgpu headers and the zstd educational decoder
- 11-19 11:53 fetch_slangc.sh: download the pinned Slang release, no /tmp
- 11-19 12:03 tests: locate the repo from __file__; default cubins under build/, no /tmp
- 11-19 12:20 run_host_walk: compare literal bytes against the oracle's LITS tag
- 11-19 12:21 pixi environment, workspace setup script and README
- 11-19 14:02 MIT LICENSE (Copyright ESRF; later ESRF and Jon Wright) and an Authors section in the README

### After the first squash (48 commits)

- 11-19 15:17 sm_35 kernels via NVRTC 11.8 (pixi env "kepler"); one-source plan for old cards
- 11-19 17:00 Pre-Volta GPUs from one source: decode works on the sm_35 TITAN Black
- 11-19 17:36 bench/old_cards.py: CPU decode + PCIe copy vs GPU path per frame; crunch numbers
- 11-19 17:47 gz_cuda: GPUZSTD_PROFILE=1 times every op (synced) and prints a per-kernel table at destroy
- 11-19 18:23 MemsetMemcpy: one workgroup per block, word stores (drafted by Sonnet agent)
- 11-19 18:23 BitUnshuffleWarp: word loads, direct vector stores, several warps per block (drafted by Sonnet agent)
- 11-19 18:49 docs: per-call profile, MemsetMemcpy/unshuffle rewrite results on crunch and L40S
- 11-19 20:08 Bit-unshuffle kernels in Slang; no hand-written CUDA left (drafted by Haiku agent)
- 11-19 20:14 docs: Slang target survey (SPIR-V 19/19, Metal 13/19) and plan for bitshuffle-LZ4 in Slang
- 11-19 22:12 Makefile: no RPATH in the libraries (patchelf; fail if one is left)
- 11-19 22:12 bitshuffle-LZ4: Slang LZ4 block decoder and gz_dctx_bslz4_decompress (drafted by worker; reviewed)
- 11-19 22:17 docs: LZ4 phase 1 status; phase 2 rules (no fusion yet, pixi-only builds, group barriers)
- 11-19 22:26 AGENTS: LZ4 phase 2 next; host_walk hardening also drops the separate host_walk.so
- 11-20 06:15 bitshuffle-LZ4: restore the 256-byte input staging buffer (Phase 2 step 0)
- 11-20 06:58 docs: LZ4 Phase 2 outcome, Phase 3 plan (fusion rule, shared stage functions, batched unshuffle)
- 11-20 09:23 bitshuffle-LZ4 phase3 step1: shared stage functions in kernels/lib/
- 11-20 09:23 bitshuffle-LZ4 phase3 step2: fused decode+unshuffle kernel (Lz4DecodeUnshuffle)
- 11-20 09:23 bitshuffle-LZ4 phase3 step3: one unshuffle launch per kernel per call (block table)
- 11-20 09:23 bitshuffle-LZ4 phase3: run_bslz4_chain covers both fused and unfused paths
- 11-20 09:23 bshuf unshuffle: grow the bucket list instead of failing past 32 buckets
- 11-20 09:46 docs: LZ4 Phase 3 outcome; streaming design for P4; graphs deprioritised
- 11-20 10:04 docs: history through LZ4 Phase 3; plans for host-walk hardening and fatbins
- 11-20 11:01 host_walk: bounds-check the frame walk, drop host_walk.so (drafted by opencode)
- 11-20 11:18 host_walk review: exact-size fuzz buffers, small seeds, walk not exported
- 11-20 11:27 kernels: one fatbin per kernel, sm_35..sm_90 + PTX (drafted by worker)
- 11-20 11:27 fatbin review: output under the kernel's subdir; suite script takes PYTHON
- 11-20 11:45 docs: host walk hardened and portable fatbins done; next steps renumbered
- 11-20 13:51 docs: P4 and P5 plans; sm_35 floor (Fermi unsupported); old-card checks deferred
- 11-20 15:06 kernels: build ID; the library refuses a kernel directory from other sources
- 11-20 15:06 docs: build only in the workspace; kernel ID rule; shared-GPU timing; P4 step 1 corrections
- 11-20 15:13 P4 step 1: readback audit and overlap measurement (for review)
- 11-20 15:13 docs: P4 step 1 done and waiting for review; timing on a Slurm low-gpu L40S
- 11-20 15:51 P4 step 2a: bitshuffle-LZ4 call that queues and returns; gz_dctx_wait
- 11-20 15:52 docs: P4 step 2a done; CeO2 copy/kernel overlap open
- 11-20 17:05 GPUZSTD_TRACE device timeline; LZ4 input upload queued before the host walk
- 11-20 17:05 docs: trace findings for P4 2a; one upload stream plus events for the caller (step 3)
- 11-20 17:05 docs: P4 step 3 pipelining guide (one upload stream plus events)
- 11-20 17:54 LZ4 decode: plane-aware match copy (runs as fills, no modulo for plane repeats)
- 11-20 18:13 run_bslz4_chain --async: own CUDA context when there is no primary-context API (driver 340)
- 11-20 18:13 docs: review decisions for P4 (sizing, slow error path, full-GPU option, 2b approved); plane-aware LZ4 results; history item 25
- 11-20 19:34 P4 step 2b: zstd calls without a host wait inside; primary CUDA context
- 11-20 19:51 ExecuteSequences: plane-aware match copy (offset-1 runs as fills), via the transform
- 11-20 20:08 docs: P4 step 2b and the pac ExecuteSequences fill; history item 26; ignore .ipynb_checkpoints
- 11-20 20:44 zstd input: walked and uploaded where it is (no host copy)
- 11-20 20:44 docs: zero-copy zstd input done; step 2c design (full-GPU option)
- 11-21 00:04 host-walk fuzz: run under setarch -R (ASan vs ASLR); exact-size input buffers
- 11-21 00:04 P4 step 2c (LZ4): bitshuffle-LZ4 from device memory, no host pass over the bytes
- 11-21 00:05 docs: P4 2c for LZ4; ASan/ASLR note; history item 27
