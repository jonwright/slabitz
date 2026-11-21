# P4 step 1: readback and sync audit

Goal of P4: the caller uploads batch n+1 while batch n decodes, keeps decoded
frames on the device, and copies back only small results — throughput
`max(upload, device)` per frame. That needs a decode call to queue its work on
the caller's stream and **return without waiting for the GPU**. Today every
`sbz_dctx_*` call is synchronous: it enqueues uploads + kernels on the context's
single stream and `sync`s before returning, and it reads several counters back to
the host to size buffers.

This is step 1: list every host readback / sync inside a call, say what each is
for and how to remove it, and measure. It stops here for review — it decides the
scope of step 2.

## 1. What a call does today

All line numbers are `src/sbz_chain.c`. The chain is a port of DS zstdgpu.cpp in
"readback mode" (`sbz_chain.h`): CUDA has no ExecuteIndirect, so the host reads
`Counters` back after each stage and does what `ZstdGpuUpdateDispatchArgs.hlsl`
does (grid sizes) and `zstdgpu_ResourceInfo_Stage_N_InitSize` does (buffer sizes).

Per call (plain zstd, `sbz_dctx_decompress` → `host_prepare` + `stage0` +
`stage1` + `stage2`):

| # | site | what is transferred back to the host | used for | replacement |
|---|------|--------------------------------------|----------|-------------|
| 1 | `stage0` line 581 `read_counters` | `Counters` (88 B) | `Blocks_RAW/RLE/CMP` → size stage-1 buffers (line 621-633) + grid sizes | **host-bound**: `dc->host_count.{raw,rle,cmp}BlockCount` is the same value from the CPU walk |
| 2 | `stage0` line 582 `sync` | — | the host must read the values above | removed (no host read) |
| 3 | `stage0` cross-check line 585 | (same counters) | assert GPU counts == host walk counts (corrupt input detection) | consistency check **→ device status** |
| 4 | `stage1` line 711 `read_counters` | `Counters` | `Seq_Streams` (grid, line 384/844/855), `Blocks_*`, `Seq_Streams_DecodedItems`, `HUF_Streams_DecodedBytes`, `HUF_WgtStreams`, `FseHufW/LLen/Offs/MLen`, `DecompressLiteralsGroups` → size stage-2 buffers + grids | **device-count** (indirect dispatch) for content-dependent counts; `Blocks_*` are host-bound |
| 5 | `stage1` line 718 `download FrameStatus` | `FrameStatus[]` (1 u32/frame) | per-frame error check (line 723-728) | consistency check **→ device status array** |
| 6 | `stage1` line 721 `sync` | — | host waits before reading FrameStatus | removed |
| 7 | `stage2` line 808 `read_counters` | `Counters` | `DecompressLiteralsGroups` (grid, line 836/837) = value written by `ComputePrefixSum` | **device-count** (indirect dispatch) |
| 8 | `stage2` line 896 `read_counters` | `Counters` | final totals for the caller (`sbz_dctx_counters`) | **device status / omit internal use** |
| 9 | `stage2` line 897 `sync` | — | the caller expects output on device before returning | removed (caller orders with events) |
| 10 | bshuf/LZ4 line 1380 `download GzLz4Status` | `GzLz4Status[]` (1 u32/chunk) | per-chunk corrupt-LZ4 check (line 1383-1387) | consistency check **→ device status array** |
| 11 | `bshuf_common` line 1537 `sync` | — | host waits for the whole chain | removed |

The `SLABITZ_DEBUG_SYNC` syncs at lines 807-893 are debug-only (gated by an env
var) — not on the default path.

Per plain-zstd call: 4 uploads, 5 downloads (4 `Counters` + `FrameStatus`),
about 27 kernel launches and 3 stream syncs. Line numbers are those of main
at 0a8a896.

## 2. What the host already knows

`host_prepare` (line 427) runs `sbz_CountFramesAndBlocks` and `sbz_CollectFrames`
on the host. So the host knows, before any GPU work:

- `frameCount`, each frame's `uncompSize`, `windowSize`, block-type start counts,
  and the running raw/RLE byte totals (`src/host_walk.h`).
- `rawBlockCount`, `rleBlockCount`, `cmpBlockCount`, `frameByteCount`
  (`sbz_CountFramesAndBlocksInfo`), accumulated into `dc->host_count` (line 482).

Therefore `Blocks_RAW/RLE/CMP`, the per-frame/total uncompressed byte counts, and
every buffer sized by `cmp`/`nf`/`raw`/`rle` are **host-bound** and need no
readback. This covers most of stage-0/1 `buf_size` calls (lines 621-665) and the
stage-0/1 launch grids that use `nf`/`cmp`/`raw`/`rle`.

## 3. What must be read on the device (indirect dispatch)

These are produced by the parse/decode kernels and are **not** in the frame
header — they come from the compressed-block contents (RFC 8878 §3.1.1.3):

- `Seq_Streams` — number of sequence streams (per compressed block).
- `Seq_Streams_DecodedItems` — total sequence count.
- `HUF_Streams_DecodedBytes` — total decoded literal bytes.
- `HUF_WgtStreams` — number of Huffman-weight streams (raw+FSE).
- `FseHufW`, `FseLLen`, `FseOffs`, `FseMLen` — FSE table counts per kind.
- `DecompressLiteralsGroups` — literal groups (from `ComputePrefixSum`).

Used only as **grid sizes** and (for literals/sequences) **intermediate buffer
sizes**. The DS kernel set was written for exactly this: `UpdateDispatchArgs.hlsl`
fills a `DispatchArgs` buffer from `Counters` and the stage driver issues an
indirect dispatch that reads it; each kernel exits early if its real count is
lower than the worst-case grid.

The host replacement is a **worst-case launch**: size each grid and buffer from a
host knowable bound, have the kernel read the real count from `Counters` and no-op
when its group index ≥ the real count. Bounds, all host knowable from the walk.
They must come from block counts, not from frame content sizes: the content
size field is optional in zstd and the walk records 0 when it is absent
(review correction). With `cmp` compressed blocks, each decoding to at most
128 KiB (Block_Maximum_Size, RFC 8878):

- `Seq_Streams` ≤ `cmp` (at most one sequence section per compressed block).
- `HUF_WgtStreams` ≤ `cmp`; each `Fse*` count ≤ `cmp` (at most one new table
  of each kind per block).
- Decoded literal bytes (`HUF_Streams_DecodedBytes`, the literals buffer)
  ≤ `cmp` x 128 KiB (a block's literals are part of its decoded content).
- `Seq_Streams_DecodedItems` ≤ `cmp` x 128 KiB / 3 (every sequence has a match
  of at least 3 bytes inside the block's 128 KiB), or tighter by the
  format's per-block sequence-count field maximum; step 2 should take the
  smaller.
- `DecompressLiteralsGroups` follows from the literal bytes and the
  grouping in `ComputePrefixSum`; step 2 must derive it from that kernel.

Worst case for the pac files at 16 frames per call: silicon u16 (8.9 MB per
frame, 256 kB bitshuffle blocks, each one zstd frame) and CeO2 u32 (17.9 MB
per frame): the literal bound is about the output size (143 / 286 MB per
call), so the extra scratch is of the order of the output itself. Step 2
should measure the real peak and consider tighter per-block bounds from the
block sizes the walk knows.

## 4. Consistency checks and errors become device status

Today the host reads `FrameStatus` (line 718) and `GzLz4Status` (line 1380) and
the stage-0/1 counter cross-check to detect corrupt input, returning
`SBZ_ERR_CORRUPT`. In the streaming design these become a **device status array**
the caller provides and polls (0 = OK, nonzero = a documented code). Buffer sizes
must not depend on them, which they already do not — the host-walk cross-check
(line 585) is a *consistency* check that can move fully device-side and be
reported in the status array. Errors the **host walk** finds (bad header, span
outside input) still return at once (`SBZ_ERR_CORRUPT`/`SBZ_ERR_ARGS`), because
they happen before anything is queued.

## 5. Measurement: does upload overlap decode today?

(Review correction, 2026-11-20. The first version measured the repo's small
test frames, started the clock before each thread had created its context
and warmed up, and ran its two threads in separate CUDA contexts; its "two
threads are 2x slower" did not hold.)

Harness: `bench/p4_overlap.py`. Real files (bitshuffle, 16 frames per call,
input built once), the current synchronous `sbz_dctx_*` API:
- serial: one `sbz_dctx`, calls back to back;
- 2ctx: two threads, each `sbz_dctx` in its own CUDA context (no context
  current at create, so the library creates one per thread);
- shared: two threads, both with the device's primary context current at
  create: one context, two streams.
Clock starts after both threads are warm. L40S on the Slurm `low-gpu`
partition (job 38682493, GPU idle at start), `build/kernels` fatbins:

| 16 frames/call | serial ms/call | 2ctx | shared |
|---|---|---|---|
| silicon LZ4 (2.3 MB in) | 0.67 | x0.94 | x1.26 |
| CeO2 LZ4 (62.9 MB in) | 4.37 | x1.22 | x1.13 |
| silicon zstd (0.7 MB in) | 2.59 | x0.89 | x1.56 |
| CeO2 zstd (58.5 MB in) | 9.58 | x1.72 | x1.81 |

**Answer: partly, already.** Two threads sharing one context overlap one
call's upload and host work with the other's kernels, up to 1.8x, with
today's blocking calls. Separate contexts are inconsistent (slower on the
silicon files): the GPU time-slices contexts. So the API must run on the
caller's context. CeO2 LZ4 at 3.9 ms per call is near its upload bound
(62.9 MB at about 21 GB/s pageable is 2.9 ms; pinned memory would raise
that). What step 2 can still gain is the gap between these numbers and
max(upload, device) per call, and a single-threaded caller.

## 6. Implications for step 2 (workscope, for review to settle)

- **LZ4 first** (simplest, per the plan): the only mid-call readback is the
  `GzLz4Status` download (line 1380) + final sync. Keep status on the device,
  return after queueing.
- **zstd**: replace the three `read_counters` + `FrameStatus` readbacks. Move the
  content-dependent counts (section 3) to device-count/indirect dispatch by
  giving the worst-case grid and letting the kernel read `Counters` and exit
  early; size the intermediate buffers from host bounds (`totalUncomp`, `cmp`).
  Keep `SLABITZ_PROFILE=1` (it syncs by design) and add a debug switch (e.g.
  `SLABITZ_CHECK_SYNC=1`) that restores the readback checks for diagnosing
  corrupt input.
- **Device status arrays** for per-frame (zstd) and per-chunk (bitshuffle/LZ4)
  results, with codes documented in the header.
- The plain-zstd call returns without a trailing sync; the host does not wait.
  Because a context has one in-order stream and the call reuses its buffers,
  cross-call ordering needs a per-context event recorded at the end of each call
  and the next call's stream waits on it (`cuStreamWaitEvent`) before reusing
  scratch — exactly the P4 step-3 design.
- The `download`-then-sync pattern in `sbz_dctx_read_buffer` (line 1596-1597) is
  the debug hook and stays as-is.
