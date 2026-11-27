T07 zstd-version corpus: decoder bug (#1 of 1)

--patch-from frame faults the GPU instead of being rejected.

patchfrom_frame.zst   (1432 bytes)  a zstd frame written by `zstd --patch-from=<base>`.
patchfrom_base.bin    (20000 bytes) the patch reference base used to encode/decode it.

What it is: a frame produced by `zstd --patch-from=<base>` (level 3, v1.4.7 CLI).
The frame references an external prefix (the base) held by both encoder and
decoder.  It has NO Dictionary_ID in the header (libzstd decodes it only when
given the base back with `-d --patch-from`).

Expected: the slabitz decoder must REJECT the frame.  A --patch-from frame
cannot be decoded without the reference prefix, and the decoder has no such
feature (like dictionaries, it is out of scope for v1); the task expectation is
reject:SBZ_ERR_CORRUPT.

Actual: decoding the frame with sbz_dctx_decompress_frames (or
sbz_dctx_decompress) faults the GPU:

    sbz_dctx_decompress_frames: readback Counters: cuMemcpyDtoHAsync failed:
      CUresult 700 (an illegal memory access was encountered)

This is a decoder bug, not simply an unsupported mode: it is a hard GPU crash
(CUresult 700), it poisons the CUDA primary context for the whole process, and
it is reproducible with any of the corpus versions (the frames differ only in
encoder, the crash is identical).  libzstd decodes the same frame correctly
(120146 bytes, sha256 b253970ede89791ac51db93a57e331810ce63cf02c87bcc4119d015ac74e9228)
when given the base; the decoder must return reject:SBZ_ERR_CORRUPT instead of
faulting.

Reproduce (from the worktree):

    . ../../env.sh
    # the fault:
    python3 -c "import ctypes as ct,sys; sys.argv=['x'];
      import run_compat as rc; 
      # decode the frame with sbz_dctx_decompress_frames -> CUresult 700
      ..." tests/data/compat_bugs/patchfrom_frame.zst
    # libzstd decodes it:
    $SLABITZ_DEV_ROOT/toolchains/zstd-v1.4.7/programs/zstd -d -c \
      --patch-from=tests/data/compat_bugs/patchfrom_base.bin \
      tests/data/compat_bugs/patchfrom_frame.zst > /dev/null

Why a bug / task to fix: T09 (decoder rejects & bounds) or T10 should make the
GPU parser reject a frame that needs an out-of-band prefix (patch-from) with
SBZ_ERR_CORRUPT, as it already does for a Dictionary_ID.  Written by the T07
worker; the decoder was NOT changed (per the task rule "do not fix the decoder:
record them").
