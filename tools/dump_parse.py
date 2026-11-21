#!/usr/bin/env python3
"""Dump the Stage 0+1 (frame/block parse) results for one input, for diffing GPUs.

Runs sbz_dctx_parse on the zstd frames of a bitshuffle-zstd chunk (data/*.chunk.bin,
first --frames frames) and prints the counters and the parse buffers, even if the
parse fails. Run it with the same input on a known-good GPU and on the card under
test, and diff the two outputs. Only needs Python 3 (ctypes), like check_bshuf.py.

  python3 tools/dump_parse.py --lib build/libslabitz.so --kernels build/sm_35 \\
      --chunk data/z-l9-text.chunk.bin [--frames 1] [--n 16]
"""
import argparse
import ctypes as ct
import struct
import sys

COUNTERS = ("FseHufW FseLLen FseOffs FseMLen DecompressLiteralsGroups HUF_WgtStreams "
            "Seq_Streams_DecodedItems HUF_Streams_DecodedBytes Seq_Streams HUF_Streams HufLit "
            "RAW_Streams RLE_Streams Blocks_RAW Blocks_RLE Blocks_CMP BlocksBytes_RAW "
            "BlocksBytes_RLE Frames Frames_UncompressedByteSize "
            "Frames_AcceptedUncompressedByteSize Frames_RequestTooLarge").split()

BUFFERS = ("FrameStatus PerFrameBlockCountRAW PerFrameBlockCountRLE PerFrameBlockCountCMP "
           "PerFrameBlockCountAll BlocksRAWRefs BlocksRLERefs BlocksCMPRefs BlockSizePrefix "
           "RawBlockSizePrefix RleBlockSizePrefix GlobalBlockIndexPerCmpBlock "
           "SeqStreamToRef LitRefs HufRefs").split()


def chunk_frames(blob, nmax):
    """(offs, size) of the zstd frames in a bitshuffle-zstd chunk."""
    total, block_bytes = struct.unpack(">QI", blob[:12])
    pos, out = 12, []
    while pos + 4 <= len(blob) and len(out) < nmax:
        (c,) = struct.unpack(">I", blob[pos:pos + 4])
        if c == 0 or pos + 4 + c > len(blob):
            break
        out.append((pos + 4, c))
        pos += 4 + c
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lib", required=True)
    ap.add_argument("--kernels", required=True)
    ap.add_argument("--chunk", required=True)
    ap.add_argument("--frames", type=int, default=1, help="first N frames of the chunk")
    ap.add_argument("--n", type=int, default=16, help="u32 words printed per buffer")
    ap.add_argument("--device", type=int, default=0)
    a = ap.parse_args()

    blob = open(a.chunk, "rb").read()
    spans = chunk_frames(blob, a.frames)
    if not spans:
        raise SystemExit("no zstd frames in %s" % a.chunk)
    src = b"".join(blob[o:o + s] for o, s in spans)
    print("input %s: %d frame(s), %d bytes, sizes %s" % (a.chunk, len(spans), len(src),
                                                        [s for _, s in spans]))

    lib = ct.CDLL(a.lib)
    lib.sbz_dctx_create_cuda.restype = ct.c_void_p
    lib.sbz_dctx_create_cuda.argtypes = [ct.c_int, ct.c_char_p, ct.c_char_p, ct.c_size_t]
    lib.sbz_dctx_error.restype = ct.c_char_p
    lib.sbz_dctx_error.argtypes = [ct.c_void_p]
    lib.sbz_dctx_parse.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_size_t]
    lib.sbz_dctx_counters.argtypes = [ct.c_void_p, ct.c_void_p]
    lib.sbz_dctx_read_buffer.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_void_p,
                                        ct.POINTER(ct.c_size_t)]
    lib.sbz_dctx_destroy.argtypes = [ct.c_void_p]

    err = ct.create_string_buffer(512)
    dc = lib.sbz_dctx_create_cuda(a.device, a.kernels.encode(), err, len(err))
    if not dc:
        raise SystemExit("sbz_dctx_create_cuda: %s" % err.value.decode())
    rc = lib.sbz_dctx_parse(dc, src, len(src))
    print("sbz_dctx_parse rc=%d %s" % (rc, lib.sbz_dctx_error(dc).decode()))

    cnt = (ct.c_uint32 * len(COUNTERS))()
    if lib.sbz_dctx_counters(dc, cnt) == 0:
        print("counters: " + " ".join("%s=%d" % (n, v) for n, v in zip(COUNTERS, cnt) if v))
    for name in BUFFERS:
        size = ct.c_size_t(0)
        if lib.sbz_dctx_read_buffer(dc, name.encode(), None, ct.byref(size)) != 0 or not size.value:
            print("%-28s (empty)" % name)
            continue
        raw = (ct.c_uint8 * size.value)()
        lib.sbz_dctx_read_buffer(dc, name.encode(), raw, ct.byref(size))
        words = struct.unpack("<%dI" % (size.value // 4), bytes(raw)[:size.value // 4 * 4])
        print("%-28s %s" % (name, " ".join("%x" % w for w in words[:a.n])))
    lib.sbz_dctx_destroy(dc)


if __name__ == "__main__":
    sys.exit(main())
