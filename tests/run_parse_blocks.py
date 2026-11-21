#!/usr/bin/env python3
"""Run the DC ParseCompressedBlocks kernel on a single real compressed block.

Pulls a compressed block out of a real zstd frame (zstd_frame), hands it to the GPU
ParseCompressedBlocks cubin, and dumps the parse outputs (FseInfos, FseProbs,
CompressedBlocks, HufRefs, LitRefs, SeqStream refs, per-block size prefix, and the
write-side lookback buffers). Validates that the 27-buffer launch + binding works at
runtime and produces the intermediate metadata that the decompress passes read.

Usage:
    run_parse_blocks.py <cubin> [--payload-size N] [--compress-level L]
"""
import ctypes as ct
import struct
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from slabitz import loader, zstd_frame

CUBIN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build/pcb/pcb.cubin")


class Buf(ct.Structure):
    _fields_ = [("data", ct.c_uint64), ("count", ct.c_uint64)]


MAXCPB = 1 << 16
FSEPROBS = 4096
MAX_COUNTERS = 64
MAX_HUFW = MAXCPB

# DS compile-time constants (mirror zstdgpu_structs.h)
MAX_COUNT_FSEPROBS = 64
FSE_RLE_TABLE_COUNT = 256
FSE_PROB_ACC = {"HufW": 7, "LLen": 9, "Offs": 8, "MLen": 9}
FSE_PR_ACC = {"HufW": 7, "LLen": 9, "Offs": 8, "MLen": 9}


def make_payload(n=541, seed=1):
    import random
    random.seed(seed)
    nbytes = n - 200
    body = bytes(random.randrange(32, 127) for _ in range(nbytes))
    return (b"hello world " * 20 + b"\n" + body)[:n]


def make_compressed_block(payload, level=5):
    import zstandard as zstd
    comp = zstd.ZstdCompressor(level=level).compress(payload)
    frames, blocks = zstd_frame.parse_frame(comp)
    cbs = [b for b in blocks if b.type_name == "compressed"]
    assert cbs, "no compressed block produced"
    return cbs[0].data


def run(drv, fn, block):
    n = len(block)
    # inCompressedData: u32 array of the block bytes (padded to dword multiple)
    data = bytearray(block)
    data += b"\x00" * ((-len(data)) % 4)
    nbytes = len(data)
    n_u32 = nbytes // 4

    # inBlocksCMPRefs[0] = {offs=0, size=n} (byte size of the compressed block)
    refs = struct.pack("II", 0, n)

    buf_dword = loader.DeviceMemory(drv, n_u32 * 4)
    buf_dword.from_host(bytes(data))

    # --- output/scratch buffers (generous) ---
    cnt = loader.DeviceMemory(drv, MAX_COUNTERS * 4)
    fsainfo = loader.DeviceMemory(drv, MAXCPB * 4)          # FseInfo (1 u32 each)
    cbd = loader.DeviceMemory(drv, MAXCPB * 16)             # CompressedBlockData (16 B)
    huf = loader.DeviceMemory(drv, MAXCPB * 8)              # OffsetAndSize
    lit = loader.DeviceMemory(drv, MAXCPB * 16)             # LitStreamInfo (16 B)
    sqto = loader.DeviceMemory(drv, MAXCPB * 8)             # OffsetAndSize
    sll = loader.DeviceMemory(drv, MAXCPB * 4)
    sof = loader.DeviceMemory(drv, MAXCPB * 4)
    sml = loader.DeviceMemory(drv, MAXCPB * 4)
    sbi = loader.DeviceMemory(drv, MAXCPB * 4)
    bsp = loader.DeviceMemory(drv, MAXCPB * 4)              # BlockSizePrefix
    pfsm = loader.DeviceMemory(drv, MAXCPB * 4)             # PerFrameSeqStreamMinIdx
    psss = loader.DeviceMemory(drv, MAXCPB * 4)             # PerSeqStreamSeqStart
    sq_lookback = loader.DeviceMemory(drv, MAXCPB * 4)
    bsq_lookback = loader.DeviceMemory(drv, MAXCPB * 4)
    fprobs = loader.DeviceMemory(drv, FSEPROBS * 4)
    dhwc = loader.DeviceMemory(drv, MAX_HUFW * 4)           # DecompressedHuffmanWeightCount
    lit_lookback = loader.DeviceMemory(drv, MAXCPB * 4)
    hw2hl = loader.DeviceMemory(drv, MAXCPB * 4)
    hl2ls = loader.DeviceMemory(drv, MAXCPB * 4)
    hl2hw = loader.DeviceMemory(drv, MAXCPB * 4)
    hcl = loader.DeviceMemory(drv, MAXCPB * 4)
    pfb = loader.DeviceMemory(drv, 64 * 4)                  # inPerFrameBlockCountCMP
    gbi = loader.DeviceMemory(drv, MAXCPB * 4)              # inGlobalBlockIndexPerCmpBlock
    dargs = loader.DeviceMemory(drv, 64 * 4)                # inDispatchArgs
    pfbM = loader.DeviceMemory(drv, 8)                      # inBlocksCMPRefs (1 OffsetAndSize)
    gbiM = loader.DeviceMemory(drv, 4)                      # inGlobalBlockIndexPerCmpBlock
    dargsM = loader.DeviceMemory(drv, 16 * 4)               # inDispatchArgs
    pfbM.from_host(refs)                                    # {offs=0, size=n}
    gbiM.from_host(struct.pack("i", 0))
    dargsM.from_host(struct.pack("16i", *([0] * 16)))
    pfb.from_host(struct.pack("16i", *([1] + [0] * 15)))    # frame0 cumulative CMP block count
    # zero-init all outputs so unwritten entries are deterministic (RAW/RLE blocks write none)
    for m in (cnt, fsainfo, cbd, huf, lit, sqto, sll, sof, sml, sbi, bsp, pfsm, psss,
              sq_lookback, bsq_lookback, fprobs, dhwc, lit_lookback, hw2hl, hl2ls, hl2hw, hcl):
        m.from_host(bytes(m.size))
    args = [
        Buf(cnt.ptr, MAX_COUNTERS), Buf(fsainfo.ptr, MAXCPB), Buf(cbd.ptr, MAXCPB),
        Buf(huf.ptr, MAXCPB), Buf(lit.ptr, MAXCPB), Buf(sqto.ptr, MAXCPB),
        Buf(sll.ptr, MAXCPB), Buf(sof.ptr, MAXCPB), Buf(sml.ptr, MAXCPB),
        Buf(sbi.ptr, MAXCPB), Buf(bsp.ptr, MAXCPB), Buf(pfsm.ptr, MAXCPB),
        Buf(psss.ptr, MAXCPB), Buf(sq_lookback.ptr, MAXCPB), Buf(bsq_lookback.ptr, MAXCPB),
        Buf(fprobs.ptr, FSEPROBS), Buf(dhwc.ptr, MAX_HUFW), Buf(lit_lookback.ptr, MAXCPB),
        Buf(hw2hl.ptr, MAXCPB), Buf(hl2ls.ptr, MAXCPB), Buf(hl2hw.ptr, MAXCPB),
        Buf(hcl.ptr, MAXCPB),
        Buf(buf_dword.ptr, n_u32), Buf(pfbM.ptr, 1), Buf(pfb.ptr, 16), Buf(gbiM.ptr, 1),
        Buf(dargsM.ptr, 16),
        ct.c_uint32(0),                       # tgOffset
        ct.c_uint32(1),                       # workItemCount
        ct.c_uint32(nbytes),                  # compressedBufferSizeInBytes
        ct.c_uint32(1),                       # frameCount
        ct.c_uint32(1),                       # compressedBlockCount
    ]
    fn(grid=(1, 1, 1), block=(32, 1, 1), args=args, shared=0)
    drv.synchronize()

    def words(mem, count):
        raw = mem.to_host(count * 4)
        return list(struct.unpack(f"{count}I", raw))

    out = {
        "counters": words(cnt, 24),
        "fse_info": words(fsainfo, 8),
        "fse_probs_nonzero": [v for v in words(fprobs, 256) if v],
        "block_datas": [struct.unpack("IIII", cbd.to_host(16))],
        "huf_refs": [struct.unpack("II", huf.to_host(8))],
        "lit_refs": [struct.unpack("16I", lit.to_host(64))[i * 4:i * 4 + 4] for i in range(4)],
        "seq_stream_to_ref": [struct.unpack("II", sqto.to_host(8))],
        "seq_LLen": words(sll, 8), "seq_Offs": words(sof, 8), "seq_MLen": words(sml, 8),
        "seq_blockid": words(sbi, 8),
        "block_size_prefix": words(bsp, 8),
        "decomp_hw_count": words(dhwc, 16),
    }
    return out


def main():
    with open(CUBIN, "rb") as f:
        cubin = f.read()
    payload = make_payload()
    block = make_compressed_block(payload)
    print(f"payload len={len(payload)}  compressed block len={len(block)}")
    print("block head:", block[:8].hex())

    drv = loader.Driver(0)
    fn = drv.load(cubin).function("main_0")
    out = run(drv, fn, block)

    for k, v in out.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
