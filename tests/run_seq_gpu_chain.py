#!/usr/bin/env python3
"""GPU-vs-oracle comparison for the DecompressSequences SingleStream kernel.

Chain: parse the real block -> build the LLen/Offs/MLen FSE DTables (cpu_ref_ifs) ->
run the GPU DecompressSequences cubin (SingleStream, one threadgroup per seq stream) ->
read back LLen/MLen/Offs -> compare to the oracle `SEQ` hook.

Usage: run_seq_gpu_chain.py <seq.cubin>
Env: DSQ_MULTI=1 for word-soup multi-sequence blocks; DSQ_SEED/DSQ_N select the payload.
"""
import os
import random
import sys
import struct
import ctypes as ct
import zstandard as zstd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from slabitz import zstd_frame, loader
from kernels import cpu_ref_pcb, cpu_ref_ifs, cpu_ref_dsq
from tools.edu_oracle import run_oracle

CUBIN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build/seq/seq.cubin")

FSE_RLE_TABLE_COUNT = 256
FSE_MAX_COUNT_PROBS = 64
MAX_SEQ = 20000

DEF_LLEN = [4,3,2,2,2,2,2,2,2,2,2,2,2,1,1,1,2,2,2,2,2,2,2,2,2,3,2,1,1,1,1,1,-1,-1,-1,-1]
DEF_OFFS = [1,1,1,1,1,1,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1]
DEF_MLEN = [1,4,3,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1]


class Buf(ct.Structure):
    _fields_ = [("data", ct.c_uint64), ("count", ct.c_uint64)]


def make_payload(n=541, seed=1):
    if os.environ.get("DSQ_MULTI"):
        words = "the quick brown fox jumps over lazy dog run fast stop slow near far".split()
        random.seed(seed)
        return " ".join(random.choice(words) for _ in range(int(n) if n > 10000 else 12000)).encode()
    random.seed(seed)
    return (b"hello world " * 20 + b"\n" + bytes(random.randrange(32, 127) for _ in range(n - 200)))[:n]


def main():
    payload = make_payload(int(os.environ.get("DSQ_N", 541)), int(os.environ.get("DSQ_SEED", 1)))
    comp = zstd.ZstdCompressor(level=5).compress(payload)
    _, blocks = zstd_frame.parse_frame(comp)
    block = [b for b in blocks if b.type_name == "compressed"][0].data
    n = len(block)

    r = cpu_ref_pcb.parse_compressed_block(block, n)
    sf = r["seq_fse"]
    info = r["fse_info"]

    if r["seq_stream_to_ref"] is None:
        print("no sequence stream; nothing to validate")
        return 0

    def build_dtable(name):
        ds_idx = sf[name]
        local_idx = ds_idx - FSE_RLE_TABLE_COUNT
        if ds_idx in info:
            acc = info[ds_idx] >> 8
            return acc, cpu_ref_ifs.ref_initfse_table(r["fse_probs"], info[ds_idx], local_idx, 0, 0, 1 << acc)
        acc = {"LLen": 6, "Offs": 5, "MLen": 6}[name]
        dist = {"LLen": DEF_LLEN, "Offs": DEF_OFFS, "MLen": DEF_MLEN}[name]
        probs = [0] * (16 * FSE_MAX_COUNT_PROBS)
        probs[local_idx * FSE_MAX_COUNT_PROBS:local_idx * FSE_MAX_COUNT_PROBS + len(dist)] = dist
        return acc, cpu_ref_ifs.ref_initfse_table(probs, acc << 8 | len(dist), local_idx, 0, 0, 1 << acc)

    acc_llen, dtable_llen = build_dtable("LLen")
    acc_offs, dtable_offs = build_dtable("Offs")
    acc_mlen, dtable_mlen = build_dtable("MLen")

    src_offs, src_size = r["seq_stream_to_ref"]
    block_pad = block + b"\x00" * ((-len(block)) % 4)

    truth_flat = run_oracle(comp, expected=payload)["SEQ"][0]
    truth = [(truth_flat[3 * i], truth_flat[3 * i + 1], truth_flat[3 * i + 2])
             for i in range(len(truth_flat) // 3)]
    num_seqs = len(truth)
    print(f"block={n} B seq_stream=(src_offs,{src_size}) accs=({acc_llen},{acc_offs},{acc_mlen}) seqs={num_seqs}")

    start_llen = cpu_ref_dsq.compute_fse_data_start_llen(0, 1)
    start_offs = cpu_ref_dsq.compute_fse_data_start_offs(0, 1)
    start_mlen = cpu_ref_dsq.compute_fse_data_start_mlen(0, 1)
    fse_elems = [0] * 2432
    for j, v in enumerate(dtable_llen):
        fse_elems[start_llen + j] = v
    for j, v in enumerate(dtable_offs):
        fse_elems[start_offs + j] = v
    for j, v in enumerate(dtable_mlen):
        fse_elems[start_mlen + j] = v

    # ---- GPU setup ----
    data = bytearray(block_pad)
    with open(CUBIN, "rb") as f:
        cubin = f.read()
    drv = loader.Driver(0)
    fn = drv.load(cubin).function("main_0")

    def dev_u32(vals):
        m = loader.DeviceMemory(drv, len(vals) * 4)
        m.from_host(struct.pack("%dI" % len(vals), *vals))
        return m

    comp_m = dev_u32(list(struct.unpack("%dI" % (len(data) // 4), data)))

    counters = dev_u32([0] * 24)
    ctr = [0] * 24
    ctr[6] = num_seqs          # Seq_Streams_DecodedItems
    ctr[8] = 1                 # Seq_Streams
    ctr[15] = 1                # Blocks_CMP
    counters.from_host(struct.pack("%dI" % 24, *ctr))

    seq_stream_to_ref = dev_u32([src_offs, src_size])   # zstdgpu_OffsetAndSize (offs, size)
    seq_to_llen = dev_u32([sf["LLen"]])
    seq_to_offs = dev_u32([sf["Offs"]])
    seq_to_mlen = dev_u32([sf["MLen"]])
    seq_to_blockid = dev_u32([0])

    fse_info_arr = [0] * 512
    for name, idx in (("LLen", sf["LLen"]), ("Offs", sf["Offs"]), ("MLen", sf["MLen"])):
        fse_info_arr[idx] = info[idx] if idx in info else ({"LLen": 36, "Offs": 29, "MLen": 53}[name] | ({"LLen": 6, "Offs": 5, "MLen": 6}[name] << 8))
    fse_infos = dev_u32(fse_info_arr)

    per_stream_seq_start = dev_u32([0])                 # inPerSeqStreamSeqStart[0]
    fse_elems_mem = dev_u32(fse_elems)
    dispatch_args = dev_u32([0] * 16)

    out_llen_mem = dev_u32([0] * MAX_SEQ)
    out_mlen_mem = dev_u32([0] * MAX_SEQ)
    out_offs_mem = dev_u32([0] * MAX_SEQ)
    literal_size = r["block_size_prefix"]
    out_block_size = dev_u32([literal_size, 0, 0, 0])
    out_fo1 = dev_u32([0] * 8)
    out_fo2 = dev_u32([0] * 8)
    out_fo3 = dev_u32([0] * 8)

    args = [
        Buf(out_llen_mem.ptr, MAX_SEQ),
        Buf(out_mlen_mem.ptr, MAX_SEQ),
        Buf(out_offs_mem.ptr, MAX_SEQ),
        Buf(out_block_size.ptr, 4),
        Buf(out_fo1.ptr, 8),
        Buf(out_fo2.ptr, 8),
        Buf(out_fo3.ptr, 8),
        Buf(counters.ptr, 24),        # inCounters
        Buf(comp_m.ptr, len(data)),   # inCompressedData (ByteAddressBuffer)
        Buf(seq_stream_to_ref.ptr, 1),  # inSeqStreamToRef
        Buf(seq_to_llen.ptr, 1),
        Buf(seq_to_offs.ptr, 1),
        Buf(seq_to_mlen.ptr, 1),
        Buf(seq_to_blockid.ptr, 1),
        Buf(fse_infos.ptr, 512),      # inFseInfos
        Buf(per_stream_seq_start.ptr, 1),  # inPerSeqStreamSeqStart
        Buf(fse_elems_mem.ptr, len(fse_elems)),  # inFseElems
        Buf(dispatch_args.ptr, 16),   # inDispatchArgs
        ct.c_uint32(0),               # tgOffset
        ct.c_uint32(1),               # workItemCount
    ]
    print("launching main_0 (grid=1, block=1)...", flush=True)
    disp_block = 32 if os.environ.get("DSQ_SEQMS") else 1
    fn(grid=(1, 1, 1), block=(disp_block, 1, 1), args=args, shared=0)
    drv.synchronize()
    print("synced", flush=True)

    g_llen = out_llen_mem.to_host(MAX_SEQ * 4)
    g_mlen = out_mlen_mem.to_host(MAX_SEQ * 4)
    g_offs = out_offs_mem.to_host(MAX_SEQ * 4)
    g_llen = struct.unpack("%dI" % num_seqs, g_llen[:num_seqs * 4])
    g_mlen = struct.unpack("%dI" % num_seqs, g_mlen[:num_seqs * 4])
    g_offs = struct.unpack("%dI" % num_seqs, g_offs[:num_seqs * 4])

    mism = 0
    for i in range(num_seqs):
        tl, tm, to = truth[i]
        if g_llen[i] != tl or g_mlen[i] != tm:
            mism += 1
            print(f"  seq {i}: gpu=({g_llen[i]},{g_mlen[i]},{g_offs[i]}) oracle=({tl},{tm},{to}) LL_ML_MISMATCH")
        if to > 3 and g_offs[i] != to:
            mism += 1
            print(f"  seq {i}: off gpu={g_offs[i]} oracle(coded)={to} OFFS_MISMATCH")

    ok = mism == 0
    print(f"{'PASS' if ok else 'FAIL'}: GPU seq decode vs oracle ({num_seqs} seqs, {mism} mismatch)")
    bs = struct.unpack("%dI" % 4, out_block_size.to_host(16))
    expect_bs = (sum(t[1] for t in truth) + literal_size) & 0xFFFFFFFF
    print(f"  inoutBlockSizePrefix[0]={bs[0]} (expect totalMLen+litSize={expect_bs}) "
          f"finalOff1={struct.unpack('I', out_fo1.to_host(4))[0]:#x}")
    if bs[0] != expect_bs:
        print("  FAIL: blockSizePrefix mismatch")
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
