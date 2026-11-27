#!/usr/bin/env python3
"""GPU-vs-oracle comparison for the FinaliseSequenceOffsets kernel.

Chain: decode sequences (cpu_ref_dsq) -> run the GPU FinaliseSequenceOffsets cubin ->
read back the resolved real byte offsets -> compare to the oracle `OFFS` hook.
Reuses run_fso_chain.decode_block for the sequence decode + folded offsets.

Usage: run_fso_gpu_chain.py <fso.cubin>
Env: DSQ_MULTI=1 for multi-sequence blocks; DSQ_SEED/DSQ_N select the payload.
"""
import os
import sys
import struct
import ctypes as ct

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from slabitz import loader
from tools.edu_oracle import run_oracle
from run_fso_chain import decode_block, make_payload
import kparams

CUBIN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build/fso/fso.cubin")
MAX_SEQ = 20000


class Buf(ct.Structure):
    _fields_ = [("data", ct.c_uint64), ("count", ct.c_uint64)]


def main():
    payload = make_payload(int(os.environ.get("DSQ_N", 541)), int(os.environ.get("DSQ_SEED", 1)))
    comp, pl, r, folded, fo1, fo2, fo3, num_seqs = decode_block(payload)
    d = run_oracle(comp, expected=pl)
    truth = [x[0] for x in d["OFFS"]]
    print(f"seqs={num_seqs}")

    with open(CUBIN, "rb") as f:
        cubin = f.read()
    drv = loader.Driver(0)
    fn = drv.load(cubin).function("main_0")

    def dev_u32(vals):
        m = loader.DeviceMemory(drv, len(vals) * 4)
        m.from_host(struct.pack("%dI" % len(vals), *vals))
        return m

    counters = dev_u32([0] * 24)
    ctr = [0] * 24
    ctr[6] = num_seqs          # Seq_Streams_DecodedItems
    ctr[8] = 1                 # Seq_Streams
    ctr[15] = 1                # Blocks_CMP
    ctr[18] = 1                # Frames
    counters.from_host(struct.pack("%dI" % 24, *ctr))

    out_offs = dev_u32([0] * MAX_SEQ)
    fso1 = dev_u32([fo1])
    fso2 = dev_u32([fo2])
    fso3 = dev_u32([fo3])
    per_seq_stream_start = dev_u32([0])
    per_frame_block_count_all = dev_u32([1])          # frame 0 has 1 block
    per_frame_seq_stream_min_idx = dev_u32([0])       # frame 0 first seq stream = 0
    seq_stream_to_block_id = dev_u32([0])
    dispatch_args = dev_u32([0] * 16)

    # inoutDecompressedSequenceOffs: seed with the encoded DecompressSequences offsets (padded).
    offs_device = dev_u32(folded + [0] * (MAX_SEQ - num_seqs))

    args = [
        Buf(offs_device.ptr, MAX_SEQ),
        Buf(counters.ptr, 1),                  # inCounters (one 88-byte struct)
        Buf(fso1.ptr, 1),
        Buf(fso2.ptr, 1),
        Buf(fso3.ptr, 1),
        Buf(per_seq_stream_start.ptr, 1),
        Buf(per_frame_block_count_all.ptr, 1),
        Buf(per_frame_seq_stream_min_idx.ptr, 1),
        Buf(seq_stream_to_block_id.ptr, 1),
        Buf(dispatch_args.ptr, 16),
        ct.c_uint32(0),        # tgOffset
        ct.c_uint32(num_seqs), # workItemCount (the shader bounds by Seq_Streams_DecodedItems)
        ct.c_uint32(MAX_SEQ),  # seqCapacity (sequence slots in the offsets buffer)
    ]
    kparams.check_args("FinaliseSequenceOffsets", args)

    import math
    grid = (max(1, (num_seqs + 255) // 256), 1, 1)
    print(f"launching main_0 grid={grid} block=256...", flush=True)
    fn(grid=grid, block=(256, 1, 1), args=args, shared=0)
    drv.synchronize()
    print("synced", flush=True)

    g_offs = struct.unpack("%dI" % num_seqs, offs_device.to_host(num_seqs * 4))

    mism = 0
    for i in range(num_seqs):
        if g_offs[i] != truth[i]:
            mism += 1
            if mism <= 5:
                print(f"  seq {i}: gpu={g_offs[i]} oracle={truth[i]}")
    ok = mism == 0 and len(g_offs) == len(truth)
    print(f"{'PASS' if ok else 'FAIL'}: GPU FinaliseSequenceOffsets vs oracle ({num_seqs} seqs, {mism} mismatch)")
    print("compared %d sequence offsets" % (num_seqs if ok else 0))
    return 0 if ok and num_seqs > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
