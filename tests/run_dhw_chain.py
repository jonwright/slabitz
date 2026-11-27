#!/usr/bin/env python3
"""GPU-vs-CPU DecompressHuffmanWeights comparison on the real parse's HufW weights.

Chain: parse the real block (CPU), build the HufW FSE DTable (cpu_ref_ifs), then run both the
GPU DecompressHuffmanWeights cubin and the CPU reference (cpu_ref_dhw) on the same inputs and
compare the decoded Huffman weights element-exactly.

Usage:
    run_dhw_chain.py <dhw.cubin>
"""
import os
import sys
import ctypes as ct
import struct
import zstandard as zstd
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from slabitz import zstd_frame, loader
import kparams  # tests/kparams.py
from kernels import cpu_ref_pcb, cpu_ref_ifs, cpu_ref_dhw
from tools.edu_oracle import run_oracle

CUBIN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build/dhw/dhw.cubin")

MAX_HUFW = 256
FSE_ACC = {"HufW": 7, "LLen": 9, "Offs": 8, "MLen": 9}
FSE_ELEM = 128


class Buf(ct.Structure):
    _fields_ = [("data", ct.c_uint64), ("count", ct.c_uint64)]


def make_payload(n=541, seed=1):
    random.seed(seed)
    return (b"hello world " * 20 + b"\n" + bytes(random.randrange(32, 127) for _ in range(n - 200)))[:n]


def main():
    import os
    payload = make_payload(int(os.environ.get("DHW_N", 541)), int(os.environ.get("DHW_SEED", 1)))
    comp = zstd.ZstdCompressor(level=5).compress(payload)
    _, blocks = zstd_frame.parse_frame(comp)
    block = [b for b in blocks if b.type_name == "compressed"][0].data
    n = len(block)

    r = cpu_ref_pcb.parse_compressed_block(block, n)
    info = r["fse_info"].get(256)
    acc = info >> 8
    probs = r["fse_probs"][0:64]
    offs, size = r["huf_refs"][0][1]
    print(f"block={n} B  acc={acc}  weights stream offs={offs} size={size}")

    # Build HufW FSE DTable (validated cpu_ref_ifs) and place at ComputeFseDataStartHufW(0)=256
    dtable = cpu_ref_ifs.ref_initfse_table(probs, info, 0, 0, 0, 1 << acc)
    fse_elems = [0] * (512)
    base = cpu_ref_dhw.compute_fse_data_start_hufw(0, 1)
    for j, v in enumerate(dtable):
        fse_elems[base + j] = v

    # --- ground truth: instrumented zstd educational decoder (output checked byte-exact) ---
    truth = run_oracle(comp, expected=payload)["HUFW"][0]
    print(f"oracle weights count={len(truth)}")

    # --- CPU decode ---
    cw, ccount = cpu_ref_dhw.decompress_huffman_weights(block, offs, size, fse_elems, acc, 1, 0)
    print(f"CPU weights count={ccount} n={len(cw)}")

    # --- GPU decode ---
    data = bytearray(block)
    data += b"\x00" * ((-len(data)) % 4)
    n_u32 = len(data) // 4
    with open(CUBIN, "rb") as f:
        cubin = f.read()
    drv = loader.Driver(0)
    fn = drv.load(cubin).function("main_0")

    comp_data = loader.DeviceMemory(drv, len(data)); comp_data.from_host(bytes(data))
    huf_refs = loader.DeviceMemory(drv, 8)               # OffsetAndSize {offs,size}
    huf_refs.from_host(struct.pack("II", offs, size))
    counters = loader.DeviceMemory(drv, 32 * 4)
    ctr = [0] * 32
    ctr[0] = 1       # FseHufW
    ctr[15] = 1      # Blocks_CMP
    counters.from_host(struct.pack("32i", *ctr))
    fse_infos = loader.DeviceMemory(drv, 512 * 4)
    fi = [0] * 512
    fi[256] = info   # ComputeFseIndexHufW(0)=256
    fse_infos.from_host(struct.pack("512i", *fi))
    fse_elems_m = loader.DeviceMemory(drv, 512 * 4)
    fse_elems_m.from_host(struct.pack("512I", *[v & 0xFFFFFFFF for v in fse_elems]))
    dargs = loader.DeviceMemory(drv, 16 * 4)
    dargs.from_host(struct.pack("16i", *([0] * 16)))
    out_weights = loader.DeviceMemory(drv, MAX_HUFW * 4)
    out_count = loader.DeviceMemory(drv, MAX_HUFW * 4)
    out_weights.from_host(bytes(MAX_HUFW * 4))
    out_count.from_host(bytes(MAX_HUFW * 4))

    args = [
        Buf(out_weights.ptr, MAX_HUFW), Buf(out_count.ptr, MAX_HUFW),
        Buf(counters.ptr, 32), Buf(comp_data.ptr, n_u32), Buf(huf_refs.ptr, 1),
        Buf(fse_infos.ptr, 512), Buf(fse_elems_m.ptr, 512), Buf(dargs.ptr, 16),
        ct.c_uint32(0), ct.c_uint32(1),   # tgOffset, workItemCount
    ]
    kparams.check_args("DecompressHuffmanWeights", args)
    fn(grid=(1, 1, 1), block=(8, 1, 1), args=args, shared=0)
    drv.synchronize()

    raw = out_weights.to_host(MAX_HUFW * 4)
    gw = [raw[i] for i in range(MAX_HUFW * 4)]  # bytes
    # masked U8: each weight is one byte in consecutive byte slots of the dword array
    g_bytes = [raw[i] for i in range(len(raw))]
    g_count_raw = out_count.to_host(MAX_HUFW * 4)
    g_count = g_count_raw[0]
    # reconstruct weight sequence from byte-masked buffer (1 byte per dword slot? or packed?)
    # Our U8 buffer layout: byte index b -> dword b>>2, bits (b&3)*8. So g_bytes[i] = byte value.
    g_weights = g_bytes[:g_count]
    print(f"GPU weights count={g_count}")

    ok = True
    for name, w, c in (("CPU", cw, ccount), ("GPU", g_weights, g_count)):
        mism = [i for i in range(min(len(w), len(truth))) if w[i] != truth[i]]
        good = (not mism) and c == len(truth) and len(w) == len(truth)
        ok &= good
        print(f"{'PASS' if good else 'FAIL'}: {name} vs oracle ({len(w)} weights, count={c}, "
              f"oracle={len(truth)}, {len(mism)} mismatch)")
        if mism:
            print(f"  first mismatches (idx, {name}, oracle):", [(i, w[i], truth[i]) for i in mism[:10]])
    print("compared %d weights (GPU)" % (len(g_weights) if ok else 0))
    sys.exit(0 if ok and len(g_weights) > 0 else 1)


if __name__ == "__main__":
    sys.exit(main())
