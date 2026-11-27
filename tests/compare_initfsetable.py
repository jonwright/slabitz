"""Element-exact GPU-vs-CPU comparison for the InitFseTable kernel.

Builds nothing; expects a prebuilt cubin (tools/build_all.py -> build/ifs/ifs.cubin). For each case it runs the GPU kernel (pure-CUDA
device memory + {data,count} ABI), runs kernels/cpu_ref_ifs.py on the same inputs,
and diffs the valid DTable region element-by-element.

Usage:
    python3 tests/compare_initfsetable.py [build/ifs/ifs.cubin]
"""

import ctypes as ct
import struct
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from slabitz import loader
import kparams  # tests/kparams.py
from kernels.cpu_ref_ifs import ref_initfse_table, MaxCountFseProbs, FseRleTableCount

CUBIN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build/ifs/ifs.cubin")


class Buf(ct.Structure):
    _fields_ = [("data", ct.c_uint64), ("count", ct.c_uint64)]


def run_gpu(drv, fn, probs, fse_info, table_start_index, group_id,
            table_data_start, table_data_count, work_item_count, table_type, tg_offset):
    n_probs = len(probs)
    N = 4096
    infos = [0] * 1024
    ti = table_start_index + group_id
    infos[FseRleTableCount + ti] = fse_info
    counters = [1] + [0] * 15
    dargs = [0] * 16

    # DS FseProbs is bound to the shader as Buffer<int32> on GPU (ZSTDGPU_*_TYPED_BUFFER maps
    # StorageType only on CPU); the shader indexes it by int32 element. Pass probs as dwords.
    pbytes = struct.pack(f"{n_probs}i", *probs)
    probsM = loader.DeviceMemory(drv, max(len(pbytes), 4))
    outM = loader.DeviceMemory(drv, N * 4)
    infosM = loader.DeviceMemory(drv, 1024 * 4)
    cntM = loader.DeviceMemory(drv, 16 * 4)
    dargM = loader.DeviceMemory(drv, 16 * 4)

    probsM.from_host(pbytes)
    infosM.from_host(struct.pack("1024i", *infos))
    cntM.from_host(struct.pack("16i", *counters))
    dargM.from_host(struct.pack("16i", *dargs))
    outM.from_host(struct.pack(f"{N}i", *([-12345678] * N)))

    args = [
        Buf(probsM.ptr, n_probs),
        Buf(outM.ptr, N),
        Buf(infosM.ptr, 1024),
        Buf(cntM.ptr, 16),
        Buf(dargM.ptr, 16),
        ct.c_uint32(tg_offset),
        ct.c_uint32(work_item_count),
        ct.c_uint32(table_type),
        ct.c_uint32(table_start_index),
        ct.c_uint32(table_data_start),
        ct.c_uint32(table_data_count),
    ]
    kparams.check_args("InitFseTable", args)
    fn(grid=(group_id + 1, 1, 1), block=(128, 1, 1), args=args, shared=3212)
    drv.synchronize()
    return list(struct.unpack(f"{N}i", outM.to_host(N * 4)))


def main():
    with open(CUBIN, "rb") as f:
        cubin = f.read()
    drv = loader.Driver(0)
    fn = drv.load(cubin).function("main_0")

    cases = [
        ("C1 single-symbol acc6 freq64", 6, 1, [64] + [0] * 63, 1),
        ("C2 three-symbol acc5", 5, 3, [20, 10, 2] + [0] * 61, 1),
        ("C3 negative-freq acc5", 5, 7, [16, 8, 4, -1, -1, -1, -1] + [0] * 57, 1),
    ]
    ok = True
    total = 0
    for name, acc, pcount, probs, table_type in cases:
        fse_info = ((acc << 8) | pcount) & 0xFFFFFFFF
        table_start_index = 0
        group_id = 0
        table_data_start = 0
        table_data_count = 1 << acc
        gpu = run_gpu(drv, fn, probs, fse_info, table_start_index, group_id,
                      table_data_start, table_data_count, 1, table_type, 0)
        cpu = ref_initfse_table(probs, fse_info, table_start_index, group_id,
                                table_data_start, table_data_count)
        n = table_data_count
        mism = [(j, gpu[j], cpu[j]) for j in range(n) if gpu[j] != cpu[j]]
        total += n
        status = "PASS" if not mism else "FAIL"
        if mism:
            ok = False
        print(f"{status}: {name} ({n} elems, {len(mism)} mismatch)")
        if mism:
            print("   first 10:", mism[:10])
            print("   gpu[:10]:", gpu[:10])
            print("   cpu[:10]:", cpu[:10])
    print("compared %d table elements" % (total if ok else 0))
    sys.exit(0 if ok and total > 0 else 1)


if __name__ == "__main__":
    main()
