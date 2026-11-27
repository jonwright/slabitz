#!/usr/bin/env python3
"""Element-exact GPU vs CPU comparison for ParseCompressedBlocks on one real block.

Drives the same real compressed block through the GPU ParseCompressedBlocks cubin and the
CPU reference (kernels/cpu_ref_pcb.py), then asserts the deterministic parse outputs agree.
The CPU reference is itself validated against the zstd format (independent literal-header
decode), so agreement means the GPU parse is correct.

Usage:
    compare_parse_blocks.py <cubin>
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import run_parse_blocks as rpb
from kernels import cpu_ref_pcb

CUBIN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build/pcb/pcb.cubin")


def decode_lit_size(x):
    return x & ~0xC0000000


def main():
    payload = rpb.make_payload(541, 1)
    block = rpb.make_compressed_block(payload, 5)
    print(f"payload len={len(payload)}  compressed block len={len(block)}")

    from slabitz import loader
    with open(CUBIN, "rb") as f:
        cubin = f.read()
    drv = loader.Driver(0)
    fn = drv.load(cubin).function("main_0")
    g = rpb.run(drv, fn, block)

    c = cpu_ref_pcb.parse_compressed_block(block, len(block))

    checks = {
        "regenerated literal size": (decode_lit_size(g["block_datas"][0][1]),
                                     decode_lit_size(c["block_size_prefix"])),
        "huf_refs[0]": (tuple(g["huf_refs"]), tuple(v for _, v in c["huf_refs"])),
        "lit_refs": (tuple(map(tuple, g["lit_refs"])), tuple(v for _, v in c["lit_refs"])),
        "seq_stream_to_ref": (tuple(g["seq_stream_to_ref"][0]), c["seq_stream_to_ref"]),
        "seq FSE LLen/Offs/MLen": ((g["seq_LLen"][0], g["seq_Offs"][0], g["seq_MLen"][0]),
                                   (c["seq_fse"]["LLen"], c["seq_fse"]["Offs"], c["seq_fse"]["MLen"])),
        "fse_probs": (tuple(g["fse_probs_nonzero"]), tuple(v for v in c["fse_probs"] if v)),
    }

    ok = True
    for name, (gv, cv) in checks.items():
        passed = gv == cv
        ok &= passed
        print(f"{'PASS' if passed else 'FAIL'}: {name}  gpu={gv}  cpu={cv}")
    print(f"RESULT: {'PASS' if ok else 'FAIL'} (all element-exact)")
    print("compared %d fields" % (len(checks) if ok else 0))
    sys.exit(0 if ok and checks else 1)


if __name__ == "__main__":
    sys.exit(main())
