"""GPU ExecuteSequences: per-frame input prep, tiling, byte-exact verify, and kernel timing.

This measures ONLY the last stage of the decode pipeline (the LZ77 sequence execution: literal
copies + overlapping match copies into the output). All earlier stages (parse, FSE/Huffman table
init, literal decompression, sequence decompression, offset finalise) are run here as Python CPU
reference code to PREPARE the kernel inputs -- exactly as in tests/run_frame_chain.py. We reuse the
existing harness functions rather than re-deriving them.

The kernel is launched with one 32-thread workgroup per frame (grid = (nFrames, 1, 1)); each
workgroup walks that frame's blocks in order. To build a big batch cheaply we prepare only K
DISTINCT frames with the CPU reference path, then "tile" them N times: we concatenate the K frames
into one batched launch and add per-frame/per-block base offsets (dest, literal, compressed-data,
global block index, cmp ordinal, seq start) so each copy writes its own output region.

Importing this module does not allocate a GPU context; call :func:`run_gpu_bench` to do work.
"""
import os
import sys
import ctypes as ct

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "tests"))

from slabitz import zstd_frame, loader
from kernels import cpu_ref_pcb, cpu_ref_ifs
from tools.edu_oracle import run_oracle
from run_exec_multiblock import decode_block_seqs, global_finalise, Buf, FSE_RLE_TABLE_COUNT

CUBIN = os.path.join(_REPO, "build", "exec", "exec.cubin")

LIT_CMP = 0xC0000000
LIT_RAW = 0x40000000
LIT_RLE = 0x80000000


class _GpuCtx:
    """One CUDA driver context + loaded kernel (no streams/events exposed by loader)."""

    def __init__(self, cubin):
        with open(cubin, "rb") as f:
            self.bin = f.read()
        self.drv = loader.Driver(0)
        self.fn = self.drv.load(self.bin).function("main_0")

    def up(self, host_bytes):
        padded = bytes(host_bytes) + b"\x00" * ((-len(host_bytes)) % 4)
        m = loader.DeviceMemory(self.drv, max(len(padded), 4))
        m.from_host(padded)
        return m, len(padded)

    def up_u32(self, arr):
        a = np.ascontiguousarray(np.asarray(arr, dtype=np.uint32))
        m = loader.DeviceMemory(self.drv, max(len(a.tobytes()), 4))
        if a.size:
            m.from_host(a.tobytes())
        return m, int(a.size)


def prepare_frame(comp, payload):
    """Run the CPU reference chain for ONE frame; return per-frame arrays (frame-relative).

    Reuses the same code path as tests/run_frame_chain.py: decode each compressed block's
    sequences, run the cross-block offset finalise, and collect the literal/sequence/block arrays
    that the ExecuteSequences kernel needs.
    """
    fi, blocks = zstd_frame.parse_frame(comp)
    nblk = len(blocks)
    cmp_blocks = [i for i, b in enumerate(blocks) if b.type_name == "compressed"]
    ncmp = len(cmp_blocks)

    d = run_oracle(comp, expected=payload)
    seq = d["SEQ"]  # one flat ll/ml/off list per cmp block
    lits_hook = d.get("LITS")  # decompressed literal bytes per CMP-literal block, in block order
    assert len(seq) == ncmp, (len(seq), ncmp)
    oracle_offs = [x[0] for x in d["OFFS"]]

    ltypes = set()

    cmpbc = ncmp
    fse_alloc_start = {"LLen": 0, "Offs": 0, "MLen": 0}
    frame_fse = [0] * (256 + cmpbc * 2048 * 8)
    for s in range(FSE_RLE_TABLE_COUNT):
        frame_fse[s] = cpu_ref_ifs._pack_fse_elem(s, 0, 0)

    per_cmp = {}
    infos_order = [None] * ncmp
    starts_order = [0] * ncmp
    blockids_order = [0] * ncmp
    prev = {}
    lit_all, lit_offs_cmp = [], {}
    acc_s = 0
    cmp_data = b""
    ll_all, ml_all = [], []
    si = 0
    lit_seen = 0
    for gi, b in enumerate(blocks):
        if b.type_name != "compressed":
            continue
        ord_ = si
        num_seqs = len(seq[si]) // 3
        info = decode_block_seqs(b.data, num_seqs, cmpbc, fse_alloc_start, prev, frame_fse)
        si += 1
        per_cmp[gi] = info
        infos_order[ord_] = info
        starts_order[ord_] = acc_s
        acc_s += num_seqs
        blockids_order[ord_] = gi
        ltypes.add(info["lit_type"])
        ll_all += info["ll"]
        ml_all += info["ml"]
        if info["lit_type"] == LIT_CMP:
            lit_offs = sum(len(x) for x in lit_all)
            # decompressed literals come straight from the oracle LITS hook (the oracle is the
            # ground truth and avoids the words-soup-only fse_info[256] literal-table lookup)
            if lits_hook is None or lit_seen >= len(lits_hook):
                raise RuntimeError("oracle LITS hook missing a CMP-literal block")
            lits = bytes(lits_hook[lit_seen])
            lit_seen += 1
            lit_all.append(lits)
            lit_offs_cmp[ord_] = lit_offs
        else:
            lit_offs_cmp[ord_] = len(cmp_data) + info["lit_offs_in_block"]
        cmp_data += b.data

    offs_all = global_finalise(infos_order, starts_order, blockids_order)
    total_seqs = len(offs_all)
    mism = (sum(1 for j in range(total_seqs) if offs_all[j] != oracle_offs[j])
            if len(oracle_offs) == total_seqs else None)

    size_prefix, dest_offs = [], []
    acc = 0
    for gi, b in enumerate(blocks):
        sz = per_cmp[gi]["block_decomp"] if b.type_name == "compressed" else len(b.data)
        acc += sz
        size_prefix.append(acc)
        dest_offs.append(acc - sz)
    out_size = len(payload)
    if acc != out_size:
        raise ValueError(f"frame decomp {acc} != payload {out_size}")

    literals = b"".join(lit_all)
    global_block_index_cmp = [cmp_blocks[i] for i in range(ncmp)]
    compressed_blocks = [[lit_offs_cmp[i], per_cmp[cmp_blocks[i]]["lit_type"]
                          | per_cmp[cmp_blocks[i]]["lit_size"], i, i]
                         for i in range(ncmp)]

    raw_fill = []
    for gi, b in enumerate(blocks):
        if b.type_name in ("raw", "rle"):
            raw_fill.append((dest_offs[gi], b.data))

    return {
        "out_size": out_size,
        "literals": literals,
        "cmp_data": cmp_data,
        "size_prefix": size_prefix,
        "dest_offs": dest_offs,
        "global_block_index_cmp": global_block_index_cmp,
        "compressed_blocks": compressed_blocks,
        "seq_starts": starts_order,
        "ll": ll_all,
        "ml": ml_all,
        "offs": offs_all,
        "raw_fill": raw_fill,
        "nblk": nblk,
        "ncmp": ncmp,
        "total_seqs": total_seqs,
        "payload": payload,
        "offset_mismatches": mism,
        "lit_types": ltypes,
    }


def build_tile(frames):
    """Concatenate K prepared frames into ONE tile-block, arrays relative to the tile start.

    Returns a dict describing the single tile block. The tile block's own sizes are:
      t_out_size, t_lit_size, t_comp_size, t_nblocks, t_ncmp, t_seqs
    and arrays (all within-tile):
      size_prefix (len t_nblocks), dest_offs (len t_nblocks),
      global_block_index_cmp (len t_ncmp), compressed_blocks (len t_ncmp*4),
      seq_starts (len t_ncmp), ll/ml/offs (len t_seqs), literals, cmp_data, raw_fill (abs within tile)
    """
    K = len(frames)
    size_prefix, dest_offs, gbi, comp_blk, seq_starts = [], [], [], [], []
    ll, ml, offs = [], [], []
    bytes_written = 0
    literals = bytearray()
    cmp_data = bytearray()
    raw_fill = []
    out_running = 0
    comp_running = 0
    block_running = 0
    cmp_running = 0
    seq_running = 0
    per_frame_ncmp = []
    per_frame_nblocks = []
    per_frame_out = []
    per_frame_comp = []
    for f in frames:
        # literals concatenation base
        lit_base = len(literals)
        # compressed-data concatenation base
        cdata_base = comp_running
        out_base = out_running
        # per-frame block -> in-tile block block index offset
        blk_base = block_running
        cmp_base = cmp_running
        seq_base = seq_running
        per_frame_ncmp.append(f["ncmp"])
        per_frame_nblocks.append(f["nblk"])
        per_frame_out.append(f["out_size"])
        per_frame_comp.append(len(f["cmp_data"]))
        # block prefix offsets (in-tile global)
        fs = f["size_prefix"]
        fd = f["dest_offs"]
        assert len(fs) == f["nblk"] and len(fd) == f["nblk"]
        for w in range(f["nblk"]):
            size_prefix.append(out_base + fs[w])
            dest_offs.append(out_base + fd[w])
        for i, src in enumerate(f["global_block_index_cmp"]):
            gbi.append(blk_base + src)
        for i in range(f["ncmp"]):
            offs0, size0, _li, _si = f["compressed_blocks"][i]
            lit_type = size0 & 0xC0000000
            if lit_type == LIT_CMP:
                offs_g = lit_base + offs0
            else:
                offs_g = cdata_base + offs0
            comp_blk.append([offs_g, size0, cmp_base + i, cmp_base + i])
        for s in f["seq_starts"]:
            seq_starts.append(seq_base + s)
        ll += f["ll"]
        ml += f["ml"]
        offs += f["offs"]
        for (dest, data) in f["raw_fill"]:
            raw_fill.append((out_base + dest, data))
        literals += f["literals"]
        cmp_data += f["cmp_data"]
        bytes_written += f["out_size"]
        out_running += f["out_size"]
        comp_running += len(f["cmp_data"])
        block_running += f["nblk"]
        cmp_running += f["ncmp"]
        seq_running += f["total_seqs"]
    return {
        "size_prefix": size_prefix,
        "dest_offs": dest_offs,
        "global_block_index_cmp": gbi,
        "compressed_blocks": comp_blk,
        "seq_starts": seq_starts,
        "ll": ll, "ml": ml, "offs": offs,
        "literals": bytes(literals),
        "cmp_data": bytes(cmp_data),
        "raw_fill": raw_fill,
        "t_out_size": out_running,
        "t_lit_size": len(literals),
        "t_comp_size": len(cmp_data),
        "t_nblocks": block_running,
        "t_ncmp": cmp_running,
        "t_seqs": seq_running,
        "K": K,
        "per_frame_ncmp": per_frame_ncmp,
        "per_frame_nblocks": per_frame_nblocks,
        "per_frame_out": per_frame_out,
        "per_frame_comp": per_frame_comp,
    }


def expand(tile, N):
    """Repeat one tile-block N times into the global (whole-batch) SRT arrays."""
    nFrames = tile["K"] * N
    per_frame_cmp = []        # len nFrames
    size_prefix, dest_offs = [], []
    gbi, comp_blk, seq_starts = [], [], []
    ll, ml, offs = [], [], []
    raw_fill = []
    literals = bytearray()
    cmp_data = bytearray()
    out_running = 0
    comp_running = 0
    block_running = 0
    cmp_running = 0
    seq_running = 0
    for t in range(N):
        t_out = t * tile["t_out_size"]
        t_lit = t * tile["t_lit_size"]
        t_comp = t * tile["t_comp_size"]
        t_blk = t * tile["t_nblocks"]
        t_cmp = t * tile["t_ncmp"]
        t_seq = t * tile["t_seqs"]
        literals += tile["literals"]
        cmp_data += tile["cmp_data"]
        for f in range(tile["K"]):
            per_frame_cmp.append(cmp_running)
            cmp_running += tile["per_frame_ncmp"][f]
        for i in range(tile["t_nblocks"]):
            size_prefix.append(t_out + tile["size_prefix"][i])
            dest_offs.append(t_out + tile["dest_offs"][i])
        for v in tile["global_block_index_cmp"]:
            gbi.append(t_blk + v)
        for (offs_g, size0, cs, ss) in tile["compressed_blocks"]:
            lit_type = size0 & 0xC0000000
            if lit_type == LIT_CMP:
                offs_gg = t_lit + offs_g
            else:
                offs_gg = t_comp + offs_g
            comp_blk.append([offs_gg, size0, t_cmp + cs, t_cmp + ss])
        for v in tile["seq_starts"]:
            seq_starts.append(t_seq + v)
        ll += tile["ll"]
        ml += tile["ml"]
        offs += tile["offs"]
        for (dest, data) in tile["raw_fill"]:
            raw_fill.append((t_out + dest, data))
        out_running += tile["t_out_size"]
        block_running += tile["t_nblocks"]
        seq_running += tile["t_seqs"]

    total_out = out_running
    per_frame_cmp = np.asarray(per_frame_cmp, dtype=np.uint32)
    return {
        "size_prefix": np.asarray(size_prefix, dtype=np.uint32),
        "dest_offs": np.asarray(dest_offs, dtype=np.uint32),
        "global_block_index_cmp": np.asarray(gbi, dtype=np.uint32),
        "compressed_blocks": np.asarray(comp_blk, dtype=np.uint32),
        "seq_starts": np.asarray(seq_starts, dtype=np.uint32),
        "ll": np.asarray(ll, dtype=np.uint32),
        "ml": np.asarray(ml, dtype=np.uint32),
        "offs": np.asarray(offs, dtype=np.uint32),
        "literals": bytes(literals),
        "cmp_data": bytes(cmp_data),
        "raw_fill": raw_fill,
        "per_frame_cmp": per_frame_cmp,
        "out_size": total_out,
        "nFrames": nFrames,
        "t_nblocks": tile["t_nblocks"] * N,
        "t_ncmp": tile["t_ncmp"] * N,
        "t_seqs": tile["t_seqs"] * N,
    }


def _expected_output(frames, N):
    """The tiled payload (concatenation of K distinct payloads repeated N times)."""
    base = b"".join(f["payload"] for f in frames)
    return base * N


def _time_kernel(ctx, launch, warmup=2, reps=5):
    """Time one GPU launch with CUDA events; returns median seconds of `reps` runs."""
    _cuEvent = _cuda_sym("cuEventCreate", ct.POINTER(ct.c_void_p), ct.c_uint)
    _cuEventRecord = _cuda_sym("cuEventRecord", ct.c_void_p, ct.c_void_p)
    _cuEventSynchronize = _cuda_sym("cuEventSynchronize", ct.c_void_p)
    _cuEventDestroy = _cuda_sym("cuEventDestroy", ct.c_void_p)
    _cuEventElapsedTime = _cuda_sym("cuEventElapsedTime", ct.POINTER(ct.c_float), ct.c_void_p, ct.c_void_p)

    def mk_event():
        e = ct.c_void_p()
        _cuEvent(ct.byref(e), 0)
        return e

    def one_run():
        start, end = mk_event(), mk_event()
        _cuEventRecord(start, None)
        launch()
        _cuEventRecord(end, None)
        _cuEventSynchronize(end)
        ms = ct.c_float()
        _cuEventElapsedTime(ct.byref(ms), start, end)
        _cuEventDestroy(start)
        _cuEventDestroy(end)
        return ms.value / 1000.0

    for _ in range(warmup):
        one_run()
    times = [one_run() for _ in range(reps)]
    return sorted(times)[len(times) // 2]


_cuda_cache = {}


def _cuda_sym(name, *args):
    if name not in _cuda_cache:
        cu = _cuda_cache.get("cu")
        if cu is None:
            cu = ct.CDLL("libcuda.so.1")
            _cuda_cache["cu"] = cu
        f = getattr(cu, name)
        f.restype = ct.c_int
        f.argtypes = list(args)
        _cuda_cache[name] = f
    return _cuda_cache[name]


def _time_wall(ctx, launch, warmup=2, reps=5):
    """Time a GPU launch with a device synchronize + host wall clock (CUDA events gave a fixed
    floor on this driver). Returns the median of ``reps`` runs (seconds). The launch itself is
    async; ``cuCtxSynchronize`` waits for completion, so this measures launch + kernel time, and
    the launch overhead is negligible for the large batches we time. Host<->device copies are
    excluded (all uploads happen before the timed region)."""
    import time
    for _ in range(warmup):
        ctx.drv.synchronize()
        launch()
        ctx.drv.synchronize()
    ts = []
    for _ in range(reps):
        ctx.drv.synchronize()
        t0 = time.perf_counter()
        launch()
        ctx.drv.synchronize()
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2]


def run_gpu_bench(frames, N, cubin=CUBIN, warmup=2, reps=5, verify=True, per_frame=True):
    """Tile K frames N times, verify byte-exact, and time the ExecuteSequences kernel.

    Returns a dict with the timing and verification summary. Only the kernel launch is timed;
    host<->device copies are excluded (all uploads happen before the timed region).

    Dispatch: per_frame=True (default) is the DirectStorage design, one 32-thread workgroup per
    frame, grid.x = nFrames. per_frame=False runs one workgroup over every block serially (a
    single-warp reference). An earlier version of this file claimed grid.x>1 was broken; that was
    a bug in kernels/ExecuteSequences.slang (scalar SV_GroupID), fixed in slabitz a8812cc.
    """
    import time
    ctx = _GpuCtx(cubin)
    tile = build_tile(frames)
    g = expand(tile, N)
    nFrames = g["nFrames"]

    expected = _expected_output(frames, N)

    # ---- upload all buffers (outside the timed region) ----
    comp_mem, comp_cnt = ctx.up(g["cmp_data"])
    lit_mem, lit_cnt = ctx.up(g["literals"] if g["literals"] else b"\x00")
    out_prefill = bytearray(g["out_size"] + (-g["out_size"]) % 16)
    for (dest, data) in g["raw_fill"]:
        out_prefill[dest:dest + len(data)] = data
    out_mem, out_cnt = ctx.up(bytes(out_prefill))

    # Single-workgroup batch: one 32-thread workgroup walks every compressed block.
    counters_vals = [0] * 24
    counters_vals[8] = g["t_ncmp"]   # Seq_Streams
    counters_vals[6] = g["t_seqs"]   # Seq_Streams_DecodedItems
    counters_vals[15] = g["t_ncmp"]  # Blocks_CMP
    # per_frame=True: DS design, one 32-thread workgroup per frame (grid.x = nFrames).
    # per_frame=False: one workgroup walks every block (serial, single-warp reference).
    counters_vals[18] = nFrames if per_frame else 1   # Frames
    ctr_mem, ctr_cnt = ctx.up_u32(counters_vals)
    pfb_mem, pfb_cnt = ctx.up_u32(g["per_frame_cmp"] if per_frame else [0])
    grid_x = nFrames if per_frame else 1
    assert grid_x < (1 << 15), "ConvertTo32BitGroupId needs grid.x < 2^15"

    sp_mem, sp_cnt = ctx.up_u32(g["size_prefix"])
    do_mem, do_cnt = ctx.up_u32(g["dest_offs"])
    ll_mem, ll_cnt = ctx.up_u32(g["ll"])
    ml_mem, ml_cnt = ctx.up_u32(g["ml"])
    of_mem, of_cnt = ctx.up_u32(g["offs"])
    gi_mem, gi_cnt = ctx.up_u32(g["global_block_index_cmp"])
    ss_mem, ss_cnt = ctx.up_u32(g["seq_starts"])
    cb_mem, cb_cnt = ctx.up_u32(g["compressed_blocks"])

    args = [
        Buf(comp_mem.ptr, comp_cnt),
        Buf(lit_mem.ptr, lit_cnt),
        Buf(out_mem.ptr, out_cnt),
        Buf(out_mem.ptr, out_cnt // 4),
        Buf(out_mem.ptr, out_cnt // 16),
        Buf(ctr_mem.ptr, ctr_cnt),
        Buf(pfb_mem.ptr, pfb_cnt),
        Buf(sp_mem.ptr, sp_cnt),
        Buf(do_mem.ptr, do_cnt),
        Buf(ll_mem.ptr, ll_cnt),
        Buf(ml_mem.ptr, ml_cnt),
        Buf(of_mem.ptr, of_cnt),
        Buf(gi_mem.ptr, gi_cnt),
        Buf(ss_mem.ptr, ss_cnt),
        Buf(cb_mem.ptr, cb_cnt),
        ct.c_uint32(lit_cnt),
        ct.c_uint32(g["t_seqs"]),
    ]

    def launch():
        ctx.fn(grid=(grid_x, 1, 1), block=(32, 1, 1), args=args, shared=0)

    # ---- verify (byte-exact) ----
    verified = False
    gpu_out = None
    if verify:
        launch()
        ctx.drv.synchronize()
        raw = out_mem.to_host(out_cnt)
        gpu_out = bytes(raw[:g["out_size"]])
        verified = gpu_out == expected

    # ---- timing (median of reps, after warmup) ----
    seconds_median = _time_wall(ctx, launch, warmup=warmup, reps=reps)

    total_cmp_bytes = sum(f["out_size"] for f in frames) * N
    result = {
        "seconds_median": seconds_median,
        "nFrames": nFrames,
        "total_seqs": g["t_seqs"],
        "total_blocks": g["t_nblocks"],
        "total_ncmp": g["t_ncmp"],
        "total_warps": grid_x,
        "out_bytes": g["out_size"],
        "cmp_bytes": total_cmp_bytes,
        "verified": verified,
        "gpu_out": gpu_out,
        "payload": expected,
        "offset_mismatches": frames[0].get("offset_mismatches"),
        "lit_types": frames[0].get("lit_types"),
    }
    result["gbps_cmp"] = result["cmp_bytes"] / seconds_median / 1e9
    result["gbps_total"] = result["out_bytes"] / seconds_median / 1e9
    # Free the driver context (the loader's Driver has no __del__; leaking it holds device
    # memory and can cause later cuMemAlloc (e.g. nvCOMP/cupy in the same process) to fail).
    try:
        ctx.drv.destroy()
    except Exception:
        pass
    return result
