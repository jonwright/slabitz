"""CPU reference for the FSE-index PropagateFseIndex kernel.

Sequential faithful transcription of zstdgpu_PropagateFseTableIndex's decoupled
lookback: groups of 32 lanes are "blocks". STEP1/STEP2 write each block's ready
lookback cell (encoded with the bit-31 ready flag); STEP3 resolves a leading
`Repeat` by scanning backward for the nearest ready cell holding a non-Unused
value; STEP4 gang-propagates the resolved value rightward to fill remaining
`Repeat` lanes. Constant lane counts (aligned work_item_count) are assumed, so
every block has full 32 active lanes.
"""

UNUSED = 0x3FFFFFFF
REPEAT = UNUSED - 1
READY = 0x80000000
BLOCK = 32


def _encode(x):
    return (x & ~READY) | READY


def _flags(x):
    return x & READY


def _value(x):
    return x & ~READY


def _propagate(x):
    """WAVE_PROPAGATE_STEP over gs in {2,4,8,16,32} for a 32-lane word.
    Partial blocks (len < 32) only carry the well-defined aligned behaviour for
    the fully-active lanes; a lane read at an out-of-range index is treated as
    the lane's own value (the GPU result there is lane-size/activemask dependent
    and is not a defined comparison, so callers should test aligned counts)."""
    x = list(x)
    L = len(x)
    for gs in (2, 4, 8, 16, 32):
        xold = list(x)
        for j in range(L):
            tgt = (j & ~(gs - 1)) | (gs // 2 - 1)
            bv = xold[tgt] if tgt < L else xold[j]
            if (j & (gs // 2)) and xold[j] >= REPEAT and bv < REPEAT:
                x[j] = bv
    return x


def ref_propagate_fse_index(fse_ids, work_item_count):
    """fse_ids: mutable list of per-seq-stream FSE table indices.
    Returns the propagated list (same length as fse_ids) after all blocks run."""
    n = work_item_count
    B = (n + BLOCK - 1) // BLOCK
    lookback = [0] * B
    res = list(fse_ids)

    for b in range(B):
        lo, hi = b * BLOCK, min((b + 1) * BLOCK, n)
        vals = [res[t] for t in range(lo, hi)]
        L = len(vals)
        any_valid = any(v < REPEAT for v in vals)

        # STEP1/STEP2: store this block's ready lookback cell (exactly one of the two).
        if not any_valid:
            lookback[b] = _encode(UNUSED)
        else:
            x = _propagate(vals)
            lookback[b] = _encode(x[L - 1])

        # STEP3 (always): resolve a leading Repeat from a preceding block.
        lane0 = vals[0]
        needs_lookback = (lane0 == REPEAT) and any(v != UNUSED for v in vals)
        propagated = list(vals)
        if b != 0 and needs_lookback:
            fse_idx_lookback = vals[0]
            prev = b - 1
            while True:
                lbi = lookback[prev]
                if _flags(lbi) > 0:
                    v = _value(lbi)
                    if v == UNUSED:
                        if prev == 0:
                            break
                        prev -= 1
                    else:
                        fse_idx_lookback = v
                        break
            if lane0 != UNUSED:
                propagated[0] = fse_idx_lookback

        # STEP4 (always): gang-propagate to fill remaining Repeat lanes.
        need_prop = [p == REPEAT for p in propagated]
        if any(need_prop):
            y = _propagate(propagated)
            for j in range(L):
                if need_prop[j]:
                    propagated[j] = y[j]

        for j in range(L):
            res[lo + j] = propagated[j]

    return res
