"""CPU reference for the PrefixSequenceOffsets kernel.

Sequential, full-32-lane-block transcription of the (inline-body) shader:
per 32-lane block, resolve each seq-stream's final offset triple, propagate
non-encoded offsets within the block, and (when a leading lane needs it)
lookback to the nearest preceding block whose offsets were absolute. The
decoupled-lookback cells encode Self (bit30) / Full (bit31) block sums.
"""

ENC = 1 << 29
REP_MASK = 3 << 27
REP_BASE = 27
MASK30 = 0x3FFFFFFF
SELF = 0x40000000
FULL = 0x80000000
FULLMASK = 0xC0000000
DEFAULT_TRIPLE = (4, 7, 11)  # (1,4,8)+3


def _encoded(x):
    return x & ENC


def _decode_single(x, p1, p2, p3):
    idx = (x & REP_MASK) >> REP_BASE
    cnt = (~x) & 0x07FFFFFF
    base = p3 if idx == 3 else (p2 if idx == 2 else p1)
    return (base - cnt) & 0xFFFFFFFF


def _decode_pair(o1, o2, o3, p1, p2, p3):
    if _encoded(o1):
        o1 = _decode_single(o1, p1, p2, p3)
    elif o1 == 0:
        o1 = p1
    if _encoded(o2):
        o2 = _decode_single(o2, p1, p2, p3)
    elif o2 == 0:
        o2 = p2
    if _encoded(o3):
        o3 = _decode_single(o3, p1, p2, p3)
    elif o3 == 0:
        o3 = p3
    return o1, o2, o3


def _enc_self(x):
    return (x & ~FULLMASK) | SELF


def _enc_full(x):
    return (x & ~FULLMASK) | FULL


def _dec_flags(x):
    return x & FULLMASK


def _dec_value(x):
    return x & ~FULLMASK


def _check_full(flags):
    return 1 if flags == FULL else 0


def _offset_or(o):
    return o[0] | o[1] | o[2]


def _propagate_step(os):
    """WAVE_PROPAGATE_STEP over gs in {2,4,8,16,32} for a full 32-lane o-list."""
    os = [tuple(t) for t in os]
    L = len(os)
    for gs in (2, 4, 8, 16, 32):
        if any(_encoded(_offset_or(o)) for o in os):
            old = list(os)
            for j in range(L):
                t = (j & ~(gs - 1)) | (gs // 2 - 1)
                b = old[t] if t < L else old[j]
                if j & (gs // 2):
                    os[j] = _decode_pair(old[j][0], old[j][1], old[j][2], b[0], b[1], b[2])
    return os


def _binary_search(per_frame_block_count_all, frame_cnt, block_idx):
    prev = 0
    while prev < frame_cnt and per_frame_block_count_all[prev + 1] <= block_idx:
        prev += 1
    return prev


def ref_prefix_sequence_offsets(fin1, fin2, fin3, seq_stream_to_block_id,
                                per_frame_block_count_all, per_frame_seq_stream_min_idx,
                                frame_count):
    """fin1/2/3 are mutable lists (in-place). Returns the mutated triples."""
    ns = len(fin1)
    B = (ns + 31) // 32
    lookback = [[0, 0, 0] for _ in range(B)]

    for b in range(B):
        lo, hi = b * 32, min((b + 1) * 32, ns)
        L = hi - lo
        os = [(fin1[lo + j], fin2[lo + j], fin3[lo + j]) for j in range(L)]
        need_prop = [False] * L
        last = L - 1

        for j in range(L):
            i = lo + j
            block_id = seq_stream_to_block_id[i]
            frame_id = _binary_search(per_frame_block_count_all, frame_count, block_id)
            first_in_frame = per_frame_seq_stream_min_idx[frame_id]
            if first_in_frame != 0xFFFFFFFF and first_in_frame == i:
                os[j] = _decode_pair(os[j][0], os[j][1], os[j][2], *DEFAULT_TRIPLE)
                need_prop[j] = True

        # STEP1
        if _encoded(_offset_or(os[last])) == 0:
            for j in range(L):
                need_prop[j] = _encoded(_offset_or(os[j])) > 0
            if last:
                lookback[b] = [_enc_full(os[last][0]), _enc_full(os[last][1]), _enc_full(os[last][2])]
        else:
            p = _propagate_step(os)
            for j in range(L):
                need_prop[j] = _encoded(_offset_or(p[j])) > 0
            if not any(need_prop):
                for j in range(L):
                    fin1[lo + j], fin2[lo + j], fin3[lo + j] = p[j]
            if last:
                if need_prop[last]:
                    lookback[b] = [_enc_self(p[last][0]), _enc_self(p[last][1]), _enc_self(p[last][2])]
                else:
                    lookback[b] = [_enc_full(p[last][0]), _enc_full(p[last][1]), _enc_full(p[last][2])]

        # STEP3 lookback (first lane only)
        if b != 0 and any(need_prop):
            prev = b - 1
            while True:
                lb = lookback[prev]
                flags = _dec_flags(lb[0] & lb[1] & lb[2])
                if flags > 0:
                    pv = (_dec_value(lb[0]), _dec_value(lb[1]), _dec_value(lb[2]))
                    os[0] = _decode_pair(os[0][0], os[0][1], os[0][2], pv[0], pv[1], pv[2])
                    if _check_full(flags) or prev == 0:
                        break
                    prev -= 1
                else:
                    break

        # STEP4
        if any(need_prop):
            p = _propagate_step(os)
            if last:
                lookback[b] = [_enc_full(p[last][0]), _enc_full(p[last][1]), _enc_full(p[last][2])]
            for j in range(L):
                fin1[lo + j], fin2[lo + j], fin3[lo + j] = p[j]
        else:
            # else if (isFirstSequenceStreamInFrame): a first-in-frame lane whose offsets
            # were decoded from defaults must still be written back.
            for j in range(L):
                i = lo + j
                frame_id = _binary_search(per_frame_block_count_all, frame_count, seq_stream_to_block_id[i])
                if per_frame_seq_stream_min_idx[frame_id] == i:
                    fin1[lo + j], fin2[lo + j], fin3[lo + j] = os[j]

    return fin1, fin2, fin3
