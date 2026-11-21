"""CPU reference for the DirectStorage zstd `InitFseTable` shader (DEFAULT method).

A faithful, sequential transcription of `zstdgpu_ShaderEntry_InitFseTable`
(zstdgpu_shaders.h:1621-2344) with `ZSTD_BITCNT_NSTATE_METHOD=DEFAULT` and
`IS_MULTI_WAVE=1` on an NVIDIA L40S (WaveGetLaneCount()=32 -> LaneCnt=32,
WaveCnt=4). It reproduces HLSL wave intrinsics and the group-shared (LDS)
atomic/barrier semantics by simulating the 128 threads of one threadgroup in
lockstep.

The GPU kernel and this reference must produce byte-identical packed FSE elements
`symbol | (bitcnt<<8) | (nstate<<16)` for the valid DTable region
`[tbl_data_offset, tbl_data_offset + tbl_data_count)`. The GPU rounds the final
loop up to a multiple of 128 and reads stale LDS in the tail, so only the valid
region is compared.

Phases (all run for every method; DEFAULT branch implements the last two):
  1. Distribute symbol frequencies across threads/waves, accumulate per-group and
     per-wave aggregates, store negative symbols and (symbol<<24|prefix) for
     positive symbols.
  A. Spread: assign a symbol to each of the tblAllDataCount table slots via
     `symIndexStep` striding and a binary search over the compacted positive
     symbol prefix-sums; write the symbol into GS_SpreadSymbols.
  B. Transpose: for each of 8 bit-planes, group-ballot the symbols' bits into
     GS_SymbolBitMasks.
  C. Rank: for each slot, count how many same-symbol slots precede it (bit-mask
     popcount prefix), derive nstate and bitcnt, and emit the packed FSE elem.

Constants match zstdgpu_structs.h / ZstdGpuInitFseTable.hlsl on PC/CUDA.
"""

TgSizeX = 128              # kzstdgpu_TgSizeX_InitFseTable
LaneCnt = 32               # WaveGetLaneCount() on L40S
WaveCnt = TgSizeX // LaneCnt
MaxCountFseProbs = 64     # kzstdgpu_MaxCount_FseProbs
MaxCountFseElems = 512    # kzstdgpu_MaxCount_FseElems
MaxCountFseElemsAllDigitBits = 128
MaxCountFseElemsOneDigitBits = MaxCountFseElemsAllDigitBits // 8  # 16
FseRleTableCount = 256

# LDS region base offsets (sequential layout, base=0):
GS_CompactedPositiveFrqPrefixSumAndSymbols = 0        # len MaxCountFseProbs
GS_SymbolBitMasks = 64                                # len MaxCountFseElemsAllDigitBits
GS_PerWaveDword0 = 192                                # len = worst-case WaveCountMax
GS_PerWaveDword1 = 224
GS_PerWaveDword2 = 256
GS_PerGroupDword0 = 288
GS_PerGroupDword1 = 289
GS_PerGroupDword2 = 290
GS_SpreadSymbols = 291                                # len MaxCountFseElems
LDS_TOTAL = GS_SpreadSymbols + MaxCountFseElems       # 803


def _min_u32(a, b):
    return a if a < b else b


def _align_up(x, a):
    return (x + a - 1) & ~(a - 1)


def _count_bits(v):
    return bin(v & 0xFFFFFFFF).count("1")


def _first_bit_hi(v):
    return v.bit_length() - 1


def _pack_fse_elem(symbol, bitcnt, nstate):
    return (symbol & 0xFF) | ((bitcnt & 0xFF) << 8) | ((nstate & 0xFFFF) << 16)


def _wave_ballot(preds):
    m = 0
    for lane, p in enumerate(preds):
        if p:
            m |= 1 << lane
    return m


def _pred_prefix(preds, lane):
    return sum(1 for j in range(lane) if preds[j])


def _val_prefix(vals, lane):
    return sum(vals[j] for j in range(lane))


def _wave_sub(preds):
    return sum(1 for p in preds if p)


def _wave_subval(vals):
    return sum(vals)


def ref_initfse_table(in_fse_probs, fse_prob_count_and_accuracy_log2,
                      table_start_index=0, group_id=0,
                      table_data_start=0, table_data_count=None):
    """Build one FSE DTable. Returns the valid packed-elem list [0, tbl_data_count).

    in_fse_probs: list[int], symbol frequencies indexed by
        [MaxCountFseProbs*table_index + symbol]; must index every symbol value
        the table references (up to 255).
    fse_prob_count_and_accuracy_log2: (accuracyLog<<8) | probCount.
    """
    table_index = table_start_index + group_id
    frq_data_offset = table_index * MaxCountFseProbs
    tbl_data_offset = table_data_start + group_id * table_data_count

    accuracy_log2 = fse_prob_count_and_accuracy_log2 >> 8
    frq_data_count = _min_u32(fse_prob_count_and_accuracy_log2 & 0xFF, MaxCountFseProbs)
    tbl_data_count = 1 << accuracy_log2
    tbl_all_data_count = _min_u32(tbl_data_count, MaxCountFseElems)
    frq_data_count_aligned = _align_up(frq_data_count, TgSizeX)
    tbl_all_data_count_aligned = _align_up(tbl_all_data_count, TgSizeX)

    lds = [0] * LDS_TOTAL
    threads = list(range(TgSizeX))

    negative_frq_sym_start = tbl_all_data_count
    positive_frq_sym_count = 0
    positive_frq_sum_total = 0
    positive_frq_sym_index_count = 0

    # IS_MULTI_WAVE=1: init the per-group aggregate dwords (thread 0).
    lds[GS_PerGroupDword0] = negative_frq_sym_start
    lds[GS_PerGroupDword1] = positive_frq_sym_count
    lds[GS_PerGroupDword2] = positive_frq_sum_total

    # ---- Loop 1: distribute symbol frequencies across threads ----
    iters1 = frq_data_count_aligned // TgSizeX
    for k in range(iters1):
        syms = [i + TgSizeX * k for i in threads]
        frqs = []
        is_neg = []
        is_pos = []
        pos_frq = []
        for s in syms:
            frq = in_fse_probs[frq_data_offset + s] if s < frq_data_count else 0
            frqs.append(frq)
            is_neg.append(frq == -1)
            is_pos.append(frq > 0)
            pos_frq.append(frq if frq > 0 else 0)

        wave_neg_cnt = [_wave_sub(is_neg[w * LaneCnt:(w + 1) * LaneCnt]) for w in range(WaveCnt)]
        wave_pos_cnt = [_wave_sub(is_pos[w * LaneCnt:(w + 1) * LaneCnt]) for w in range(WaveCnt)]
        wave_pos_sum = [_wave_subval(pos_frq[w * LaneCnt:(w + 1) * LaneCnt]) for w in range(WaveCnt)]

        prev_pg0 = lds[GS_PerGroupDword0]
        prev_pg1 = lds[GS_PerGroupDword1]
        prev_pg2 = lds[GS_PerGroupDword2]

        for w in range(WaveCnt):
            lds[GS_PerWaveDword0 + w] = wave_neg_cnt[w]
            lds[GS_PerWaveDword1 + w] = wave_pos_cnt[w]
            lds[GS_PerWaveDword2 + w] = wave_pos_sum[w]

        # lane0 of each wave atomically updates group aggregates (min/max)
        for w in range(WaveCnt):
            lane0 = w * LaneCnt
            tn = prev_pg0 - sum(lds[GS_PerWaveDword0 + t] for t in range(w))
            tp = prev_pg1 + sum(lds[GS_PerWaveDword1 + t] for t in range(w))
            ts = prev_pg2 + sum(lds[GS_PerWaveDword2 + t] for t in range(w))
            cand_neg = tn - wave_neg_cnt[w]
            cand_pcnt = tp + wave_pos_cnt[w]
            cand_psum = ts + wave_pos_sum[w]
            lds[GS_PerGroupDword0] = min(lds[GS_PerGroupDword0], cand_neg)
            lds[GS_PerGroupDword1] = max(lds[GS_PerGroupDword1], cand_pcnt)
            lds[GS_PerGroupDword2] = max(lds[GS_PerGroupDword2], cand_psum)

        for i in threads:
            w = i // LaneCnt
            lane = i % LaneCnt
            tn = prev_pg0 - sum(lds[GS_PerWaveDword0 + t] for t in range(w))
            tp = prev_pg1 + sum(lds[GS_PerWaveDword1 + t] for t in range(w))
            ts = prev_pg2 + sum(lds[GS_PerWaveDword2 + t] for t in range(w))
            if is_neg[i]:
                neg_idx = tn - _pred_prefix(is_neg[w * LaneCnt:(w + 1) * LaneCnt], lane) - 1
                lds[GS_SpreadSymbols + neg_idx] = syms[i]
            if is_pos[i]:
                pos_idx = tp + _pred_prefix(is_pos[w * LaneCnt:(w + 1) * LaneCnt], lane)
                pos_pref = ts + _val_prefix(pos_frq[w * LaneCnt:(w + 1) * LaneCnt], lane)
                lds[GS_CompactedPositiveFrqPrefixSumAndSymbols + pos_idx] = (
                    (syms[i] << 24) | pos_pref)

    # ---- after loop 1 ----
    negative_frq_sym_start = lds[GS_PerGroupDword0]
    positive_frq_sym_count = lds[GS_PerGroupDword1]
    positive_frq_sum_total = lds[GS_PerGroupDword2]
    lds[GS_PerGroupDword1] = positive_frq_sym_index_count  # = 0

    sym_index_step = (tbl_all_data_count >> 1) + (tbl_all_data_count >> 3) + 3
    sym_index_mask = tbl_all_data_count - 1

    # ---- Loop A: spread symbols into GS_SpreadSymbols ----
    itersA = tbl_all_data_count_aligned // TgSizeX
    pos_sym_index_count = 0
    for k in range(itersA):
        wids = [i + TgSizeX * k for i in threads]
        posidx = [(wids[i] * sym_index_step) & sym_index_mask for i in threads]
        ispos = [posidx[i] < negative_frq_sym_start for i in threads]

        wave_cnt = [_wave_sub(ispos[w * LaneCnt:(w + 1) * LaneCnt]) for w in range(WaveCnt)]
        for w in range(WaveCnt):
            lds[GS_PerWaveDword1 + w] = wave_cnt[w]
        pg1 = lds[GS_PerGroupDword1]  # previous group count
        this_start = [pg1 + sum(lds[GS_PerWaveDword1 + t] for t in range(i // LaneCnt)) for i in threads]
        start = [this_start[i] + _pred_prefix(ispos[i // LaneCnt * LaneCnt:(i // LaneCnt + 1) * LaneCnt], i % LaneCnt) for i in threads]

        new_count = [this_start[i] + _wave_sub([ispos[l] and start[l] < positive_frq_sum_total for l in range(i // LaneCnt * LaneCnt, (i // LaneCnt + 1) * LaneCnt)]) for i in threads]
        for w in range(WaveCnt):
            lds[GS_PerGroupDword1] = max(lds[GS_PerGroupDword1], new_count[w * LaneCnt])

        for i in threads:
            if ispos[i] and start[i] < positive_frq_sum_total:
                compacted_sym_index = 0
                compacted_sym_count = positive_frq_sym_count
                while compacted_sym_count > 1:
                    c_ofs = compacted_sym_count >> 1
                    c_nxt = compacted_sym_index + c_ofs
                    c_prefix = lds[GS_CompactedPositiveFrqPrefixSumAndSymbols + c_nxt] & 0xFFFFFF
                    compacted_sym_index = compacted_sym_index if start[i] < c_prefix else c_nxt
                    compacted_sym_count -= c_ofs
                prefix_and_symbol = lds[GS_CompactedPositiveFrqPrefixSumAndSymbols + compacted_sym_index]
                symbol = prefix_and_symbol >> 24
                lds[GS_SpreadSymbols + posidx[i]] = symbol

    # ---- Loop B: ballot-transpose symbols into GS_SymbolBitMasks ----
    itersB = tbl_all_data_count_aligned // TgSizeX
    for k in range(itersB):
        for i in threads:
            symbol = lds[GS_SpreadSymbols + (i + TgSizeX * k)]
            wave_ofs = (i // LaneCnt) + k * WaveCnt
            for bit_idx in range(8):
                preds = [(((lds[GS_SpreadSymbols + (l + TgSizeX * k)] >> bit_idx) & 1) != 0)
                         for l in range(i // LaneCnt * LaneCnt, (i // LaneCnt + 1) * LaneCnt)]
                b = _wave_ballot(preds)
                off = MaxCountFseElemsOneDigitBits * bit_idx + wave_ofs
                lds[GS_SymbolBitMasks + off] = b

    # ---- Loop C: rank + pack (valid region only; the GPU tail reads stale LDS) ----
    out = {}
    for wid in range(tbl_data_count):
        uint_idx = wid >> 5
        uint_ofs = wid & 0x1F
        symbol = lds[GS_SpreadSymbols + wid]

        prefix = 0
        for ui in range(uint_idx):
            mask = 0xFFFFFFFF
            for bit_idx in range(8):
                bits = lds[GS_SymbolBitMasks + MaxCountFseElemsOneDigitBits * bit_idx + ui]
                mask &= (((((symbol >> bit_idx) & 1) - 1) ^ bits) & 0xFFFFFFFF)
            prefix += _count_bits(mask)
        mask = (1 << uint_ofs) - 1
        for bit_idx in range(8):
            bits = lds[GS_SymbolBitMasks + MaxCountFseElemsOneDigitBits * bit_idx + uint_idx]
            mask &= (((((symbol >> bit_idx) & 1) - 1) ^ bits) & 0xFFFFFFFF)
        prefix += _count_bits(mask)

        prob = in_fse_probs[frq_data_offset + symbol]
        if prob < 0:
            prob = 1
        nstate = prefix + prob
        bitcnt = accuracy_log2 - _first_bit_hi(nstate)
        nstate <<= bitcnt
        nstate -= tbl_all_data_count
        out[wid] = _pack_fse_elem(symbol, bitcnt, nstate)

    return [out[j] for j in range(tbl_data_count)]
