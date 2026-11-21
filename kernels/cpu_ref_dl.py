#!/usr/bin/env python3
"""CPU reference for the DirectStorage zstd literal-decompression stage (MIT).

Transcribes `zstdgpu_ShaderEntry_InitHuffmanTable_And_DecompressLiterals`
(zstdgpu_shaders.h:2811-2899) for one threadgroup that carries one block's Huffman
table: `PreInitHuffmanTableToLds` (2505-2717) builds the Huffman code table, the
Expand step (2860-2881) densifies it into `GS_HuffmanTable`, then
`DecompressHuffmanCompressedLiterals` (2913-3010) decodes the literal bit-streams.

GPU/CPU conventions used (all on the PC/CUDA path):
- tgSize = kzstdgpu_TgSizeX_DecompressLiterals = 32, so with WaveGetLaneCount()=32
  there is a single wave (waveCnt=1, laneCnt=32), which makes the wave intrinsics in
  PreInit reduce to simple group reductions over the 32 lanes.
- LDS regions (dwords) for the literals kernel, base=0:
      CodeAndSymbol [0,256)
      PreInit       [256,256+PRE_INIT_SIZE)   (Bits,BitsMask,RankCount,RankCountPrefix,WeightSum,BitsMax)
      RankIndex     [256+PRE_INIT_SIZE, +RANKS)
      HuffmanTable  [..., ...,1<<(MaxCount_HuffmanWeightBits-1))
- This reference builds the DS HuffmanTable directly (deterministic prefix-sum), which is
  exactly what the (single-wave) GPU group produces. It decodes literals and is validated
  against the educational decoder's decompressed-literal bytes (oracle `LITS` hook).

Constants: MaxCount_HuffmanWeights=256, MaxCount_HuffmanWeightBits=11,
MaxCount_HuffmanWeightRanks=12, MaxCount_HuffmanWeightsOneDigitBits=8,
MaxCount_HuffmanWeightsAllDigitBits=40.
"""

MAX_HUFW_WEIGHTS = 256
WEIGHT_BITS = 11
RANKS = WEIGHT_BITS + 1
ONE_DIGIT_BITS = MAX_HUFW_WEIGHTS // 32            # 8
ALL_DIGIT_BITS = ONE_DIGIT_BITS * 5                # 40

# PreInit LDS region offsets (relative to the PreInit base):
GS_BITS = 0
GS_BITSMASK = 256
GS_RANKCOUNT = GS_BITSMASK + ALL_DIGIT_BITS        # 296
GS_RANKCOUNTPREFIX = GS_RANKCOUNT + RANKS          # 308
GS_WEIGHTSUM = GS_RANKCOUNTPREFIX + RANKS          # 320
GS_BITSMAX = GS_WEIGHTSUM + 1                      # 321
PRE_INIT_SIZE = GS_BITSMAX + 1                     # 322


def _first_bit_hi(v):
    return v.bit_length() - 1


def _first_bit_hi_nonzero(v):
    # DS zstdgpu_FindFirstBitHiU32_Nonzero: returns 32 for v==0, else highest set-bit index.
    return 32 if v == 0 else v.bit_length() - 1


def _count_bits(v):
    return bin(v & 0xFFFFFFFF).count("1")


def _byte_offset_load_u32(data, byte_offset):
    if byte_offset + 4 > len(data):
        return 0
    return int.from_bytes(data[byte_offset:byte_offset + 4], "little")


def _byte_offset_load_u64(data, byte_offset):
    if byte_offset + 8 > len(data):
        return int.from_bytes(data[byte_offset:len(data)], "little")
    return int.from_bytes(data[byte_offset:byte_offset + 8], "little")


def _binsearch(lds, start, count, key, mask):
    # zstdgpu_BinarySearchLds: largest i in [start, start+count) with (lds[i]&mask) <= key
    range_base = start
    range_size = count
    while range_size > 1:
        range_test = range_size >> 1
        range_next = range_base + range_test
        tst = lds[range_next] & mask
        range_base = range_base if key < tst else range_next
        range_size -= range_test
    return range_base


def huffman_bits(weights, weight_count):
    """DS PreInit weight->bits computation.

    weight_count is `weightCntMinusOne` (= HuffmanWeightCount[tableId]): the number of
    FSE-decoded weights. The final (weight_count'th) symbol's weight is implicit.
    Returns (bits[0..weight_count], bits_max).
    """
    wsum = 0
    for i in range(weight_count):
        w = weights[i]
        wsum += (1 << (w - 1)) if w > 0 else 0
    bits_max = _first_bit_hi(wsum) + 1 if wsum > 0 else 0
    leftover = (1 << bits_max) - wsum
    last_weight = _first_bit_hi(leftover) + 1 if leftover > 0 else 0

    bits = []
    n = weight_count + 1
    for i in range(n):
        if i < weight_count:
            w = weights[i]
            bits.append((bits_max + 1 - w) if w > 0 else 0)
        else:
            bits.append(bits_max + 1 - last_weight)
    return bits, bits_max


def preinit_huffman_table(weights, weight_count):
    """DS PreInitHuffmanTableToLds (single wave). Returns
    (bits_max, code_table_size, rank_index, rank_count_prefix, code_and_symbol)."""
    bits, bits_max = huffman_bits(weights, weight_count)
    n = weight_count + 1

    rank_count = [0] * RANKS
    for b in bits:
        rank_count[b] += 1

    # RankIndex / RankCountPrefix over workItemId in [0, bits_max+1):
    rank_index = [0] * RANKS
    rank_count_prefix = [0] * RANKS
    acc_rindex = 0
    acc_rprefix = 0
    for widx in range(bits_max + 1):
        rcid = bits_max - widx
        rc = rank_count[rcid]
        rank_index[widx] = acc_rindex
        rank_count_prefix[widx] = acc_rprefix
        acc_rindex += rc << widx
        acc_rprefix += rc

    code_and_symbol = [0] * MAX_HUFW_WEIGHTS
    for weight_id in range(n):
        b = bits[weight_id]
        prefix = sum(1 for wi in range(weight_id) if bits[wi] == b)
        bits_inv = bits_max - b
        code = rank_index[bits_inv] + (prefix << bits_inv)
        code_pos = rank_count_prefix[bits_inv] + prefix
        code_and_symbol[code_pos] = (weight_id << 24) | (code & 0xFFFFFF)

    code_table_size = rank_count_prefix[bits_max]
    return bits_max, code_table_size, rank_index, rank_count_prefix, code_and_symbol


def expand_huffman_table(bits_max, code_table_size, rank_index, code_and_symbol):
    """DS Expand step: densify CodeAndSymbol into the state-indexed GS_HuffmanTable.

    Returns list of packed dwords, entry i packs (symbol1<<8|bitcnt1)<<16 | (symbol0<<8|bitcnt0)
    for states 2*i (low 16) and 2*i+1 (high 16).
    """
    state_cnt = rank_index[bits_max]
    state_pair_cnt = state_cnt >> 1
    huff = []
    for sp in range(state_pair_cnt):
        s0 = sp << 1
        s1 = s0 + 1
        si0 = _binsearch(code_and_symbol, 0, code_table_size, s0, 0xFFFFFF)
        si1 = _binsearch(code_and_symbol, 0, code_table_size, s1, 0xFFFFFF)
        bci0 = _binsearch(rank_index, 0, bits_max + 1, s0, 0xFFFFFFFF)
        bci1 = _binsearch(rank_index, 0, bits_max + 1, s1, 0xFFFFFFFF)
        sym0 = code_and_symbol[si0] >> 24
        sym1 = code_and_symbol[si1] >> 24
        bc0 = bits_max - bci0
        bc1 = bits_max - bci1
        sab0 = (sym0 << 8) | bc0
        sab1 = (sym1 << 8) | bc1
        huff.append((sab1 << 16) | sab0)
    return huff


def _sample_symbol_and_bitcnt(state, huff_table):
    sab = huff_table[state >> 1] >> ((state & 1) << 4)
    return (sab >> 8) & 0xFF, sab & 0xFF


class HuffmanStream:
    """Port of zstdgpu_HuffmanStream (backward HUF bit reader)."""

    def __init__(self, data, seg_offs, seg_size, max_bits):
        last_byte_idx = seg_offs + seg_size - 1
        self.final_byte_offset = seg_offs & -8
        self.last_byte_offset = last_byte_idx & -8
        data64 = _byte_offset_load_u64(data, self.last_byte_offset)
        oob = 7 - (last_byte_idx & 7)
        oob_bits = oob * 8
        data64 <<= oob_bits
        hi32 = (data64 >> 32) & 0xFFFFFFFF
        non_data_bit_count = 32 - _first_bit_hi_nonzero(hi32)
        data64 <<= non_data_bit_count
        kept_bit_count = 64 - (oob_bits + non_data_bit_count)

        self.data = data
        self.final_byte_offset = self.final_byte_offset
        self.last_byte_offset = self.last_byte_offset
        self.data_spare = 0
        self.num_bits_spare = 0
        self.data0 = data64 & ((1 << 64) - 1)
        self.num_bits0 = 0xFFFFFFFF if self.final_byte_offset == self.last_byte_offset else kept_bit_count
        self.max_bits_per_code = max_bits
        self.s32_minus_max = 32 - max_bits
        self.data1 = 0
        self.needs_fetch_soon = True
        self.conditional_fetch()

    def conditional_fetch(self):
        if (self.needs_fetch_soon & 1) and (self.final_byte_offset < self.last_byte_offset):
            lb = self.last_byte_offset - 8
            self.last_byte_offset = lb
            self.data1 = _byte_offset_load_u64(self.data, lb)
        self.needs_fetch_soon = False

    def get_from_fetched(self):
        if self.num_bits0 < self.max_bits_per_code:
            self.data_spare = (self.data0 >> 32) & 0xFFFFFFFF
            self.num_bits_spare = self.num_bits0
            self.data0 = self.data1
            self.num_bits0 = 0xFFFFFFFF if self.final_byte_offset == self.last_byte_offset else 64
            self.needs_fetch_soon = True
        k = self.s32_minus_max
        d0_hi_shift = k + self.num_bits_spare
        d0_hi = (self.data0 >> 32) & 0xFFFFFFFF
        return ((self.data_spare >> k) | (d0_hi >> d0_hi_shift)) & 0xFFFFFFFF

    def consume(self, actual_bit_count):
        ns = self.num_bits_spare
        data0_consumed = max(0, actual_bit_count - ns)
        self.num_bits_spare = max(0, ns - actual_bit_count)
        self.data_spare = (self.data_spare << actual_bit_count) & 0xFFFFFFFF
        self.num_bits0 -= data0_consumed
        self.data0 = (self.data0 << data0_consumed) & ((1 << 64) - 1)


def decompress_huffman_literals(compressed_data, lit_streams, huff_table, bits_max):
    """Decode one block's literal streams. `lit_streams` is a list of
    (src.offs, src.size, dst.offs, dst.size) in stream order; returns the literal byte buffer
    (length = max dst.offs+dst.size)."""
    total = max((d_offs + d_size) for (_, _, d_offs, d_size) in lit_streams)
    out = bytearray(total)

    for (src_offs, src_size, dst_offs, dst_size) in lit_streams:
        stream = HuffmanStream(compressed_data, src_offs, src_size, bits_max)

        byte_aligned_end = dst_offs + dst_size
        dword_aligned_end = max(dst_offs, byte_aligned_end & ~3)
        dword_aligned_beg = min((dst_offs + 3) & ~3, dword_aligned_end)

        # Head bytes (up to 3) before the first dword-aligned output address.
        byte_cursor = dst_offs
        while byte_cursor < dword_aligned_beg:
            state = stream.get_from_fetched()
            symbol, bitcnt = _sample_symbol_and_bitcnt(state, huff_table)
            out[byte_cursor] = symbol
            byte_cursor += 1
            stream.consume(bitcnt)
            stream.conditional_fetch()

        # Interior: whole dwords (4 decoded symbols each), matching DS statement order
        # (GetFromFetched x3 -> fetch -> GetFromFetched then fetch... the fourth GetFromFetched
        # happens, then the fetch is issued before sampling the 4th symbol).
        dword_idx = dword_aligned_beg >> 2
        dw_end = dword_aligned_end >> 2
        while dword_idx < dw_end:
            dword = 0

            state = stream.get_from_fetched()
            symbol, bitcnt = _sample_symbol_and_bitcnt(state, huff_table)
            dword |= symbol
            stream.consume(bitcnt)

            state = stream.get_from_fetched()
            symbol, bitcnt = _sample_symbol_and_bitcnt(state, huff_table)
            dword |= symbol << 8
            stream.consume(bitcnt)

            state = stream.get_from_fetched()
            symbol, bitcnt = _sample_symbol_and_bitcnt(state, huff_table)
            dword |= symbol << 16
            stream.consume(bitcnt)

            state = stream.get_from_fetched()
            stream.conditional_fetch()
            symbol, bitcnt = _sample_symbol_and_bitcnt(state, huff_table)
            dword |= symbol << 24
            stream.consume(bitcnt)

            base = dword_idx * 4
            out[base] = dword & 0xFF
            out[base + 1] = (dword >> 8) & 0xFF
            out[base + 2] = (dword >> 16) & 0xFF
            out[base + 3] = (dword >> 24) & 0xFF
            dword_idx += 1

        # Tail bytes.
        byte_cursor = dword_aligned_end
        while byte_cursor < byte_aligned_end:
            state = stream.get_from_fetched()
            symbol, bitcnt = _sample_symbol_and_bitcnt(state, huff_table)
            out[byte_cursor] = symbol
            byte_cursor += 1
            stream.consume(bitcnt)
            stream.conditional_fetch()

    return bytes(out)
