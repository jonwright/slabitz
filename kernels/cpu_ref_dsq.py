#!/usr/bin/env python3
"""CPU reference for zstdgpu_ShaderEntry_DecompressSequences_SingleStream (DirectStorage, MIT).

Transcribes the SingleStream variant (zstdgpu_shaders.h:3300-3444): one thread per seq-stream
decodes that stream's sequences from the sequence-stream segment using the backward bit-buffer and
the LLen/Offs/MLen FSE D-tables already built by InitFseTable. The LDS FSE cache preload is a
no-op for the scalar decoder, so the whole loop is scalar.

GPU/CPU conventions (PC/CUDA, single compressed block, single seq stream):
- seqStreamCnt = counters[Seq_Streams] = 1; cmpBlockCnt = counters[Blocks_CMP] = 1.
- seqRef.src = inSeqStreamToRef[0] = (byte_offs, byte_size) in inCompressedData.
- FSE table indices from the parse: LLen=257, Offs=259, MLen=261; their D-tables live in
  inFseElems at ComputeFseDataStartFromFseIndex*(idx, cmpBlockCnt):
      startLLen = 384 (FseRleTableCount + cmp*128 + 0*512)
      startOffs = 1408 (FseRleTableCount+512 + cmp*(128+512) + 0*256)
      startMLen = 1920 (FseRleTableCount+512+256 + cmp*(128+512+256) + 0*512)
  (FseRleTableCount=256, max HufW=128, LLen=512, Offs=256, MLen=512.)
- dst = GetSequenceStartAndCount: offs = inPerSeqStreamSeqStart[0] = 0, size =
  counters[Seq_Streams_DecodedItems] = 1 (single stream -> last stream).
- outOffs[i] is the offset-history-*encoded* value Write zstdgpu_UpdatePreviousAndRecomputeIncoming
  returns; for a non-repeat offset it equals the raw coded offset the oracle SEQ reports.

Constants: kzstdgpu_SeqOffset_Encoded_BitBase = 29, RepType_BitBase = 27, Encoded_Mask = 1<<29,
RepType_Mask = 3<<27.
"""

SEQ_OFFSET_ENC_BITBASE = 29
SEQ_OFFSET_ENC_MASK = 1 << SEQ_OFFSET_ENC_BITBASE
SEQ_OFFSET_REPTYPE_BITBASE = 27
SEQ_OFFSET_REPTYPE_MASK = 3 << SEQ_OFFSET_REPTYPE_BITBASE

# bit-packed (baseline << 5) | extra_bits — from zstdgpu_shaders.h
SEQ_LLEN_BASELINES_EXTRABITS = [
    0 << 5 | 0, 1 << 5 | 0, 2 << 5 | 0, 3 << 5 | 0,
    4 << 5 | 0, 5 << 5 | 0, 6 << 5 | 0, 7 << 5 | 0,
    8 << 5 | 0, 9 << 5 | 0, 10 << 5 | 0, 11 << 5 | 0,
    12 << 5 | 0, 13 << 5 | 0, 14 << 5 | 0, 15 << 5 | 0,
    16 << 5 | 1, 18 << 5 | 1, 20 << 5 | 1, 22 << 5 | 1,
    24 << 5 | 2, 28 << 5 | 2, 32 << 5 | 3, 40 << 5 | 3,
    48 << 5 | 4, 64 << 5 | 6, 128 << 5 | 7, 256 << 5 | 8,
    512 << 5 | 9, 1024 << 5 | 10, 2048 << 5 | 11, 4096 << 5 | 12,
    8192 << 5 | 13, 16384 << 5 | 14, 32768 << 5 | 15, 65536 << 5 | 16,
]

SEQ_MLEN_BASELINES_EXTRABITS = [
    3 << 5 | 0, 4 << 5 | 0, 5 << 5 | 0, 6 << 5 | 0,
    7 << 5 | 0, 8 << 5 | 0, 9 << 5 | 0, 10 << 5 | 0,
    11 << 5 | 0, 12 << 5 | 0, 13 << 5 | 0, 14 << 5 | 0,
    15 << 5 | 0, 16 << 5 | 0, 17 << 5 | 0, 18 << 5 | 0,
    19 << 5 | 0, 20 << 5 | 0, 21 << 5 | 0, 22 << 5 | 0,
    23 << 5 | 0, 24 << 5 | 0, 25 << 5 | 0, 26 << 5 | 0,
    27 << 5 | 0, 28 << 5 | 0, 29 << 5 | 0, 30 << 5 | 0,
    31 << 5 | 0, 32 << 5 | 0, 33 << 5 | 0, 34 << 5 | 0,
    35 << 5 | 1, 37 << 5 | 1, 39 << 5 | 1, 41 << 5 | 1,
    43 << 5 | 2, 47 << 5 | 2, 51 << 5 | 3, 59 << 5 | 3,
    67 << 5 | 4, 83 << 5 | 4, 99 << 5 | 5, 131 << 5 | 7,
    259 << 5 | 8, 515 << 5 | 9, 1027 << 5 | 10, 2051 << 5 | 11,
    4099 << 5 | 12, 8195 << 5 | 13, 16387 << 5 | 14, 32771 << 5 | 15, 65539 << 5 | 16,
]


def _byte_offset_load_u32(data, byte_offset):
    if byte_offset + 4 > len(data):
        return 0
    return int.from_bytes(data[byte_offset:byte_offset + 4], "little")


def _first_bit_hi_nonzero(v):
    return 32 if v == 0 else v.bit_length() - 1


class BackwardBufV0:
    """Port of zstdgpu_Backward_BitBuffer_V0 (see cpu_ref_dhw)."""

    def __init__(self, data, seg_offs, seg_size):
        self.data = data
        self.bytesz = len(data)
        datasz = seg_offs + seg_size
        self.base_byte_offset = seg_offs & -4
        self.last_byte_offset = (datasz - 1) & -4

        bitcnt = (datasz & 3) << 3
        bitmsk = 0xFFFFFFFF if bitcnt == 0 else ((1 << bitcnt) - 1)
        bitbuf = _byte_offset_load_u32(data, self.last_byte_offset) & bitmsk

        bitcnt = _first_bit_hi_nonzero(bitbuf)
        bitmsk = (1 << bitcnt) - 1
        bitbuf &= bitmsk

        self.bitcnt_last = (seg_offs & 3) << 3
        lobitcnt = self.bitcnt_last if self.base_byte_offset == self.last_byte_offset else 0
        self.bitcnt = bitcnt - lobitcnt
        self.bitbuf = bitbuf >> lobitcnt
        self.had_last_refill = self.base_byte_offset == self.last_byte_offset

    def can_refill(self, bitcnt):
        return True if self.bitcnt >= bitcnt else not self.had_last_refill

    def refill(self, bitcnt):
        if self.bitcnt < bitcnt:
            assert not self.had_last_refill
            self.last_byte_offset -= 4
            next_byte_offset = self.last_byte_offset
            assert next_byte_offset >= self.base_byte_offset
            lobitcnt = 0 if next_byte_offset > self.base_byte_offset else self.bitcnt_last
            hibitcnt = 32 - lobitcnt
            load_value = _byte_offset_load_u32(self.data, next_byte_offset)
            self.bitbuf = (self.bitbuf << hibitcnt) | (load_value >> lobitcnt)
            self.bitcnt += hibitcnt
            self.had_last_refill = not (next_byte_offset > self.base_byte_offset)

    def top(self, bitcnt):
        return self.bitbuf >> (self.bitcnt - bitcnt)

    def pop(self, bitcnt):
        self.bitbuf &= ~(~0 << (self.bitcnt - bitcnt))
        self.bitcnt -= bitcnt

    def get(self, bitcnt):
        self.refill(bitcnt)
        b = self.top(bitcnt)
        self.pop(bitcnt)
        return b

    def get_no_refill(self, bitcnt):
        b = self.top(bitcnt)
        self.pop(bitcnt)
        return b


def _fse_elem_symbol(p):
    return p & 0xFF


def _fse_elem_bitcnt(p):
    return (p >> 8) & 0xFF


def _fse_elem_nstate(p):
    return p >> 16


def _encode_seq_repeat_offset(x):
    assert 1 <= x <= 3
    return (x << 27) | 0x27FFFFFF


def _decode_seq_repeat_offset_encoded(x):
    return x & SEQ_OFFSET_ENC_MASK


def _decode_seq_repeat_offset(x):
    return (x & SEQ_OFFSET_REPTYPE_MASK) >> SEQ_OFFSET_REPTYPE_BITBASE


def _decode_seq_repeat_offset_subtracted_bytes(x):
    return ~(x | 0xF8000000)


def _subtract_byte_from_seq_offset(x):
    return x - 1


def update_previous_and_recompute_incoming(offset1, offset2, offset3, offset, llen):
    """Returns (offset1, offset2, offset3, encoded_offset); mirrors structs.h:862."""
    is_rep = offset <= 3
    offset_adj = offset + (1 if (is_rep and llen == 0) else 0)

    if offset_adj < 3:
        rep_offset = offset1 if offset_adj < 2 else offset2
    elif offset_adj < 4:
        rep_offset = offset3
    else:
        rep_offset = _subtract_byte_from_seq_offset(offset1)
    encoded_offset = rep_offset if is_rep else offset

    offset3 = offset2 if offset_adj >= 3 else offset3
    offset2 = offset1 if offset_adj >= 2 else offset2
    offset1 = encoded_offset
    return offset1, offset2, offset3, encoded_offset


def read_seq_bits_and_decompress(bb, symbol_llen, symbol_offs, symbol_mlen, skip_offs_refill=False):
    symbol_offs_clamped = min(symbol_offs, SEQ_OFFSET_ENC_BITBASE - 1)
    llen_info = SEQ_LLEN_BASELINES_EXTRABITS[symbol_llen]
    mlen_info = SEQ_MLEN_BASELINES_EXTRABITS[symbol_mlen]
    bitcnt_llen = llen_info & 31
    bitcnt_offs = symbol_offs_clamped
    bitcnt_mlen = mlen_info & 31

    if skip_offs_refill:
        bits_offs = bb.get_no_refill(bitcnt_offs)
    else:
        bits_offs = bb.get(bitcnt_offs)

    bits_mlen_llen = bb.get(bitcnt_mlen + bitcnt_llen)
    bits_llen = bits_mlen_llen & ((1 << bitcnt_llen) - 1)
    bits_mlen = bits_mlen_llen >> bitcnt_llen

    out_offs = (1 << symbol_offs_clamped) + bits_offs
    out_mlen = (mlen_info >> 5) + bits_mlen
    out_llen = (llen_info >> 5) + bits_llen
    return out_llen, out_offs, out_mlen


def read_extra_bits_and_update_state(bb, fse_elem_llen, fse_elem_offs, fse_elem_mlen,
                                     state_llen, state_offs, state_mlen):
    restbitcnt_llen = _fse_elem_bitcnt(fse_elem_llen)
    restbitcnt_mlen = _fse_elem_bitcnt(fse_elem_mlen)
    restbitcnt_offs = _fse_elem_bitcnt(fse_elem_offs)

    packed_bits = bb.get(restbitcnt_llen + restbitcnt_mlen + restbitcnt_offs)
    rest_offs = packed_bits & ((1 << restbitcnt_offs) - 1)
    packed_bits >>= restbitcnt_offs
    rest_mlen = packed_bits & ((1 << restbitcnt_mlen) - 1)
    packed_bits >>= restbitcnt_mlen
    rest_llen = packed_bits

    state_llen = _fse_elem_nstate(fse_elem_llen) + rest_llen
    state_mlen = _fse_elem_nstate(fse_elem_mlen) + rest_mlen
    state_offs = _fse_elem_nstate(fse_elem_offs) + rest_offs
    return state_llen, state_offs, state_mlen


def compute_fse_data_start_llen(index_llen, cmp_block_count):
    return 256 + cmp_block_count * 128 + index_llen * 512


def compute_fse_data_start_offs(index_offs, cmp_block_count):
    return (256 + 512) + cmp_block_count * (128 + 512) + index_offs * 256


def compute_fse_data_start_mlen(index_mlen, cmp_block_count):
    return (256 + 512 + 256) + cmp_block_count * (128 + 512 + 256) + index_mlen * 512


def decompress_sequences_single_stream(compressed_data, seq_stream_offs, seq_stream_size,
                                       fse_elems, start_llen, start_offs, start_mlen,
                                       fse_acc_llen, fse_acc_offs, fse_acc_mlen,
                                       out_start, out_count):
    """Returns (out_llen, out_offs, out_mlen, final_offset1, final_offset2, final_offset3)."""
    bb = BackwardBufV0(compressed_data, seq_stream_offs, seq_stream_size)

    init_bitcnt_llen = fse_acc_llen
    init_bitcnt_offs = fse_acc_offs
    init_bitcnt_mlen = fse_acc_mlen

    bb.refill(init_bitcnt_llen + init_bitcnt_offs + init_bitcnt_mlen)
    state_llen = bb.get_no_refill(init_bitcnt_llen)
    state_offs = bb.get_no_refill(init_bitcnt_offs)
    state_mlen = bb.get_no_refill(init_bitcnt_mlen)

    offset1 = _encode_seq_repeat_offset(1)
    offset2 = _encode_seq_repeat_offset(2)
    offset3 = _encode_seq_repeat_offset(3)

    packed_llen = fse_elems[start_llen + state_llen]
    packed_offs = fse_elems[start_offs + state_offs]
    packed_mlen = fse_elems[start_mlen + state_mlen]

    out_llen = []
    out_offs = []
    out_mlen = []
    total_mlen = 0

    i = out_start
    output_end = out_start + out_count
    while True:
        llen, offs, mlen = read_seq_bits_and_decompress(
            bb, _fse_elem_symbol(packed_llen), _fse_elem_symbol(packed_offs),
            _fse_elem_symbol(packed_mlen), skip_offs_refill=False)
        offset1, offset2, offset3, offs = update_previous_and_recompute_incoming(
            offset1, offset2, offset3, offs, llen)

        total_mlen += mlen
        is_last = (i + 1 == output_end)

        if not is_last:
            state_llen, state_offs, state_mlen = read_extra_bits_and_update_state(
                bb, packed_llen, packed_offs, packed_mlen, state_llen, state_offs, state_mlen)
            packed_llen = fse_elems[start_llen + state_llen]
            packed_offs = fse_elems[start_offs + state_offs]
            packed_mlen = fse_elems[start_mlen + state_mlen]

        out_llen.append(llen)
        out_offs.append(offs)
        out_mlen.append(mlen)

        if is_last:
            break
        i += 1

    return out_llen, out_offs, out_mlen, offset1, offset2, offset3, total_mlen
