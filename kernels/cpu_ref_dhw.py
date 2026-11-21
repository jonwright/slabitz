#!/usr/bin/env python3
"""CPU reference for zstdgpu_ShaderEntry_DecompressHuffmanWeights (DirectStorage, MIT).

Transcribes the `__hlsl_dx_compiler` path: decode the FSE-compressed Huffman weights for one
HufW FSE stream using the Backward_BitBuffer_V0 (bidirectional byte-reversed bit reader) and the
InitFseTable-produced FSE decode table (FseElems). One stream per (thread) HufW table.

The FSE DTable layout (FseElems[offset]) is packed as  symbol | bitcnt<<8 | nstate<<16.
State indexes into the table; decode = state -> elem -> (symbol, bitcnt, nstate); nextstate =
nstate + Get(bits). Bitcnt<0 means end-of-stream.

Constants (zstdgpu_structs.h): FseElemMaxCount_HufW = 1<<7 = 128, MaxCount_HuffmanWeights = 255,
FseRleTableCount = 256.
"""

FSE_RLE_TABLE_COUNT = 256
FSE_ELEM_MAX_COUNT_HUFW = 128
MAX_COUNT_HUFFMAN_WEIGHTS = 255
FSE_PROB_TABLE_INDEX_UNUSED = 0x3FFFFFFF


def compute_fse_data_start_hufw(index_hufw, cmp_block_count):
    return FSE_RLE_TABLE_COUNT + index_hufw * FSE_ELEM_MAX_COUNT_HUFW


def fse_elem_symbol(p):  return p & 0xFF


def fse_elem_bitcnt(p):  return (p >> 8) & 0xFF


def fse_elem_nstate(p):  return p >> 16


def _byte_offset_load_u32(data, byte_offset):
    if byte_offset + 4 > len(data):
        return 0
    return int.from_bytes(data[byte_offset:byte_offset + 4], "little")


def bit_length_nonzero(v):
    # DS FindFirstBitHiU32_Nonzero returns 32 for v==0 (HLSL firstbithigh(0)=-1)
    return 32 if v == 0 else v.bit_length() - 1


class BackwardBufV0:
    """Port of zstdgpu_Backward_BitBuffer_V0 (non-reversed store)."""

    def __init__(self, data, seg_offs, seg_size):
        self.data = data
        self.bytesz = len(data)
        datasz = seg_offs + seg_size
        self.base_byte_offset = seg_offs & -4
        self.last_byte_offset = (datasz - 1) & -4

        bitcnt = (datasz & 3) << 3
        bitmsk = 0xFFFFFFFF if bitcnt == 0 else ((1 << bitcnt) - 1)
        bitbuf = _byte_offset_load_u32(data, self.last_byte_offset) & bitmsk

        bitcnt = bit_length_nonzero(bitbuf)
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


def decompress_huffman_weights(compressed_segment_data, seg_offs, seg_size, fse_dtable,
                               accuracy_log2, cmp_block_count, thread_id=0):
    """Returns (weights list, count)."""
    buf = BackwardBufV0(compressed_segment_data, seg_offs, seg_size)
    state0 = buf.get(accuracy_log2)
    state1 = buf.get(accuracy_log2)

    fse_table_offset = compute_fse_data_start_hufw(thread_id, cmp_block_count)
    huf_table_offset = thread_id * MAX_COUNT_HUFFMAN_WEIGHTS

    huf_weight_index = huf_table_offset
    weights = []
    hwm = MAX_COUNT_HUFFMAN_WEIGHTS - thread_id

    while True:
        offset_state0 = fse_table_offset + state0
        fse_elem0 = fse_dtable[offset_state0]
        weights.append(fse_elem_symbol(fse_elem0))
        huf_weight_index += 1

        bits0 = fse_elem_bitcnt(fse_elem0)
        nextstate0 = fse_elem_nstate(fse_elem0)
        if buf.can_refill(bits0):
            state0 = nextstate0 + buf.get(bits0)
        else:
            weights.append(fse_elem_symbol(fse_dtable[fse_table_offset + state1]))
            break

        offset_state0 = fse_table_offset + state0
        fse_elem1 = fse_dtable[fse_table_offset + state1]
        weights.append(fse_elem_symbol(fse_elem1))
        huf_weight_index += 1

        bits1 = fse_elem_bitcnt(fse_elem1)
        nextstate1 = fse_elem_nstate(fse_elem1)
        if buf.can_refill(bits1):
            state1 = nextstate1 + buf.get(bits1)
        else:
            weights.append(fse_elem_symbol(fse_dtable[offset_state0]))
            break

    # The final symbol appended on stream end is a real weight too (the educational decoder's
    # FSE_decompress_interleaved2 counts it), so the count is len(weights).
    return weights, len(weights)
