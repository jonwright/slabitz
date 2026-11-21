#!/usr/bin/env python3
"""CPU reference for zstdgpu_ShaderEntry_ParseCompressedBlocks (DirectStorage, MIT).

A faithful transcription of the DS parse of a single compressed block (the `__hlsl_dx_compiler`
path). The wave-intrinsic/lookback parts reduce to their single-active-lane (lane 0) values for a
single-compressed-block test, so this is a pure bit-stream parser + FSE-table extracted. Run it
against the GPU ParseCompressedBlocks cubin via tests/compare_parse_blocks.py.

Only the deterministic parse OUTPUTS are modeled (CompressedBlockData, HufRefs, LitRefs,
SeqStreamToRef*, FseInfos, FseProbs, BlockSizePrefix) - not the wave-reduction counters.

Fields mirror the DS SRT structs:
  OffsetAndSize        = (offs, size)
  CompressedBlockData  = (literal.offs, literal.size, litStreamIndex, seqStreamIndex)
  LitStreamInfo        = (src.offs, src.size, dst.offs, dst.size)
"""

# ---- DS constants (zstdgpu_structs.h) ----
MAX_COUNT_FSEPROBS = 64
FSE_RLE_TABLE_COUNT = 256
FSE_PROB_TABLE_INDEX_UNUSED = 0x3FFFFFFF
FSE_PROB_TABLE_INDEX_REPEAT = FSE_PROB_TABLE_INDEX_UNUSED - 1
FSE_ACC_MAX = {"HufW": 7, "LLen": 9, "Offs": 8, "MLen": 9}
TG_SIZE = 32


def _comp_fse_index(kind, index, cmp_block_count):
    if kind == "HufW":
        return FSE_RLE_TABLE_COUNT + index
    if kind == "LLen":
        return FSE_RLE_TABLE_COUNT + cmp_block_count + index
    if kind == "Offs":
        return FSE_RLE_TABLE_COUNT + (2 * cmp_block_count + 1) + index
    if kind == "MLen":
        return FSE_RLE_TABLE_COUNT + (3 * cmp_block_count + 2) + index
    raise ValueError(kind)


def _decode_lit_size(x):
    return x & ~0xC0000000


class ForwardBuf:
    """Port of zstdgpu_Forward_BitBuffer (uint32-element, little-endian)."""

    def __init__(self, data):
        # data: raw BYTES of the (dword-padded) compressed segment
        self._data = data
        self.bytesz = len(data)
        self.datasz = 0
        self.bitbuf = 0
        self.offset = 0
        self.bitcnt = 0

    def _dword(self, idx):
        off = idx * 4
        if off + 4 > len(self._data):
            return 0
        return int.from_bytes(self._data[off:off + 4], "little")

    def init_segment(self, seg_offs, seg_size):
        # InitWithSegment: datasz = offs+size; bytesz = full buffer bytes; then Skip(offs)
        self.datasz = seg_offs + seg_size
        self.skip(seg_offs)

    def refill(self, bitcnt):
        assert bitcnt <= 32
        if self.bitcnt < bitcnt:
            self.bitbuf |= self._dword(self.offset) << self.bitcnt
            self.bitcnt += 32
            self.offset += 1

    def top(self, bitcnt):
        return self.bitbuf & ((1 << bitcnt) - 1) if bitcnt < 64 else self.bitbuf

    def pop(self, bitcnt):
        self.bitbuf >>= bitcnt
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

    def get_byte_offset(self):
        return (self.offset << 2) - (self.bitcnt >> 3)

    def skip(self, bytecnt):
        leftcnt = self.bitcnt >> 3
        skipcnt = min(leftcnt, bytecnt)
        self.pop(skipcnt << 3)
        bytecnt -= skipcnt
        if bytecnt > 0:
            self.offset += bytecnt >> 2
            self.bitcnt = 0
            reqbits = (bytecnt & 3) << 3
            if reqbits > 0:
                self.bitbuf = self._dword(self.offset) >> reqbits
                self.offset += 1
                self.bitcnt = 32 - reqbits

    def byte_align(self):
        self.pop(self.bitcnt & 7)


def _bitfield_extract(x, start, count):
    return (x >> start) & ((1 << count) - 1)


def _find_first_bit_hi(v):
    return v.bit_length() - 1


def _parse_fse_header(buf, fse_infos, fse_probs, table_index, acc_max):
    out_offs = table_index * MAX_COUNT_FSEPROBS - FSE_RLE_TABLE_COUNT * MAX_COUNT_FSEPROBS
    accuracy = 5 + buf.get(4)
    assert accuracy <= acc_max, f"accuracy {accuracy} > {acc_max}"
    remain = 1 << accuracy
    symbol = 0
    while remain > 0 and symbol < MAX_COUNT_FSEPROBS:
        bits = _find_first_bit_hi(remain + 1) + 1
        buf.refill(bits)
        val = buf.top(bits)
        lower_mask = (1 << (bits - 1)) - 1
        threshold = (1 << bits) - 1 - (remain + 1)
        if (val & lower_mask) < threshold:
            buf.pop(bits - 1)
            val = val & lower_mask
        elif val > lower_mask:
            buf.pop(bits)
            val = val - threshold
        else:
            buf.pop(bits)
        prob = val - 1
        remain -= -prob if prob < 0 else prob
        fse_probs[out_offs + symbol] = prob
        symbol += 1
        if prob == 0:
            repeat = buf.get(2)
            while True:
                for _ in range(repeat):
                    if symbol >= MAX_COUNT_FSEPROBS:
                        break
                    fse_probs[out_offs + symbol] = 0
                    symbol += 1
                if repeat == 3:
                    repeat = buf.get(2)
                else:
                    break
    buf.byte_align()
    assert remain == 0 and symbol < MAX_COUNT_FSEPROBS, "corrupt FSE distribution"
    fse_infos[table_index] = symbol | (accuracy << 8)
    return symbol, accuracy


def _order_append_index_single():  # lane0 of a single block -> 0
    return 0


def parse_compressed_block(block_bytes, compressed_buffer_size_bytes, frame_count=1,
                           cmp_block_count=1, per_frame_cmp_block_counts=None,
                           global_block_index=0, fse_alloc_start=None):
    """Parse one compressed block. Returns dict of deterministic outputs.

    fse_alloc_start: optional dict {"LLen","Offs","MLen"} of the shared FSE alloc counters carried
    across blocks in a frame (the DS ParseCompressedBlocks allocates tables via shared counters). It
    is updated in place and also returned as result["fse_alloc_counts"] so a multi-block frame parses
    each block with a monotonically increasing slot base (mirrors the DS's Fse##name counters).
    """
    buf = ForwardBuf(block_bytes)
    buf.init_segment(0, len(block_bytes))

    # block-index-in-frame (inGlobalBlockIndexPerCmpBlock[threadId])
    block_index_in_frame = global_block_index
    # binary search frame index (frame_count=1 => 0)
    frame_index = 0

    buf.refill(2 + 2)
    literal_block_type = buf.get_no_refill(2)
    literal_block_szfmt = buf.get_no_refill(2)

    huf_lit_stream_count = (1 if literal_block_szfmt == 0 else 4) if literal_block_type >= 2 else 0
    huf_lit_stream_start = _order_append_index_single()

    regenerated_size = 0
    out_block_literal = [0, 0]
    lit_stream_index = ~0
    seq_stream_index = ~0
    fse_table_index_hufw = FSE_PROB_TABLE_INDEX_UNUSED
    huf_refs = {}
    lit_refs = {}

    fse_infos = [0] * (FSE_RLE_TABLE_COUNT + 4 * cmp_block_count + 4 * 2)  # generous
    # Sz to hold every FSE prob table in the frame: max allocated index grows with cmp_block_count
    # (LLen=256+cmpbc+i, Offs=256+(2*cmpbc+1)+i, MLen=256+(3*cmpbc+2)+i, i<=cmpbc), and probs are
    # stored indexed by (index-256)*MaxCountFseProbs + symbol.
    fse_probs = [0] * ((3 * cmp_block_count + 3) * MAX_COUNT_FSEPROBS + 16)

    fse_alloc_counts = {
        "HufW": 0, "LLen": 0, "Offs": 0, "MLen": 0,
    }
    if fse_alloc_start is not None:
        fse_alloc_counts.update({k: fse_alloc_start.get(k, 0)
                                 for k in ("LLen", "Offs", "MLen")})

    def alloc_fse_table(name):
        idx = _comp_fse_index(name, fse_alloc_counts[name], cmp_block_count)
        fse_alloc_counts[name] += 1
        _parse_fse_header(buf, fse_infos, fse_probs, idx, FSE_ACC_MAX[name])
        return idx

    compressed_size = 0

    if literal_block_type <= 1:
        # raw / RLE
        if _bitfield_extract(literal_block_szfmt, 0, 1) == 0:
            regenerated_size = (buf.get(8 - 4) << 1) | _bitfield_extract(literal_block_szfmt, 1, 1)
        elif literal_block_szfmt == 1:
            regenerated_size = buf.get(16 - 4)
        else:
            regenerated_size = buf.get(24 - 4)
        if literal_block_type == 0:
            out_block_literal[0] = buf.get_byte_offset()
            out_block_literal[1] = (regenerated_size & ~0xC0000000) | 0x40000000
            buf.skip(regenerated_size)
        else:
            out_block_literal[0] = buf.get(8)
            out_block_literal[1] = (regenerated_size & ~0xC0000000) | 0x80000000
    else:
        # compressed / treeless
        stream_count = 1 if literal_block_szfmt == 0 else 4
        if literal_block_szfmt <= 1:
            buf.refill(10 + 10)
            regenerated_size = buf.get_no_refill(10)
            compressed_size = buf.get_no_refill(10)
        elif literal_block_szfmt == 2:
            buf.refill(14 + 14)
            regenerated_size = buf.get_no_refill(14)
            compressed_size = buf.get_no_refill(14)
        else:
            regenerated_size = buf.get(18)
            compressed_size = buf.get(18)

        regenerated_offset = 0
        out_block_literal[0] = regenerated_offset
        out_block_literal[1] = (regenerated_size & ~0xC0000000) | 0xC0000000
        lit_stream_index = huf_lit_stream_start

        if literal_block_type == 2:
            header_byte = buf.get(8)
            if header_byte < 128:
                fse_prob_offs = buf.get_byte_offset()
                fse_table_index_hufw = alloc_fse_table("HufW")
                # AFTER allocation, DS subtracts ComputeFseIndexHufW(0) to get a compact id
                fse_table_index_hufw -= _comp_fse_index("HufW", 0, cmp_block_count)
                weights_offs = buf.get_byte_offset()
                weights_size = header_byte - (weights_offs - fse_prob_offs)
                buf.skip(weights_size)
                huf_refs[fse_table_index_hufw] = (weights_offs, weights_size)
                # decompressedHuffmanWeightCount[tableId] initialized to 0 (set in later kernel)
                compressed_size -= header_byte + 1
            else:
                cnt = header_byte - 127
                weights_offs = buf.get_byte_offset()
                weights_size = (cnt + 1) >> 1
                buf.skip(weights_size)
                # compact id = compressedBlockCount - 1 - fseIndexHufW(lane0=0-based wgtStream start)
                fse_table_index_hufw = cmp_block_count - 1
                huf_refs[fse_table_index_hufw] = (weights_offs, weights_size)
                # decompressedHuffmanWeightCount[tableId] = cnt
                compressed_size -= weights_size + 1
        else:
            fse_table_index_hufw = FSE_PROB_TABLE_INDEX_REPEAT

        if stream_count == 1:
            lit_refs[huf_lit_stream_start] = (
                buf.get_byte_offset(), compressed_size, regenerated_offset, regenerated_size)
            buf.skip(compressed_size)
        else:
            buf.refill(16 + 16)
            s0 = buf.get_no_refill(16)
            s1 = buf.get_no_refill(16)
            s2 = buf.get(16)
            compressed_size -= 6
            s3 = compressed_size - (s0 + s1 + s2)
            dst_size = (regenerated_size + 3) // 4
            dst_offs = regenerated_offset
            for k, sz in enumerate((s0, s1, s2, s3)):
                lit_refs[huf_lit_stream_start + k] = (
                    buf.get_byte_offset(), sz, dst_offs,
                    dst_size if k < 3 else regenerated_size - dst_size * 3)
                buf.skip(sz)
                dst_offs += dst_size

    block_size_prefix = _decode_lit_size(out_block_literal[1])

    is_huf_lit = (literal_block_type == 2)
    huf_lit_id = 0 if is_huf_lit else 0

    # ---- sequences section ----
    sequence_byte0 = buf.get(8)
    if sequence_byte0 < 128:
        seq_count = sequence_byte0
    elif sequence_byte0 < 255:
        seq_count = ((sequence_byte0 - 128) << 8) + buf.get(8)
    else:
        seq_count = buf.get(16) + 0x7F00

    seq_stream_present = seq_count != 0
    seq_stream_index = _order_append_index_single()
    seq_index = _order_append_index_single()

    fse_llen = FSE_PROB_TABLE_INDEX_UNUSED
    fse_offs = FSE_PROB_TABLE_INDEX_UNUSED
    fse_mlen = FSE_PROB_TABLE_INDEX_UNUSED

    seq_stream_to_ref = None

    if seq_count != 0:
        compression_modes = buf.get(8)
        assert _bitfield_extract(compression_modes, 0, 2) == 0, "reserved bits nonzero"

        fse_modes = {}

        def parse_fse(start_bit, name):
            mode = _bitfield_extract(compression_modes, start_bit, 2)
            fse_modes[name] = mode
            if mode == 2:
                return alloc_fse_table(name)
            if mode == 0:
                return _comp_fse_index(name, 0, cmp_block_count)
            if mode == 1:
                return buf.get(8)
            return FSE_PROB_TABLE_INDEX_REPEAT

        fse_llen = parse_fse(6, "LLen")
        fse_offs = parse_fse(4, "Offs")
        fse_mlen = parse_fse(2, "MLen")

        offs = buf.get_byte_offset()
        seq_stream_to_ref = (offs, buf.datasz - offs)

    return {
        "compressed_blocks": [(out_block_literal[0], out_block_literal[1], lit_stream_index, seq_stream_index)],
        "huf_refs": list(huf_refs.items()),
        "lit_refs": list(lit_refs.items()),
        "seq_stream_to_ref": seq_stream_to_ref,
        "fse_info": {i: fse_infos[i] for i in range(len(fse_infos)) if fse_infos[i]},
        "fse_probs": fse_probs,
        "block_size_prefix": block_size_prefix,
        "seq_fse": {"LLen": fse_llen, "Offs": fse_offs, "MLen": fse_mlen},
        "seq_fse_modes": fse_modes,
        "seq_stream_index": seq_stream_index,
        "fse_alloc_counts": fse_alloc_counts,
    }
