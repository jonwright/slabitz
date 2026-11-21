"""CPU reference for the DecodeHuffmanWeights (raw 4-bit nibble) kernel.

One thread per Huffman-weight table. Unpacks 4-bit nibble weights stored
byte-aligned: each byte yields two weights (high nibble then low nibble). The
weight bytes are read sequentially from the table's segment in CompressedData,
so byte k reads CompressedData[segment.offs + k] (bits are consumed via the
byte-aligned Forward_BitBuffer_Get(8)).
"""

MAX_WEIGHTS = 256


def _byte_at(data_dwords, byte):
    return (data_dwords[byte >> 2] >> ((byte & 3) << 3)) & 0xFF


def ref_decode_huffman_weights(counters, data_dwords, huf_refs, weight_count, out):
    """counters: 22-element list; huf_refs: per-table (offs, size);
    weight_count: per-table count; out: mutable list of bytes."""
    streams = counters[5]          # HUF_WgtStreams
    cmp_blocks = counters[15]      # Blocks_CMP
    for thread_id in range(streams):
        huf_table = cmp_blocks - 1 - thread_id
        offs, size = huf_refs[huf_table]
        count = weight_count[huf_table]
        huf_table_offset = huf_table * MAX_WEIGHTS

        for j in range(count >> 1):
            b = _byte_at(data_dwords, offs + j)
            out[huf_table_offset + 2 * j + 0] = b >> 4
            out[huf_table_offset + 2 * j + 1] = b & 0xF
        if count & 1:
            b = _byte_at(data_dwords, offs + (count >> 1))
            out[huf_table_offset + count - 1] = b >> 4
    return out
