"""CPU reference for the MemsetMemcpy kernel (kernels/MemsetMemcpy.slang).

Faithful sequential transcription of Shaders/ZstdGpuMemsetMemcpy.hlsl: for each
linear work item in [0, workItemCount), binary-search the block whose size-prefix
contains it, then either copy one byte out of CompressedData (flags bit0 = RAW)
or fill the output byte with the RLE value stored in blockRef.offs. Output is a
byte stream (uint8), indexed by a byte offset.
"""


def _binary_search(prefix, block_cnt, key):
    """zstdgpu_BinarySearch(prefix, 0, block_cnt, key): largest idx with prefix[idx] <= key."""
    range_base = 0
    range_size = block_cnt
    while range_size > 1:
        range_test = range_size >> 1
        range_next = range_base + range_test
        if key < prefix[range_next]:
            range_base = range_base          # keep
        else:
            range_base = range_next
        range_size -= range_test
    return range_base


def ref_memsetmemcpy(counts, block_size_prefix, blocks_refs, global_block_index,
                     block_dest_offs, compressed_data, work_item_count, flags):
    """Decompress a RAW (flags bit0) or RLE block region into a byte stream.

    counts: sequence with Blocks_RAW=14, Blocks_RLE=15 (or a dict with those keys).
    block_size_prefix: prefix[i] = bytes before block i (prefix[0] = 0).
    blocks_refs: per-block (offs, size).
    global_block_index: per-block index into the global block list.
    block_dest_offs: per-global-block destination byte offset.
    compressed_data: the raw compressed byte buffer (bytes-like, dword-indexed).
    """
    if isinstance(counts, dict):
        block_cnt = counts["Blocks_RAW"] if (flags & 1) else counts["Blocks_RLE"]
    else:
        block_cnt = counts[13] if (flags & 1) else counts[14]  # zstdgpu_Counters: Blocks_RAW=13, Blocks_RLE=14
    # CompressedData is a uint32 buffer on the GPU; index it as dwords (byte_ofs >> 2).
    if isinstance(compressed_data, (bytes, bytearray)):
        dwords = [int.from_bytes(compressed_data[i * 4:i * 4 + 4], "little")
                  for i in range((len(compressed_data) + 3) // 4)]
        compressed_data = dwords
    out_len = max((block_dest_offs[global_block_index[i]] + blocks_refs[i][1]
                   for i in range(block_cnt)), default=0)
    out = bytearray(out_len)
    for i in range(work_item_count):
        block_idx = _binary_search(block_size_prefix, block_cnt, i)
        size = blocks_refs[block_idx][1]
        byte_idx = i - block_size_prefix[block_idx]
        if byte_idx >= size:
            continue
        dst_block_offset = block_dest_offs[global_block_index[block_idx]]
        if flags & 1:
            byte_ofs = blocks_refs[block_idx][0] + byte_idx
            dv = (compressed_data[byte_ofs >> 2] >> ((byte_ofs & 3) << 3)) & 0xff
        else:
            dv = blocks_refs[block_idx][0] & 0xff
        out[dst_block_offset + byte_idx] = dv
    return bytes(out)
