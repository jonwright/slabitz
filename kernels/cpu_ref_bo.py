"""CPU reference for the ComputeDestBlockOffsets kernel.

Faithful sequential transcription of Shaders/ZstdGpuComputeDestBlockOffsets.hlsl:
for each block, find its frame (binary search over per-frame cumulative block
counts), then emit the block's absolute destination byte offset = frame base
offset + (block's global size offset - the frame's first block size offset).
"""


def _binary_search(per_frame_block_count_all, frame_cnt, block_idx):
    range_base, range_size = 0, frame_cnt
    while range_size > 1:
        range_test = range_size >> 1
        range_next = range_base + range_test
        if block_idx < per_frame_block_count_all[range_next]:
            range_base = range_base
        else:
            range_base = range_next
        range_size -= range_test
    return range_base


def ref_compute_dest_block_offsets(block_size_prefix, per_frame_block_count_all,
                                   frame_refs, work_item_count, frame_count):
    """Return a list of per-block destination byte offsets.

    block_size_prefix: prefix[k] = uncompressed bytes before block k (k in [0, total_blocks)).
    per_frame_block_count_all: cumulative block count at each frame start (len == frame_count+1).
    frame_refs: per-frame (offs, size) — offs is the frame's base dest byte offset.
    """
    out = []
    for block_idx in range(work_item_count):
        frame_idx = _binary_search(per_frame_block_count_all, frame_count, block_idx)
        first_frame_block_idx = per_frame_block_count_all[frame_idx]
        first_frame_block_offset = (block_size_prefix[first_frame_block_idx - 1]
                                    if first_frame_block_idx > 0 else 0)
        block_offset = (block_size_prefix[block_idx - 1] if block_idx > 0 else 0)
        out.append(frame_refs[frame_idx][0] + block_offset - first_frame_block_offset)
    return out
