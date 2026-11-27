"""CPU reference for the ComputeDestBlockOffsets kernel.

Faithful sequential transcription of Shaders/ZstdGpuComputeDestBlockOffsets.hlsl:
for each block, find its frame (binary search over per-frame cumulative block
counts), then emit the block's absolute destination byte offset = frame base
offset + (block's global size offset - the frame's first block size offset).

T09: a block that ends past its frame's size, decodes to more than 128 KiB, or
is the frame's last block and ends short of the frame's size gets INVALID and
its frame a status (ref_frame_bounds_status).
"""

INVALID = 0xFFFFFFFF
STATUS_PAST, STATUS_SHORT, STATUS_BLOCK_TOO_LARGE = 0xA07A0007, 0xA07A0008, 0xA07A0009


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
    return ref_frame_bounds_status(block_size_prefix, per_frame_block_count_all, frame_refs,
                                   work_item_count, frame_count)[0]


def ref_frame_bounds_status(block_size_prefix, per_frame_block_count_all, frame_refs,
                            work_item_count, frame_count):
    """Return (per-block dest offsets, {frame_idx: status}) with the T09 bounds."""
    out, status = [], {}
    for block_idx in range(work_item_count):
        frame_idx = _binary_search(per_frame_block_count_all, frame_count, block_idx)
        first_frame_block_idx = per_frame_block_count_all[frame_idx]
        first_frame_block_offset = (block_size_prefix[first_frame_block_idx - 1]
                                    if first_frame_block_idx > 0 else 0)
        block_offset = (block_size_prefix[block_idx - 1] if block_idx > 0 else 0)
        block_size = block_size_prefix[block_idx] - block_offset
        next_first = (per_frame_block_count_all[frame_idx + 1] if frame_idx + 1 < frame_count
                      else work_item_count)
        frame_offs, frame_size = frame_refs[frame_idx]
        in_frame_end = block_offset - first_frame_block_offset + block_size
        st = 0
        if block_size > 128 * 1024:
            st = STATUS_BLOCK_TOO_LARGE
        elif in_frame_end > frame_size:
            st = STATUS_PAST
        elif block_idx + 1 == next_first and in_frame_end != frame_size:
            st = STATUS_SHORT
        if st:
            out.append(INVALID)
            status[frame_idx] = st
        else:
            out.append(frame_offs + block_offset - first_frame_block_offset)
    return out, status
