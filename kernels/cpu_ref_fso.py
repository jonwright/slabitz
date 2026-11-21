#!/usr/bin/env python3
"""CPU reference for zstdgpu_ShaderEntry_FinaliseSequenceOffsets (DirectStorage, MIT).

Transcribes zstdgpu_shaders.h:3661-3699 (one thread per sequence): it converts DecompressSequences'
offset-history-*encoded* offsets into the real (zstd-style) byte distance. If the incoming offset
has bit 29 set it is a "repeat relative to the previous sequence stream's last sequence" and is
resolved via DecodeSeqRepeatOffsetAndApplyPreviousOffsets; every offset is then `offset -= 3`.

For a single-block / single-seq-stream frame this reduces to `offset = coded >= 4 ? coded - 3 : ...`,
with the default previous-stream offsets (1,2,3)+3 = (4,7,11).
"""


def decode_repeat_offset_encoded(x):
    return x & (1 << 29)


def decode_repeat_offset(x):
    return (x & (3 << 27)) >> 27


def decode_repeat_offset_subtracted_bytes(x):
    return ~(x | 0xF8000000) & 0xFFFFFFFF


def decode_repeat_offset_and_apply_previous_offsets(offset, prev1, prev2, prev3):
    idx = decode_repeat_offset(offset)
    cnt = decode_repeat_offset_subtracted_bytes(offset)
    offset = prev3 if idx == 3 else (prev2 if idx == 2 else prev1)
    return offset - cnt


def _binary_search(arr, start, count, key):
    # zstdgpu_BinarySearch: largest i in [start, start+count) with arr[i] <= key
    lo, hi = start, start + count
    while lo < hi:
        mid = (lo + hi) >> 1
        if arr[mid] <= key:
            lo = mid + 1
        else:
            hi = mid
    return lo - 1


def finalise_sequence_offsets(in_offsets, seq_count, seq_stream_cnt,
                              per_seq_stream_seq_start, seq_stream_to_block_id,
                              per_frame_block_count_all, frame_cnt,
                              per_frame_seq_stream_min_idx,
                              per_seq_stream_final_offset1,
                              per_seq_stream_final_offset2,
                              per_seq_stream_final_offset3):
    """Returns the resolved real byte offsets (list). Values are also written back in places."""
    out = []
    for seq_idx in range(seq_count):
        offset = in_offsets[seq_idx]
        if decode_repeat_offset_encoded(offset) > 0:
            seq_stream_idx = _binary_search(per_seq_stream_seq_start, 0, seq_stream_cnt, seq_idx)
            block_idx = seq_stream_to_block_id[seq_stream_idx]
            frame_idx = _binary_search(per_frame_block_count_all, 0, frame_cnt, block_idx)
            seq_stream_first = per_frame_seq_stream_min_idx[frame_idx]

            offset1 = 1 + 3
            offset2 = 4 + 3
            offset3 = 8 + 3
            if seq_stream_first != seq_stream_idx:
                prev = seq_stream_idx - 1
                offset1 = per_seq_stream_final_offset1[prev]
                offset2 = per_seq_stream_final_offset2[prev]
                offset3 = per_seq_stream_final_offset3[prev]
            offset = decode_repeat_offset_and_apply_previous_offsets(offset, offset1, offset2, offset3)
        offset -= 3
        out.append(offset & 0xFFFFFFFF)
    return out
