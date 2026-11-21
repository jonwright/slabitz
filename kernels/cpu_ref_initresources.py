"""CPU reference for the InitResources kernel.

Sequential transcription of zstdgpu_ShaderEntry_InitResources:
  stage 0: set counters[0] to the default per-frame struct (thread 0 only).
  stage 1: seed the DEFAULT FSE tables (LLen/Offs/MLen), the 256 dense RLE
           entries, and copy the default FSE probabilities into inoutFseProbs.
"""

RLE_COUNT = 256
MAX_PROBS = 64
LLEN_COUNT = 36
LLEN_ACC = 6
OFFS_COUNT = 29
OFFS_ACC = 5
MLEN_COUNT = 53
MLEN_ACC = 6

_FIELD_ORDER = [
    "FseHufW", "FseLLen", "FseOffs", "FseMLen", "DecompressLiteralsGroups",
    "HUF_WgtStreams", "Seq_Streams_DecodedItems", "HUF_Streams_DecodedBytes",
    "Seq_Streams", "HUF_Streams", "HufLit", "RAW_Streams", "RLE_Streams",
    "Blocks_RAW", "Blocks_RLE", "Blocks_CMP", "BlocksBytes_RAW",
    "BlocksBytes_RLE", "Frames", "Frames_UncompressedByteSize",
    "Frames_AcceptedUncompressedByteSize", "Frames_RequestTooLarge",
]
_STAGE0 = dict.fromkeys(_FIELD_ORDER, 0)
_STAGE0.update(FseLLen=1, FseOffs=1, FseMLen=1)


def _fse_index_llen(index_llen, cmp):
    return 256 + cmp + index_llen


def _fse_index_offs(index_offs, cmp):
    return 256 + (2 * cmp + 1) + index_offs


def _fse_index_mlen(index_mlen, cmp):
    return 256 + (3 * cmp + 2) + index_mlen


def ref_init_resources(in_counters, in_fse_probs_default, in_fse_infos,
                       in_fse_elems, in_fse_probs, stage):
    """Return (counters, fse_infos, fse_elems, fse_probs) after the stage.
    Each input is a mutable list; in_fse_probs_default, in_fse_infos, ... start
    as copies so untouched regions keep their marker pattern."""
    counters = list(in_counters)
    fse_infos = list(in_fse_infos)
    fse_elems = list(in_fse_elems)
    fse_probs = list(in_fse_probs)

    if stage == 0:
        for k, v in _STAGE0.items():
            counters[_FIELD_ORDER.index(k)] = v
        return counters, fse_infos, fse_elems, fse_probs

    cmp = counters[_FIELD_ORDER.index("Blocks_CMP")]

    fse_infos[_fse_index_llen(0, cmp)] = LLEN_COUNT | (LLEN_ACC << 8)
    fse_infos[_fse_index_offs(0, cmp)] = OFFS_COUNT | (OFFS_ACC << 8)
    fse_infos[_fse_index_mlen(0, cmp)] = MLEN_COUNT | (MLEN_ACC << 8)

    for i in range(RLE_COUNT):
        fse_elems[i] = i
        fse_infos[i] = 0

    dst_start = cmp * MAX_PROBS
    dst_stride = cmp * MAX_PROBS + MAX_PROBS
    src_start = 0

    for i in range(LLEN_COUNT):
        fse_probs[dst_start + i] = in_fse_probs_default[src_start + i]
    dst_start += dst_stride
    src_start += LLEN_COUNT

    for i in range(OFFS_COUNT):
        fse_probs[dst_start + i] = in_fse_probs_default[src_start + i]
    dst_start += dst_stride
    src_start += OFFS_COUNT

    for i in range(MLEN_COUNT):
        fse_probs[dst_start + i] = in_fse_probs_default[src_start + i]

    return counters, fse_infos, fse_elems, fse_probs
