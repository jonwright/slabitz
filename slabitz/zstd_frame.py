"""Zstd frame parsing: split a compressed frame into zstd blocks.

Only enough of the zstd frame format is implemented to locate block boundaries and
classify each block (raw / RLE / compressed) - the parsing the GPU stays on the DS
side. Used by the test harness to build the per-block inputs that
zstdgpu_ShaderEntry_ParseFrames / _ParseCompressedBlocks expect, and to cross-check
against python-zstandard.

Frame format (little-endian):
    Magic_Number (4 bytes, 0xFD2FB528 / 0x184D2A50 magicless)
    Frame_Header_Descriptor (1 byte): FCS_Field_Size, Single_Segment, etc.
    [Window_Descriptor]
    [Frame_Content_Size]  (per FCS_Field_Size)
    [Dict_ID]
    Block_Header (3 bytes): Last_Block(1) | Block_Type(2) | Block_Size(21)
    Block_Size bytes of data
"""

import struct

ZSTD_MAGIC = 0xFD2FB528
ZSTD_MAGICLESS = 0x184D2A50
BLOCK_TYPE = {0: "raw", 1: "rle", 2: "compressed", 3: "reserved"}
_BLOCK_SIZE_MASK = 0x3FFFFF  # low 21 bits


class ZstdBlock:
    __slots__ = ("type", "type_name", "last", "data")

    def __init__(self, btype, last, data):
        self.type = btype
        self.type_name = BLOCK_TYPE.get(btype, "reserved")
        self.last = last
        self.data = data

    def __repr__(self):
        return f"ZstdBlock(type={self.type_name}, last={self.last}, size={len(self.data)})"


def _fcs_size(desc, single_seg):
    """Returns the number of bytes the Frame_Content_Size field occupies."""
    fcs_flag = (desc >> 6) & 0x3
    if fcs_flag == 0:
        return 1 if single_seg else 0
    return {1: 2, 2: 4, 3: 8}[fcs_flag]


def parse_frame(compressed):
    """Yield ZstdBlock for each block in a single-frame buffer. Returns (frames, blocks)."""
    off = 0
    n = len(compressed)
    frames = []
    blocks = []

    while off < n:
        if off + 4 > n:
            raise ValueError("truncated frame")
        magic = struct.unpack_from("<I", compressed, off)[0]
        off += 4
        if magic == ZSTD_MAGICLESS:
            # skip the magic - this is a magicless (single-segment) frame
            # descriptor still follows; treat as if magic present.
            pass
        elif magic != ZSTD_MAGIC:
            # no magic: treat the buffer as already-starting at a block within a
            # known frame (caller passes block-level data) -> break to caller.
            break

        # Frame_Header_Descriptor
        if off >= n:
            break
        desc = compressed[off]
        off += 1
        single_seg = desc & 0x20
        fcs_flag = (desc >> 6) & 0x3
        has_dict = desc & 0x01
        has_checksum = desc & 0x04
        has_window = (not single_seg) and True

        # Window_Descriptor (only if not single_segment)
        if not single_seg:
            off += 1

        # Frame_Content_Size
        off += _fcs_size(desc, single_seg)

        # Dict_ID
        if has_dict:
            dict_flag = (desc >> 3) & 0x3
            off += {0: 0, 1: 1, 2: 2, 3: 4}[dict_flag]

        frame_start = off
        # Blocks
        while off < n:
            if off + 3 > n:
                raise ValueError("truncated block header")
            bh = struct.unpack_from("<I", compressed, off)[0] & 0xFFFFFF
            off += 3
            last = bh & 1
            btype = (bh >> 1) & 0x3
            bsize = (bh >> 3) & _BLOCK_SIZE_MASK
            if off + bsize > n:
                raise ValueError("truncated block data")
            data = compressed[off:off + bsize]
            off += bsize
            blk = ZstdBlock(btype, last, data)
            blocks.append(blk)
            if last:
                break
        if has_checksum and off + 4 <= n:
            off += 4  # skip xxhash checksum (not needed for block extraction)
        frames.append((frame_start, off))
        if off >= n:
            break

    return frames, blocks


def split_for_parse(compressed):
    """Return a dict of per-block metadata for direct feeding of the DS parse stage.

    Caller can use blocks[i].type_name / .last / .data. No full validation beyond block
    framing; the DS ParseCompressedBlocks will do the deep parse.
    """
    frames, blocks = parse_frame(compressed)
    return {"frames": frames, "blocks": blocks}
