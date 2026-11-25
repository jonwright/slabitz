"""zstd frames: a header-level analyser and a hand-built frame writer.

The analyser reads frame and block headers, the literals section header and
the sequences section header (counts, table modes, FSE accuracy logs).  It does
not decode data (``tools/edu_oracle.py`` does that, for the Huffman weights
and the sequences).  The writer builds the frames libzstd will not emit on its
own: raw, RLE and compressed blocks with raw or RLE literals and no sequences,
skippable frames, an exact 131072-byte compressed block.
"""

import struct

MAGIC = 0xFD2FB528
LITS = ("raw", "rle", "compressed", "treeless")
MODES = ("predefined", "rle", "fse", "repeat")
PREDEF_LOG = {"ll": 6, "of": 5, "ml": 6}


def _frame_header(data, pos):
    """Parse a frame header at pos; return (info, pos_after)."""
    magic, = struct.unpack_from("<I", data, pos)
    if magic != MAGIC:
        raise ValueError("not a zstd frame at %d" % pos)
    pos += 4
    fhd = data[pos]
    pos += 1
    fcs_flag, single, checksum, dictid = fhd >> 6, (fhd >> 5) & 1, (fhd >> 2) & 1, fhd & 3
    window_log = None
    if not single:
        wd = data[pos]
        pos += 1
        window_log = 10 + (wd >> 3)
    pos += {0: 0, 1: 1, 2: 2, 3: 4}[dictid]
    fcs_size = {0: 1 if single else 0, 1: 2, 2: 4, 3: 8}[fcs_flag]
    fcs = None
    if fcs_size:
        fcs = int.from_bytes(data[pos:pos + fcs_size], "little")
        if fcs_size == 2:
            fcs += 256
        pos += fcs_size
    return {"single": bool(single), "checksum": bool(checksum), "dict": dictid,
            "window_log": window_log, "fcs": fcs, "fcs_size": fcs_size}, pos


def _read_ncount_len(data, pos):
    """Return (accuracy_log, bytes_used) of an FSE normalized-count header."""
    bits = int.from_bytes(data[pos:pos + 8].ljust(8, b"\0"), "little")
    al = (bits & 15) + 5
    bits >>= 4
    nbits_used = 4
    remaining = (1 << al) + 1
    threshold = 1 << al
    nb = al + 1
    sym = 0
    prev0 = False
    cur = pos
    # re-read bits on demand from the byte stream
    bitpos = 4

    def getbits(n):
        nonlocal bitpos
        v = 0
        for i in range(n):
            byte = data[pos + ((bitpos + i) >> 3)]
            v |= ((byte >> ((bitpos + i) & 7)) & 1) << i
        bitpos += n
        return v

    while remaining > 1 and sym < 256:
        if prev0:
            n0 = sym
            while True:
                r = getbits(2)
                n0 += r
                if r != 3:
                    break
            sym = n0
            prev0 = False
            continue
        maxv = (2 * threshold - 1) - remaining
        low = getbits(nb - 1)
        if low < maxv:
            count = low
        else:
            hi = getbits(1)
            count = low + (hi << (nb - 1))
            if count >= threshold:
                count -= maxv
        count -= 1
        remaining -= abs(count)
        sym += 1
        prev0 = count == 0
        while remaining < threshold:
            nb -= 1
            threshold >>= 1
    return al, (bitpos + 7) >> 3


def table_log(weights):
    """Table log from the explicit Huffman weights (the last weight is implied)."""
    tot = sum(1 << (w - 1) for w in weights if w)
    return tot.bit_length()


def analyse(data):
    """Per frame: header info and per block: type, sizes, and for compressed
    blocks the literals and sequences section headers."""
    frames = []
    pos = 0
    while pos + 4 <= len(data):
        magic, = struct.unpack_from("<I", data, pos)
        if (magic & 0xFFFFFFF0) == 0x184D2A50:
            size, = struct.unpack_from("<I", data, pos + 4)
            frames.append({"skippable": True, "bytes": size})
            pos += 8 + size
            continue
        info, pos = _frame_header(data, pos)
        info["blocks"] = []
        while True:
            hdr = data[pos] | (data[pos + 1] << 8) | (data[pos + 2] << 16)
            pos += 3
            last, btype, bsize = hdr & 1, (hdr >> 1) & 3, hdr >> 3
            blk = {"type": ("raw", "rle", "compressed", "reserved")[btype], "size": bsize}
            if btype == 1:
                pos += 1
            elif btype == 2:
                blk.update(_compressed_block(data[pos:pos + bsize]))
                pos += bsize
            else:
                pos += bsize
            info["blocks"].append(blk)
            if last:
                break
        if info["checksum"]:
            pos += 4
        frames.append(info)
    return frames


def _compressed_block(b):
    t = b[0] & 3
    fmt = (b[0] >> 2) & 3
    out = {"lits": LITS[t]}
    if t <= 1:
        if fmt in (0, 2):
            regen, hl = b[0] >> 3, 1
        elif fmt == 1:
            regen, hl = (b[0] | (b[1] << 8)) >> 4, 2
        else:
            regen, hl = (b[0] | (b[1] << 8) | (b[2] << 16)) >> 4, 3
        consumed = hl + (regen if t == 0 else 1)
        out["lit_streams"] = 1
    else:
        if fmt <= 1:
            v = b[0] | (b[1] << 8) | (b[2] << 16)
            regen, csize, hl = (v >> 4) & 0x3FF, (v >> 14) & 0x3FF, 3
        elif fmt == 2:
            v = int.from_bytes(b[:4], "little")
            regen, csize, hl = (v >> 4) & 0x3FFF, (v >> 18) & 0x3FFF, 4
        else:
            v = int.from_bytes(b[:5], "little")
            regen, csize, hl = (v >> 4) & 0x3FFFF, (v >> 22) & 0x3FFFF, 5
        out["lit_streams"] = 1 if fmt == 0 else 4
        out["lit_csize"] = csize
        consumed = hl + csize
        if t == 2:
            hb = b[hl]
            out["huf_weights"] = "fse" if hb < 128 else "direct"
            if hb >= 128:
                nsym = hb - 127
                out["huf_nsym"] = nsym
                nib = []
                for byte in b[hl + 1:hl + 1 + (nsym + 1) // 2]:
                    nib += [byte >> 4, byte & 15]
                out["huf_table_log"] = table_log(nib[:nsym])
    out["lit_regen"] = regen
    p = consumed
    nb = b[p]
    if nb == 0:
        nseq, p = 0, p + 1
    elif nb < 128:
        nseq, p = nb, p + 1
    elif nb < 255:
        nseq, p = ((nb - 128) << 8) + b[p + 1], p + 2
    else:
        nseq, p = b[p + 1] + (b[p + 2] << 8) + 0x7F00, p + 3
    out["nseq"] = nseq
    if nseq:
        mode = b[p]
        p += 1
        names = ("ll", "of", "ml")
        modes = ((mode >> 6) & 3, (mode >> 4) & 3, (mode >> 2) & 3)
        out["modes"] = {n: MODES[m] for n, m in zip(names, modes)}
        logs = {}
        for n, m in zip(names, modes):
            if m == 0:
                logs[n] = PREDEF_LOG[n]
            elif m == 1:
                logs[n] = 0
                p += 1
            elif m == 2:
                al, used = _read_ncount_len(b, p)
                logs[n] = al
                p += used
            else:
                logs[n] = None
        out["accuracy"] = logs
    return out


def block_mix(z):
    """Block-type counts and bytes of the frames in z."""
    out = {"raw": 0, "rle": 0, "compressed": 0, "reserved": 0, "blocks": 0, "frames": 0}
    for f in analyse(z):
        if f.get("skippable"):
            continue
        out["frames"] += 1
        for b in f["blocks"]:
            out[b["type"]] += 1
            out["blocks"] += 1
    return out


# ------------------------------------------------------------- writer
def frame_header(content_size, window_log=None, checksum=False):
    """Frame header with the content size present.  window_log None makes it a
    single-segment frame (window = content size); otherwise a window
    descriptor for 2**window_log follows the header byte."""
    n = content_size
    single = window_log is None
    if single and n <= 255:
        flag, fcs = 0, bytes([n])
    elif 256 <= n <= 65791:
        flag, fcs = 1, struct.pack("<H", n - 256)
    elif n < (1 << 32):
        flag, fcs = 2, struct.pack("<I", n)
    else:
        flag, fcs = 3, struct.pack("<Q", n)
    fhd = (flag << 6) | (int(single) << 5) | (int(checksum) << 2)
    out = struct.pack("<I", MAGIC) + bytes([fhd])
    if not single:
        out += bytes([(window_log - 10) << 3])
    return out + fcs


def block_header(last, btype, size):
    return struct.pack("<I", (size << 3) | (btype << 1) | int(last))[:3]


def raw_block(data, last=False):
    return block_header(last, 0, len(data)) + data


def rle_block(byte, size, last=False):
    return block_header(last, 1, size) + bytes([byte])


def _lit_header(kind, size):
    if size < 32:
        return bytes([(size << 3) | kind])
    if size < 4096:
        return struct.pack("<H", (size << 4) | (1 << 2) | kind)
    return struct.pack("<I", (size << 4) | (3 << 2) | kind)[:3]


def literals_block(lits, last=False):
    """A compressed block with raw literals and no sequences."""
    body = _lit_header(0, len(lits)) + lits + b"\x00"
    return block_header(last, 2, len(body)) + body


def rle_literals_block(byte, size, last=False):
    body = _lit_header(1, size) + bytes([byte]) + b"\x00"
    return block_header(last, 2, len(body)) + body


def skippable(payload, nibble=0):
    return struct.pack("<II", 0x184D2A50 | nibble, len(payload)) + payload
