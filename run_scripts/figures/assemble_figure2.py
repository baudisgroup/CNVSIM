#!/usr/bin/env python3
from __future__ import annotations

"""Assemble the final Figure 2 from completed Figure 2A and 2B PNG panels."""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def _font(size: int) -> ImageFont.ImageFont:
    candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        Path("/System/Library/Fonts/Supplemental/Helvetica Bold.ttf"),
        Path("/Library/Fonts/Arial Bold.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _load_panel(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, WHITE + (255,))
        return Image.alpha_composite(background, rgba).convert("RGB")


def _resize_to_width(image: Image.Image, width: int) -> Image.Image:
    if image.width == width:
        return image
    height = round(image.height * width / image.width)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def assemble_figure2(
    panel_a_path: Path,
    panel_b_path: Path,
    out_path: Path,
    *,
    label_gutter: int,
    panel_gap: int,
    label_size: int,
    dpi: int,
    overwrite: bool,
) -> tuple[int, int]:
    if out_path.suffix.lower() != ".png":
        raise ValueError("Figure 2 output must be a PNG file")
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {out_path}")

    panel_a = _load_panel(panel_a_path)
    panel_b = _load_panel(panel_b_path)

    # Avoid enlarging raster text: use the narrower panel as the common width.
    content_width = min(panel_a.width, panel_b.width)
    panel_a = _resize_to_width(panel_a, content_width)
    panel_b = _resize_to_width(panel_b, content_width)

    canvas_width = label_gutter + content_width
    canvas_height = panel_a.height + panel_gap + panel_b.height
    canvas = Image.new("RGB", (canvas_width, canvas_height), WHITE)
    panel_b_top = panel_a.height + panel_gap
    canvas.paste(panel_a, (label_gutter, 0))
    canvas.paste(panel_b, (label_gutter, panel_b_top))

    draw = ImageDraw.Draw(canvas)
    font = _font(label_size)
    label_x = max(8, round(label_gutter * 0.16))
    label_y_offset = max(8, round(label_size * 0.12))
    draw.text((label_x, label_y_offset), "A", font=font, fill=BLACK)
    draw.text(
        (label_x, panel_b_top + label_y_offset),
        "B",
        font=font,
        fill=BLACK,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG", dpi=(dpi, dpi), optimize=True)
    return canvas.size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vertically assemble completed Figure 2A and 2B PNG panels."
    )
    parser.add_argument("--panel-a", type=Path, required=True)
    parser.add_argument("--panel-b", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label-gutter", type=int, default=72)
    parser.add_argument("--panel-gap", type=int, default=28)
    parser.add_argument("--label-size", type=int, default=52)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.label_gutter <= 0:
        raise ValueError("--label-gutter must be positive")
    if args.panel_gap < 0:
        raise ValueError("--panel-gap cannot be negative")
    if args.label_size <= 0:
        raise ValueError("--label-size must be positive")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")

    width, height = assemble_figure2(
        args.panel_a,
        args.panel_b,
        args.out,
        label_gutter=args.label_gutter,
        panel_gap=args.panel_gap,
        label_size=args.label_size,
        dpi=args.dpi,
        overwrite=args.overwrite,
    )
    print("[CNVSIM] Figure 2 assembled")
    print(f"  output: {args.out.resolve()}")
    print(f"  pixels: {width} x {height}")
    print(f"  dpi: {args.dpi}")


if __name__ == "__main__":
    main()
