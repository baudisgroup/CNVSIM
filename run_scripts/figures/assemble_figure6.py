#!/usr/bin/env python3
from __future__ import annotations

"""Vertically assemble the final Figure 6 validation panels A and B."""

import argparse
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont


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


def _load(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(path)
    with Image.open(path) as source:
        rgba = source.convert("RGBA")
        background = Image.new("RGBA", rgba.size, WHITE + (255,))
        return Image.alpha_composite(background, rgba).convert("RGB")


def _tight_crop(image: Image.Image, padding: int = 10) -> Image.Image:
    background = Image.new("RGB", image.size, WHITE)
    difference = ImageChops.difference(image, background).convert("L")
    mask = difference.point(lambda value: 255 if value > 4 else 0)
    bounds = mask.getbbox()
    if bounds is None:
        return image
    left, top, right, bottom = bounds
    return image.crop(
        (
            max(0, left - padding),
            max(0, top - padding),
            min(image.width, right + padding),
            min(image.height, bottom + padding),
        )
    )


def _resize_to_width(image: Image.Image, width: int) -> Image.Image:
    if image.width == width:
        return image
    height = round(image.height * width / image.width)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def assemble(
    panel_a_path: Path,
    panel_b_path: Path,
    out_path: Path,
    *,
    max_panel_width: int,
    outer_margin: int,
    label_gutter: int,
    panel_gap: int,
    label_size: int,
    dpi: int,
    overwrite: bool,
) -> tuple[int, int]:
    if out_path.suffix.lower() != ".png":
        raise ValueError("Figure 6 output must be a PNG file")
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {out_path}")
    if min(
        max_panel_width,
        outer_margin,
        label_gutter,
        label_size,
        dpi,
    ) <= 0:
        raise ValueError("Widths, margins, label size, and DPI must be positive")
    if panel_gap < 0:
        raise ValueError("--panel-gap cannot be negative")

    panel_a = _tight_crop(_load(panel_a_path))
    panel_b = _tight_crop(_load(panel_b_path))
    panel_width = min(max_panel_width, panel_a.width, panel_b.width)
    panel_a = _resize_to_width(panel_a, panel_width)
    panel_b = _resize_to_width(panel_b, panel_width)

    canvas_width = 2 * outer_margin + label_gutter + panel_width
    panel_a_y = outer_margin
    panel_b_y = panel_a_y + panel_a.height + panel_gap
    canvas_height = panel_b_y + panel_b.height + outer_margin
    canvas = Image.new("RGB", (canvas_width, canvas_height), WHITE)

    panel_x = outer_margin + label_gutter
    canvas.paste(panel_a, (panel_x, panel_a_y))
    canvas.paste(panel_b, (panel_x, panel_b_y))

    draw = ImageDraw.Draw(canvas)
    font = _font(label_size)
    label_x = outer_margin + max(4, round(label_gutter * 0.08))
    label_y_offset = max(3, round(label_size * 0.06))
    draw.text(
        (label_x, panel_a_y + label_y_offset),
        "A",
        font=font,
        fill=BLACK,
    )
    draw.text(
        (label_x, panel_b_y + label_y_offset),
        "B",
        font=font,
        fill=BLACK,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG", dpi=(dpi, dpi), optimize=True)
    return canvas.size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vertically assemble final Figure 6 panels A and B."
    )
    parser.add_argument("--panels-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--panel-a-name",
        default="A_technical_composition.png",
    )
    parser.add_argument(
        "--panel-b-name",
        default="B_signature_recovery.png",
    )
    parser.add_argument("--max-panel-width", type=int, default=2100)
    parser.add_argument("--outer-margin", type=int, default=42)
    parser.add_argument("--label-gutter", type=int, default=72)
    parser.add_argument("--panel-gap", type=int, default=54)
    parser.add_argument("--label-size", type=int, default=58)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    panels_dir = args.panels_dir.expanduser().resolve()
    out_path = args.out.expanduser().resolve()
    size = assemble(
        panels_dir / args.panel_a_name,
        panels_dir / args.panel_b_name,
        out_path,
        max_panel_width=args.max_panel_width,
        outer_margin=args.outer_margin,
        label_gutter=args.label_gutter,
        panel_gap=args.panel_gap,
        label_size=args.label_size,
        dpi=args.dpi,
        overwrite=args.overwrite,
    )
    print(
        f"[CNVSIM] Figure 6 assembled from panels A and B: "
        f"{out_path} ({size[0]}x{size[1]})"
    )


if __name__ == "__main__":
    main()
