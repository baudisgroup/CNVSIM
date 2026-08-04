#!/usr/bin/env python3
from __future__ import annotations

"""Assemble the final Figure 4 from the confirmed A, B, and C panels."""

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


def _load_panel(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, WHITE + (255,))
        return Image.alpha_composite(background, rgba).convert("RGB")


def _tight_crop(
    image: Image.Image,
    *,
    threshold: int = 4,
    padding: int = 12,
) -> Image.Image:
    background = Image.new("RGB", image.size, WHITE)
    difference = ImageChops.difference(image, background).convert("L")
    mask = difference.point(lambda value: 255 if value > threshold else 0)
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
    if width <= 0:
        raise ValueError("Target width must be positive")
    height = round(image.height * width / image.width)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _resize_to_height(image: Image.Image, height: int) -> Image.Image:
    if height <= 0:
        raise ValueError("Target height must be positive")
    width = round(image.width * height / image.height)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _draw_panel_label(
    draw: ImageDraw.ImageDraw,
    label: str,
    x: int,
    y: int,
    *,
    font: ImageFont.ImageFont,
) -> None:
    draw.text((x, y), label, font=font, fill=BLACK)


def assemble_figure4(
    panel_a_path: Path,
    panel_b_path: Path,
    panel_c_path: Path,
    out_path: Path,
    *,
    content_width: int,
    outer_margin: int,
    label_gutter: int,
    column_gap: int,
    row_gap: int,
    label_size: int,
    dpi: int,
    overwrite: bool,
) -> tuple[int, int]:
    if out_path.suffix.lower() != ".png":
        raise ValueError("Figure 4 output must be a PNG file")
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {out_path}")

    panel_a = _tight_crop(_load_panel(panel_a_path))
    panel_b = _tight_crop(_load_panel(panel_b_path))
    panel_c = _tight_crop(_load_panel(panel_c_path))

    top_image_width = content_width - column_gap - 2 * label_gutter
    if top_image_width <= 0:
        raise ValueError("Layout leaves no width for the top-row panels")
    top_height = round(
        top_image_width
        / (
            panel_a.width / panel_a.height
            + panel_c.width / panel_c.height
        )
    )
    panel_a = _resize_to_height(panel_a, top_height)
    panel_c = _resize_to_height(panel_c, top_height)

    panel_b_width = content_width - label_gutter
    panel_b = _resize_to_width(panel_b, panel_b_width)

    a_cell_width = label_gutter + panel_a.width
    c_cell_width = label_gutter + panel_c.width
    top_width = a_cell_width + column_gap + c_cell_width
    top_x = outer_margin + (content_width - top_width) // 2
    top_y = outer_margin
    a_x = top_x + label_gutter
    c_cell_x = top_x + a_cell_width + column_gap
    c_x = c_cell_x + label_gutter

    bottom_y = top_y + top_height + row_gap
    b_cell_x = outer_margin
    b_x = b_cell_x + label_gutter

    canvas_width = content_width + 2 * outer_margin
    canvas_height = bottom_y + panel_b.height + outer_margin
    canvas = Image.new("RGB", (canvas_width, canvas_height), WHITE)
    canvas.paste(panel_a, (a_x, top_y))
    canvas.paste(panel_c, (c_x, top_y))
    canvas.paste(panel_b, (b_x, bottom_y))

    draw = ImageDraw.Draw(canvas)
    font = _font(label_size)
    label_y_offset = max(4, round(label_size * 0.08))
    label_x_offset = max(4, round(label_gutter * 0.10))
    _draw_panel_label(
        draw,
        "A",
        top_x + label_x_offset,
        top_y + label_y_offset,
        font=font,
    )
    _draw_panel_label(
        draw,
        "C",
        c_cell_x + label_x_offset,
        top_y + label_y_offset,
        font=font,
    )
    _draw_panel_label(
        draw,
        "B",
        b_cell_x + label_x_offset,
        bottom_y + label_y_offset,
        font=font,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG", dpi=(dpi, dpi), optimize=True)
    return canvas.size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble Figure 4 with A and C on the top row and the "
            "detail-rich B panel at full width below."
        )
    )
    parser.add_argument("--panels-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--panel-a-name",
        default="A_track1_chromosome_landscape.png",
    )
    parser.add_argument(
        "--panel-b-name",
        default="B_track2_subtype_landscape.png",
    )
    parser.add_argument(
        "--panel-c-name",
        default="C_track2_segment_architecture.png",
    )
    parser.add_argument("--content-width", type=int, default=2200)
    parser.add_argument("--outer-margin", type=int, default=30)
    parser.add_argument("--label-gutter", type=int, default=58)
    parser.add_argument("--column-gap", type=int, default=30)
    parser.add_argument("--row-gap", type=int, default=58)
    parser.add_argument("--label-size", type=int, default=52)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name, value in [
        ("--content-width", args.content_width),
        ("--outer-margin", args.outer_margin),
        ("--label-gutter", args.label_gutter),
        ("--column-gap", args.column_gap),
        ("--row-gap", args.row_gap),
        ("--label-size", args.label_size),
        ("--dpi", args.dpi),
    ]:
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    panels_dir = args.panels_dir.expanduser().resolve()
    out_path = args.out.expanduser().resolve()
    width, height = assemble_figure4(
        panels_dir / args.panel_a_name,
        panels_dir / args.panel_b_name,
        panels_dir / args.panel_c_name,
        out_path,
        content_width=args.content_width,
        outer_margin=args.outer_margin,
        label_gutter=args.label_gutter,
        column_gap=args.column_gap,
        row_gap=args.row_gap,
        label_size=args.label_size,
        dpi=args.dpi,
        overwrite=args.overwrite,
    )
    print("[CNVSIM] Figure 4 assembled")
    print(f"  output: {out_path}")
    print(f"  pixels: {width} x {height}")
    print(f"  dpi: {args.dpi}")


if __name__ == "__main__":
    main()
