#!/usr/bin/env python3
from __future__ import annotations

"""Assemble Figure 7 from projection and experimental-model panels A-D."""

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


def _resize_to_height(image: Image.Image, height: int) -> Image.Image:
    if image.height == height:
        return image
    width = round(image.width * height / image.height)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def assemble(
    panels_dir: Path,
    out_path: Path,
    *,
    page_width: int,
    outer_margin: int,
    label_gutter: int,
    top_gap: int,
    row_gap: int,
    label_size: int,
    dpi: int,
    overwrite: bool,
) -> tuple[int, int]:
    if out_path.suffix.lower() != ".png":
        raise ValueError("Figure 7 output must be a PNG file")
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {out_path}")
    if min(page_width, outer_margin, label_gutter, label_size, dpi) <= 0:
        raise ValueError("Widths, margins, label size, and DPI must be positive")
    if min(top_gap, row_gap) < 0:
        raise ValueError("Panel gaps cannot be negative")

    panel_a = _tight_crop(_load(panels_dir / "A_projection_performance.png"))
    panel_b = _tight_crop(_load(panels_dir / "B_experimental_model_coverage.png"))
    panel_c = _tight_crop(
        _load(panels_dir / "C_tumour_model_concordance.png")
    )
    panel_d = _tight_crop(_load(panels_dir / "D_melanoma_use_case.png"))

    content_width = page_width - 2 * outer_margin - label_gutter
    top_panel_width = (content_width - top_gap) // 2
    panel_a = _resize_to_width(panel_a, top_panel_width)
    panel_b = _resize_to_width(panel_b, top_panel_width)
    common_top_height = max(panel_a.height, panel_b.height)
    panel_a = _resize_to_height(panel_a, common_top_height)
    panel_b = _resize_to_height(panel_b, common_top_height)
    top_width = panel_a.width + top_gap + panel_b.width
    if top_width > content_width:
        scale = content_width / top_width
        panel_a = _resize_to_width(panel_a, round(panel_a.width * scale))
        panel_b = _resize_to_width(panel_b, round(panel_b.width * scale))
        common_top_height = max(panel_a.height, panel_b.height)

    panel_c = _resize_to_width(panel_c, content_width)
    panel_d = _resize_to_width(panel_d, content_width)

    top_y = outer_margin
    panel_c_y = top_y + common_top_height + row_gap
    panel_d_y = panel_c_y + panel_c.height + row_gap
    canvas_height = panel_d_y + panel_d.height + outer_margin
    canvas = Image.new("RGB", (page_width, canvas_height), WHITE)
    panel_x = outer_margin + label_gutter
    canvas.paste(panel_a, (panel_x, top_y))
    canvas.paste(panel_b, (panel_x + panel_a.width + top_gap, top_y))
    canvas.paste(panel_c, (panel_x, panel_c_y))
    canvas.paste(panel_d, (panel_x, panel_d_y))

    draw = ImageDraw.Draw(canvas)
    font = _font(label_size)
    label_x = outer_margin + max(4, round(label_gutter * 0.08))
    offset = max(3, round(label_size * 0.05))
    draw.text((label_x, top_y + offset), "A", font=font, fill=BLACK)
    draw.text(
        (panel_x + panel_a.width + top_gap - label_gutter + 8, top_y + offset),
        "B",
        font=font,
        fill=BLACK,
    )
    draw.text((label_x, panel_c_y + offset), "C", font=font, fill=BLACK)
    draw.text((label_x, panel_d_y + offset), "D", font=font, fill=BLACK)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG", dpi=(dpi, dpi), optimize=True)
    return canvas.size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble final Figure 7 panels A-D.")
    parser.add_argument("--panels-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--page-width", type=int, default=2400)
    parser.add_argument("--outer-margin", type=int, default=42)
    parser.add_argument("--label-gutter", type=int, default=72)
    parser.add_argument("--top-gap", type=int, default=96)
    parser.add_argument("--row-gap", type=int, default=54)
    parser.add_argument("--label-size", type=int, default=58)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    size = assemble(
        args.panels_dir.expanduser().resolve(),
        args.out.expanduser().resolve(),
        page_width=args.page_width,
        outer_margin=args.outer_margin,
        label_gutter=args.label_gutter,
        top_gap=args.top_gap,
        row_gap=args.row_gap,
        label_size=args.label_size,
        dpi=args.dpi,
        overwrite=args.overwrite,
    )
    print(
        f"[CNVSIM] Figure 7 assembled from panels A-D: "
        f"{args.out.expanduser().resolve()} ({size[0]}x{size[1]})"
    )


if __name__ == "__main__":
    main()
