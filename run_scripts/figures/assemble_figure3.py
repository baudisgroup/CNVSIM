#!/usr/bin/env python3
from __future__ import annotations

"""Assemble the final Figure 3 from completed workflow and data panels."""

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
    mask = difference.point(
        lambda value: 255 if value > threshold else 0
    )
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
    position: tuple[int, int],
    *,
    font: ImageFont.ImageFont,
) -> None:
    x, y = position
    bounds = draw.textbbox((x, y), label, font=font)
    pad = max(4, round(getattr(font, "size", 24) * 0.08))
    draw.rectangle(
        (
            bounds[0] - pad,
            bounds[1] - pad,
            bounds[2] + pad,
            bounds[3] + pad,
        ),
        fill=WHITE,
    )
    draw.text((x, y), label, font=font, fill=BLACK)


def assemble_figure3(
    workflow_path: Path,
    stability_path: Path,
    coverage_path: Path,
    diagnostic_path: Path,
    out_path: Path,
    *,
    content_width: int,
    outer_margin: int,
    column_gap: int,
    row_gap: int,
    label_size: int,
    dpi: int,
    overwrite: bool,
) -> tuple[int, int]:
    if out_path.suffix.lower() != ".png":
        raise ValueError("Figure 3 output must be a PNG file")
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {out_path}")

    workflow = _tight_crop(_load_panel(workflow_path))
    stability = _tight_crop(_load_panel(stability_path))
    coverage = _tight_crop(_load_panel(coverage_path))
    diagnostic = _tight_crop(_load_panel(diagnostic_path))

    top_height = round(
        (content_width - column_gap)
        / (
            workflow.width / workflow.height
            + stability.width / stability.height
        )
    )
    workflow = _resize_to_height(workflow, top_height)
    stability = _resize_to_height(stability, top_height)
    top_width = workflow.width + column_gap + stability.width

    bottom_width = (content_width - column_gap) // 2
    coverage = _resize_to_width(coverage, bottom_width)
    diagnostic = _resize_to_width(diagnostic, bottom_width)
    bottom_height = max(coverage.height, diagnostic.height)

    canvas_width = content_width + 2 * outer_margin
    canvas_height = (
        2 * outer_margin + top_height + row_gap + bottom_height
    )
    canvas = Image.new("RGB", (canvas_width, canvas_height), WHITE)

    top_x = outer_margin + (content_width - top_width) // 2
    top_y = outer_margin
    stability_x = top_x + workflow.width + column_gap
    canvas.paste(workflow, (top_x, top_y))
    canvas.paste(stability, (stability_x, top_y))

    bottom_y = top_y + top_height + row_gap
    coverage_x = outer_margin
    diagnostic_x = outer_margin + bottom_width + column_gap
    canvas.paste(coverage, (coverage_x, bottom_y))
    canvas.paste(diagnostic, (diagnostic_x, bottom_y))

    draw = ImageDraw.Draw(canvas)
    font = _font(label_size)
    label_offset = max(8, round(label_size * 0.18))
    _draw_panel_label(
        draw,
        "A",
        (top_x + label_offset, top_y + label_offset),
        font=font,
    )
    _draw_panel_label(
        draw,
        "B",
        (coverage_x + label_offset, bottom_y + label_offset),
        font=font,
    )
    _draw_panel_label(
        draw,
        "C",
        (diagnostic_x + label_offset, bottom_y + label_offset),
        font=font,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG", dpi=(dpi, dpi), optimize=True)
    return canvas.size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble Figure 3 with the workflow and stability heatmap "
            "as panel A, reference coverage as B, and diagnostic "
            "heterogeneity as C."
        )
    )
    parser.add_argument("--panels-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--workflow-name",
        default="mermaid_generated.png",
    )
    parser.add_argument(
        "--stability-name",
        default="A_context_stability_heatmap.png",
    )
    parser.add_argument(
        "--coverage-name",
        default="B_reference_coverage.png",
    )
    parser.add_argument(
        "--diagnostic-name",
        default="C_root_diagnostic_heterogeneity.png",
    )
    parser.add_argument("--content-width", type=int, default=3200)
    parser.add_argument("--outer-margin", type=int, default=34)
    parser.add_argument("--column-gap", type=int, default=42)
    parser.add_argument("--row-gap", type=int, default=54)
    parser.add_argument("--label-size", type=int, default=64)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name, value in [
        ("--content-width", args.content_width),
        ("--outer-margin", args.outer_margin),
        ("--column-gap", args.column_gap),
        ("--row-gap", args.row_gap),
        ("--label-size", args.label_size),
        ("--dpi", args.dpi),
    ]:
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    panels_dir = args.panels_dir.expanduser().resolve()
    out_path = args.out.expanduser().resolve()
    width, height = assemble_figure3(
        panels_dir / args.workflow_name,
        panels_dir / args.stability_name,
        panels_dir / args.coverage_name,
        panels_dir / args.diagnostic_name,
        out_path,
        content_width=args.content_width,
        outer_margin=args.outer_margin,
        column_gap=args.column_gap,
        row_gap=args.row_gap,
        label_size=args.label_size,
        dpi=args.dpi,
        overwrite=args.overwrite,
    )
    print("[CNVSIM] Figure 3 assembled")
    print(f"  output: {out_path}")
    print(f"  pixels: {width} x {height}")
    print(f"  dpi: {args.dpi}")


if __name__ == "__main__":
    main()
