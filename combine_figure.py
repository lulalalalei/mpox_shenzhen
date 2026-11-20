from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def combine_panels(panel_files: list[Path], output_dir: Path, stem: str) -> None:
    if not panel_files:
        raise ValueError("At least one figure is required.")
    images = []
    for path in panel_files:
        with Image.open(path) as source:
            images.append(source.convert("RGB"))
    width = max(image.width for image in images)
    gap = 24
    label_width = 0
    heights = [round(image.height * width / image.width) for image in images]
    resized = [image.resize((width, height), Image.Resampling.LANCZOS) for image, height in zip(images, heights)]
    canvas = Image.new("RGB", (width + label_width, sum(heights) + gap * (len(images) - 1)), "white")
    y = 0
    for index, image in enumerate(resized):
        canvas.paste(image, (label_width, y))
        y += image.height + gap
    output_dir.mkdir(parents=True, exist_ok=True)
    canvas.save(output_dir / f"{stem}.png", dpi=(300, 300))
    canvas.save(output_dir / f"{stem}.tif", dpi=(300, 300), compression="tiff_lzw")
    canvas.save(output_dir / f"{stem}.pdf", resolution=300)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", default="figure_1_combined")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    combine_panels(args.panel, args.output_dir, args.stem)


if __name__ == "__main__":
    main()
