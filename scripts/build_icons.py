"""Generate deterministic application icons for native packagers."""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output
    output.mkdir(parents=True, exist_ok=True)

    size = 1024
    image = Image.new("RGBA", (size, size), "#101722")
    draw = ImageDraw.Draw(image)
    margin = 96
    draw.rounded_rectangle(
        (margin, margin, size - margin, size - margin),
        radius=180,
        fill="#4F8CFF",
    )
    font = ImageFont.truetype(_font_path(), 390)
    text = "VC"
    bounds = draw.textbbox((0, 0), text, font=font)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    draw.text(
        ((size - width) / 2, (size - height) / 2 - bounds[1]),
        text,
        font=font,
        fill="white",
    )
    image.save(output / "VideoCaptioner.png")
    image.save(
        output / "VideoCaptioner.ico",
        sizes=((16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)),
    )
    image.save(output / "VideoCaptioner.icns")
    return 0


def _font_path() -> str:
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise SystemExit("no supported bold font was found")


if __name__ == "__main__":
    raise SystemExit(main())
