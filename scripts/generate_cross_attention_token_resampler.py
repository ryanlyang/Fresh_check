from __future__ import annotations

import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH = 1600
HEIGHT = 900
BACKGROUND = "#F7F4EE"
INK = "#1F2937"
MUTED = "#536174"
ARROW = "#5C6B7D"


def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    font_name = "arialbd.ttf" if bold else "arial.ttf"
    font_path = Path("C:/Windows/Fonts") / font_name
    if font_path.exists():
        return ImageFont.truetype(str(font_path), size)
    fallback = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(fallback, size)


TITLE_FONT = load_font(48, bold=True)
SUBTITLE_FONT = load_font(24)
VIEW_FONT = load_font(20, bold=True)
BLOCK_TITLE_FONT = load_font(23, bold=True)
BLOCK_TEXT_FONT = load_font(18)
COUNT_FONT = load_font(21, bold=True)
SMALL_FONT = load_font(16, bold=True)


def centered_text(
    draw: ImageDraw.ImageDraw,
    center: tuple[float, float],
    value: str,
    *,
    font: ImageFont.FreeTypeFont,
    fill: str = INK,
) -> None:
    box = draw.textbbox((0, 0), value, font=font)
    width = box[2] - box[0]
    height = box[3] - box[1]
    draw.text(
        (center[0] - width / 2, center[1] - height / 2 - box[1]),
        value,
        font=font,
        fill=fill,
    )


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    width: int = 4,
) -> None:
    draw.line((*start, *end), fill=ARROW, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    head_size = 14
    left = (
        end[0] - head_size * math.cos(angle - math.pi / 6),
        end[1] - head_size * math.sin(angle - math.pi / 6),
    )
    right = (
        end[0] - head_size * math.cos(angle + math.pi / 6),
        end[1] - head_size * math.sin(angle + math.pi / 6),
    )
    draw.polygon((end, left, right), fill=ARROW)


def particle_view(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    title: str,
    fill: str,
    outline: str,
    seed: int,
) -> None:
    draw.rounded_rectangle(box, radius=18, fill="#FFFDFC", outline=outline, width=3)
    cx = (box[0] + box[2]) / 2
    centered_text(draw, (cx, box[1] + 26), title, font=VIEW_FONT, fill=outline)
    cloud_y = box[1] + 94
    draw.ellipse((cx - 78, cloud_y - 34, cx + 78, cloud_y + 34), fill=fill, outline=outline, width=2)
    rng = random.Random(seed)
    for index in range(16):
        angle = rng.uniform(0, 2 * math.pi)
        radial = math.sqrt(rng.uniform(0.02, 0.68))
        px = cx + math.cos(angle) * radial * 78
        py = cloud_y + math.sin(angle) * radial * 34
        radius = rng.randint(3, 7) if index else 8
        draw.ellipse(
            (px - radius, py - radius, px + radius, py + radius),
            fill=outline,
            outline=INK,
            width=1,
        )


def block(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    title: str,
    lines: tuple[str, ...],
    fill: str,
    outline: str,
) -> None:
    draw.rounded_rectangle(box, radius=23, fill=fill, outline=outline, width=3)
    cx = (box[0] + box[2]) / 2
    centered_text(draw, (cx, box[1] + 46), title, font=BLOCK_TITLE_FONT)
    for index, line in enumerate(lines):
        centered_text(
            draw,
            (cx, box[1] + 94 + index * 31),
            line,
            font=BLOCK_TEXT_FONT,
            fill=MUTED,
        )


def token_row(
    draw: ImageDraw.ImageDraw,
    *,
    start_x: int,
    y: int,
    colors: tuple[str, ...],
    tokens_per_color: int,
    token_width: int = 18,
    gap: int = 5,
) -> None:
    x = start_x
    for color in colors:
        for _ in range(tokens_per_color):
            draw.rounded_rectangle(
                (x, y - 12, x + token_width, y + 12),
                radius=4,
                fill=color,
            )
            x += token_width + gap
        x += gap


def build_slide() -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)

    draw.text(
        (76, 45),
        "Cross-Attention Fusion into One Token Set",
        font=TITLE_FONT,
        fill=INK,
    )
    draw.text(
        (78, 105),
        "Learned queries compress three reconstructed particle views into one fixed-size latent sequence",
        font=SUBTITLE_FONT,
        fill=MUTED,
    )

    views = (
        ((70, 175, 315, 315), "Reconstructed view 1", "#DDF3E8", "#32966D", 31),
        ((70, 355, 315, 495), "Reconstructed view 2", "#D9F1F4", "#24939E", 47),
        ((70, 535, 315, 675), "Reconstructed view 3", "#EAE5F8", "#7557B4", 63),
    )
    for box_spec, title, fill, outline, seed in views:
        particle_view(
            draw,
            box_spec,
            title=title,
            fill=fill,
            outline=outline,
            seed=seed,
        )

    memory_box = (405, 300, 675, 550)
    block(
        draw,
        memory_box,
        title="Combined memory",
        lines=(
            "Particle tokens + view IDs",
            "Keys and values",
        ),
        fill="#F5F7FA",
        outline="#7C8796",
    )
    token_row(
        draw,
        start_x=440,
        y=490,
        colors=("#32966D", "#24939E", "#7557B4"),
        tokens_per_color=3,
        token_width=15,
        gap=5,
    )
    centered_text(
        draw,
        (540, 590),
        "approximately 3N input tokens",
        font=COUNT_FONT,
        fill="#657181",
    )

    for box_spec, *_ in views:
        arrow(
            draw,
            (box_spec[2], (box_spec[1] + box_spec[3]) / 2),
            (memory_box[0], (memory_box[1] + memory_box[3]) / 2),
        )

    resampler_box = (785, 285, 1085, 565)
    block(
        draw,
        resampler_box,
        title="Cross-attention resampler",
        lines=(
            "N learned queries",
            "attend to all three views",
        ),
        fill="#FFF3CF",
        outline="#C99837",
    )
    centered_text(
        draw,
        (935, 465),
        "Queries",
        font=SMALL_FONT,
        fill="#8A641F",
    )
    token_row(
        draw,
        start_x=838,
        y=505,
        colors=("#D9A441",),
        tokens_per_color=8,
        token_width=17,
        gap=6,
    )
    arrow(draw, (memory_box[2], 425), (resampler_box[0], 425))

    unified_box = (1185, 300, 1515, 550)
    block(
        draw,
        unified_box,
        title="Unified latent token set",
        lines=(
            "One fused representation",
            "N latent tokens",
        ),
        fill="#E8ECF8",
        outline="#5367A5",
    )
    token_row(
        draw,
        start_x=1240,
        y=490,
        colors=("#5367A5",),
        tokens_per_color=9,
        token_width=18,
        gap=6,
    )
    centered_text(
        draw,
        (1350, 590),
        "approximately N output tokens",
        font=COUNT_FONT,
        fill="#5367A5",
    )
    arrow(draw, (resampler_box[2], 425), (unified_box[0], 425))

    centered_text(
        draw,
        (800, 720),
        "Three particle sets in  ->  one compact token set out",
        font=COUNT_FONT,
    )
    return image


def main() -> None:
    output_path = (
        Path(__file__).resolve().parents[1]
        / "teacher_logit_reco"
        / "presentation_assets"
        / "cross_attention_token_resampler.jpg"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    build_slide().save(output_path, format="JPEG", quality=96, subsampling=0)
    print(output_path)


if __name__ == "__main__":
    main()
