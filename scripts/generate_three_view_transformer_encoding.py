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
BLOCK_TITLE_FONT = load_font(25, bold=True)
BLOCK_SUBTITLE_FONT = load_font(19)
VIEW_FONT = load_font(21, bold=True)
FORMULA_FONT = load_font(23, bold=True)
NOTE_FONT = load_font(22, bold=True)


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
    draw.rounded_rectangle(box, radius=19, fill="#FFFDFC", outline=outline, width=3)
    cx = (box[0] + box[2]) / 2
    centered_text(draw, (cx, box[1] + 28), title, font=VIEW_FONT, fill=outline)
    cloud_center = (int(cx), int((box[1] + box[3]) / 2 + 15))
    rx, ry = 88, 38
    draw.ellipse(
        (
            cloud_center[0] - rx,
            cloud_center[1] - ry,
            cloud_center[0] + rx,
            cloud_center[1] + ry,
        ),
        fill=fill,
        outline=outline,
        width=2,
    )
    rng = random.Random(seed)
    for index in range(17):
        angle = rng.uniform(0, 2 * math.pi)
        radial = math.sqrt(rng.uniform(0.02, 0.68))
        px = cloud_center[0] + math.cos(angle) * radial * rx
        py = cloud_center[1] + math.sin(angle) * radial * ry
        radius = rng.randint(3, 7) if index else 8
        draw.ellipse(
            (px - radius, py - radius, px + radius, py + radius),
            fill=outline,
            outline=INK,
            width=1,
        )


def process_block(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    title: str,
    subtitle_lines: tuple[str, ...],
    fill: str,
    outline: str,
    title_font: ImageFont.FreeTypeFont = BLOCK_TITLE_FONT,
) -> None:
    draw.rounded_rectangle(box, radius=24, fill=fill, outline=outline, width=3)
    cx = (box[0] + box[2]) / 2
    centered_text(draw, (cx, box[1] + 48), title, font=title_font)
    start_y = box[1] + 100
    for index, line in enumerate(subtitle_lines):
        centered_text(
            draw,
            (cx, start_y + index * 34),
            line,
            font=BLOCK_SUBTITLE_FONT,
            fill=MUTED,
        )


def draw_token_sequence(draw: ImageDraw.ImageDraw) -> None:
    colors = ("#32966D", "#24939E", "#7557B4")
    x = 858
    y = 535
    draw.rounded_rectangle((802, 507, 1058, 563), radius=15, fill="#FFFDFC", outline="#8B95A3", width=2)
    centered_text(draw, (833, y + 1), "CLS", font=load_font(16, bold=True), fill=INK)
    for color in colors:
        for _ in range(3):
            draw.rounded_rectangle((x, y - 11, x + 17, y + 11), radius=4, fill=color)
            x += 22
        x += 5


def build_slide() -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)

    draw.text(
        (76, 45),
        "Joint Encoding of Three Reconstructed Views",
        font=TITLE_FONT,
        fill=INK,
    )
    draw.text(
        (78, 105),
        "Every particle stays a separate token; a learned view ID records where it came from",
        font=SUBTITLE_FONT,
        fill=MUTED,
    )

    views = (
        ((75, 180, 335, 330), "Reconstructed view 1", "#DDF3E8", "#32966D", 31),
        ((75, 365, 335, 515), "Reconstructed view 2", "#D9F1F4", "#24939E", 47),
        ((75, 550, 335, 700), "Reconstructed view 3", "#EAE5F8", "#7557B4", 63),
    )
    for box, title, fill, outline, seed in views:
        particle_view(
            draw,
            box,
            title=title,
            fill=fill,
            outline=outline,
            seed=seed,
        )

    embed_box = (440, 295, 720, 585)
    process_block(
        draw,
        embed_box,
        title="Shared particle embedder",
        subtitle_lines=(
            "Same projection for every view",
            "Particle features + view ID",
        ),
        fill="#FFF3CF",
        outline="#C99837",
        title_font=load_font(22, bold=True),
    )
    centered_text(
        draw,
        ((embed_box[0] + embed_box[2]) / 2, 535),
        "z = MLP(x) + e_view",
        font=FORMULA_FONT,
        fill="#8A641F",
    )

    for box, *_ in views:
        arrow(
            draw,
            (box[2], (box[1] + box[3]) / 2),
            (embed_box[0], (embed_box[1] + embed_box[3]) / 2),
        )

    concat_box = (790, 295, 1070, 585)
    process_block(
        draw,
        concat_box,
        title="Concatenate tokens",
        subtitle_lines=(
            "Add one global CLS token",
            "Keep particle masks",
            "Preserve view identity",
        ),
        fill="#F5F7FA",
        outline="#7C8796",
    )
    draw_token_sequence(draw)
    arrow(draw, (embed_box[2], 440), (concat_box[0], 440))

    transformer_box = (1150, 295, 1450, 585)
    process_block(
        draw,
        transformer_box,
        title="One Transformer",
        subtitle_lines=(
            "Self-attention across",
            "all particles and views",
            "",
            "Learns agreement",
            "and disagreement",
        ),
        fill="#E8ECF8",
        outline="#5367A5",
    )
    arrow(draw, (concat_box[2], 440), (transformer_box[0], 440))

    output_box = (1135, 650, 1465, 790)
    process_block(
        draw,
        output_box,
        title="One jet encoding",
        subtitle_lines=("CLS representation -> class logits",),
        fill="#FFFDFC",
        outline="#5367A5",
    )
    arrow(draw, (1300, transformer_box[3]), (1300, output_box[1]))

    return image


def main() -> None:
    output_path = (
        Path(__file__).resolve().parents[1]
        / "teacher_logit_reco"
        / "presentation_assets"
        / "three_view_transformer_encoding.jpg"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    build_slide().save(output_path, format="JPEG", quality=96, subsampling=0)
    print(output_path)


if __name__ == "__main__":
    main()
