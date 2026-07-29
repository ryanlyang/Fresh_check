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


TITLE_FONT = load_font(49, bold=True)
SUBTITLE_FONT = load_font(25)
FLOW_TITLE_FONT = load_font(22, bold=True)
FLOW_SUBTITLE_FONT = load_font(17)
VS_FONT = load_font(31, bold=True)
SECTION_FONT = load_font(29, bold=True)
LOSS_TITLE_FONT = load_font(28, bold=True)
LOSS_KICKER_FONT = load_font(21, bold=True)
LOSS_BODY_FONT = load_font(20)
FORMULA_FONT = load_font(31, bold=True)


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
    head_size = 13
    left = (
        end[0] - head_size * math.cos(angle - math.pi / 6),
        end[1] - head_size * math.sin(angle - math.pi / 6),
    )
    right = (
        end[0] - head_size * math.cos(angle + math.pi / 6),
        end[1] - head_size * math.sin(angle + math.pi / 6),
    )
    draw.polygon((end, left, right), fill=ARROW)


def particle_cloud(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    radii: tuple[int, int],
    *,
    fill: str,
    outline: str,
    seed: int,
    count: int,
) -> None:
    cx, cy = center
    rx, ry = radii
    draw.ellipse(
        (cx - rx, cy - ry, cx + rx, cy + ry),
        fill=fill,
        outline=outline,
        width=2,
    )
    rng = random.Random(seed)
    for index in range(count):
        angle = rng.uniform(0, 2 * math.pi)
        radial = math.sqrt(rng.uniform(0.02, 0.68))
        px = cx + math.cos(angle) * radial * rx
        py = cy + math.sin(angle) * radial * ry
        radius = rng.randint(3, 7) if index else 8
        draw.ellipse(
            (px - radius, py - radius, px + radius, py + radius),
            fill=outline,
            outline=INK,
            width=1,
        )


def flow_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    fill: str,
    outline: str,
    title: str,
    subtitle: str,
    cloud: tuple[str, str, int, int] | None = None,
) -> None:
    draw.rounded_rectangle(box, radius=18, fill=fill, outline=outline, width=3)
    cx = (box[0] + box[2]) / 2
    centered_text(
        draw,
        (cx, box[1] + 32),
        title,
        font=FLOW_TITLE_FONT,
    )
    if cloud is not None:
        cloud_fill, cloud_outline, seed, count = cloud
        particle_cloud(
            draw,
            (int(cx), box[1] + 100),
            (67, 36),
            fill=cloud_fill,
            outline=cloud_outline,
            seed=seed,
            count=count,
        )
    centered_text(
        draw,
        (cx, box[3] - 20),
        subtitle,
        font=FLOW_SUBTITLE_FONT,
        fill=MUTED,
    )


def loss_card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    fill: str,
    outline: str,
    title: str,
    kicker: str,
    lines: tuple[str, ...],
    term: str,
) -> None:
    draw.rounded_rectangle(box, radius=24, fill=fill, outline=outline, width=4)
    cx = (box[0] + box[2]) / 2
    centered_text(
        draw,
        (cx, box[1] + 43),
        title,
        font=LOSS_TITLE_FONT,
        fill=outline,
    )
    centered_text(
        draw,
        (cx, box[1] + 89),
        kicker,
        font=LOSS_KICKER_FONT,
    )
    start_y = box[1] + 136
    for index, line in enumerate(lines):
        centered_text(
            draw,
            (cx, start_y + index * 31),
            line,
            font=LOSS_BODY_FONT,
            fill=MUTED,
        )
    centered_text(
        draw,
        (cx, box[3] - 32),
        term,
        font=LOSS_KICKER_FONT,
        fill=outline,
    )


def build_diagram() -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)

    draw.text((78, 45), "Losses / Training Objective", font=TITLE_FONT, fill=INK)
    draw.text(
        (80, 105),
        "The paired offline jet supervises both constituent-level recovery and "
        "global jet consistency",
        font=SUBTITLE_FONT,
        fill=MUTED,
    )

    input_box = (85, 185, 305, 350)
    reco_box = (390, 205, 610, 330)
    reconstructed_box = (700, 185, 940, 350)
    offline_box = (1080, 185, 1320, 350)

    flow_box(
        draw,
        input_box,
        fill="#F8FBFF",
        outline="#3B88D8",
        title="Pseudo-HLT input",
        subtitle="degraded particles",
        cloud=("#DCEAF9", "#347FD0", 11, 17),
    )
    flow_box(
        draw,
        reco_box,
        fill="#FFF3CF",
        outline="#C99837",
        title="Reconstructor",
        subtitle="predicts corrected set",
    )
    flow_box(
        draw,
        reconstructed_box,
        fill="#F6FCF9",
        outline="#32966D",
        title="Reconstructed view",
        subtitle="model prediction",
        cloud=("#DDF3E8", "#32966D", 31, 21),
    )
    flow_box(
        draw,
        offline_box,
        fill="#FFFDFC",
        outline="#E35E39",
        title="Offline target",
        subtitle="paired same jet",
        cloud=("#FCE3D6", "#E35E39", 71, 24),
    )

    arrow(draw, (input_box[2], 267), (reco_box[0], 267))
    arrow(draw, (reco_box[2], 267), (reconstructed_box[0], 267))
    centered_text(
        draw,
        ((reconstructed_box[2] + offline_box[0]) / 2, 267),
        "VS",
        font=VS_FONT,
        fill="#6D58A6",
    )

    centered_text(
        draw,
        (WIDTH / 2, 405),
        "Three complementary reconstruction losses",
        font=SECTION_FONT,
    )
    draw.line((250, 435, 1350, 435), fill="#C8CED6", width=2)

    cards = (
        (
            (70, 470, 530, 760),
            "#F8FBFF",
            "#347FD0",
            "Set matching loss",
            "Permutation-invariant matching",
            (
                "Matches reconstructed and offline particles.",
                "Independent of particle ordering.",
            ),
            "L_set",
        ),
        (
            (570, 470, 1030, 760),
            "#F6FCF9",
            "#32966D",
            "Generation loss",
            "Missing-constituent recovery",
            (
                "Recovers particles lost during HLT degradation.",
                "Suppresses unsupported candidates.",
            ),
            "L_gen",
        ),
        (
            (1070, 470, 1530, 760),
            "#FFFDFC",
            "#E35E39",
            "Jet pT loss",
            "Jet-level consistency",
            (
                "Matches reconstructed and offline total pT.",
            ),
            "L_jet-pT",
        ),
    )
    for box, fill, outline, title, kicker, lines, term in cards:
        loss_card(
            draw,
            box,
            fill=fill,
            outline=outline,
            title=title,
            kicker=kicker,
            lines=lines,
            term=term,
        )

    centered_text(
        draw,
        (WIDTH / 2, 830),
        "L = w_set L_set + w_gen L_gen + w_pT L_jet-pT",
        font=FORMULA_FONT,
    )
    return image


def main() -> None:
    output_path = (
        Path(__file__).resolve().parents[1]
        / "teacher_logit_reco"
        / "presentation_assets"
        / "m2_losses_training_objective_v2.jpg"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    build_diagram().save(output_path, format="JPEG", quality=96, subsampling=0)
    print(output_path)


if __name__ == "__main__":
    main()
