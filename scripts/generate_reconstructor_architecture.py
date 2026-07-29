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


TITLE_FONT = load_font(50, bold=True)
SUBTITLE_FONT = load_font(25)
BLOCK_TITLE_FONT = load_font(26, bold=True)
BLOCK_SUBTITLE_FONT = load_font(20)
VIEW_FONT = load_font(19, bold=True)


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


def rounded_block(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    fill: str,
    outline: str,
    title: str,
    subtitle: str,
    radius: int = 24,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=3)
    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2
    centered_text(
        draw,
        (center_x, center_y - 15),
        title,
        font=BLOCK_TITLE_FONT,
    )
    centered_text(
        draw,
        (center_x, center_y + 25),
        subtitle,
        font=BLOCK_SUBTITLE_FONT,
        fill=MUTED,
    )


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = ARROW,
    width: int = 4,
    head_size: int = 14,
) -> None:
    draw.line((*start, *end), fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    left = (
        end[0] - head_size * math.cos(angle - math.pi / 6),
        end[1] - head_size * math.sin(angle - math.pi / 6),
    )
    right = (
        end[0] - head_size * math.cos(angle + math.pi / 6),
        end[1] - head_size * math.sin(angle + math.pi / 6),
    )
    draw.polygon((end, left, right), fill=color)


def particle_cloud(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    radii: tuple[int, int],
    *,
    fill: str,
    outline: str,
    seed: int,
    count: int = 20,
) -> None:
    cx, cy = center
    rx, ry = radii
    draw.ellipse(
        (cx - rx, cy - ry, cx + rx, cy + ry),
        fill=fill,
        outline=outline,
        width=3,
    )
    rng = random.Random(seed)
    for index in range(count):
        angle = rng.uniform(0, 2 * math.pi)
        radial = math.sqrt(rng.uniform(0.02, 0.68))
        px = cx + math.cos(angle) * radial * rx
        py = cy + math.sin(angle) * radial * ry
        radius = rng.randint(4, 9) if index else 10
        draw.ellipse(
            (px - radius, py - radius, px + radius, py + radius),
            fill=outline,
            outline=INK,
            width=2,
        )


def build_diagram() -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)

    draw.text((78, 48), "Reconstructor architecture", font=TITLE_FONT, fill=INK)
    draw.text(
        (80, 108),
        "A shared HLT representation drives complementary branches that populate "
        "three reconstructed views",
        font=SUBTITLE_FONT,
        fill=MUTED,
    )

    input_box = (75, 285, 350, 555)
    rounded_block(
        draw,
        input_box,
        fill="#FFFDFC",
        outline="#E98A63",
        title="Pseudo-HLT",
        subtitle="particles",
    )
    particle_cloud(
        draw,
        (212, 465),
        (86, 53),
        fill="#FCE3D6",
        outline="#E85D3F",
        seed=10,
        count=21,
    )

    backbone_box = (445, 330, 750, 505)
    rounded_block(
        draw,
        backbone_box,
        fill="#FFF3CF",
        outline="#C99837",
        title="Reconstructor backbone",
        subtitle="particle + jet context",
    )
    arrow(draw, (input_box[2], 420), (backbone_box[0], 420))

    branch_boxes = (
        ((835, 155, 1115, 245), "#DDF3E8", "#32966D", "Edit branch", "adjust existing particles"),
        ((835, 275, 1115, 365), "#D9F1F4", "#24939E", "Split branch", "parent to children"),
        ((835, 395, 1115, 485), "#EAE5F8", "#7557B4", "Generate branch", "new candidates"),
        ((835, 515, 1115, 605), "#FFF3CF", "#C99837", "Budget head", "counts + weights"),
    )

    backbone_right = (backbone_box[2], (backbone_box[1] + backbone_box[3]) / 2)
    for box, fill, outline, title, subtitle in branch_boxes:
        rounded_block(
            draw,
            box,
            fill=fill,
            outline=outline,
            title=title,
            subtitle=subtitle,
        )
        arrow(
            draw,
            backbone_right,
            (box[0], (box[1] + box[3]) / 2),
        )

    bank_box = (1245, 320, 1530, 455)
    rounded_block(
        draw,
        bank_box,
        fill="#FFFDFC",
        outline="#C99837",
        title="Candidate bank",
        subtitle="budgeted candidates",
    )

    bank_left_x = bank_box[0]
    bank_center_y = (bank_box[1] + bank_box[3]) / 2
    bank_targets = (340, 372, 404, 436)
    for branch, target_y in zip(branch_boxes, bank_targets):
        box = branch[0]
        arrow(
            draw,
            (box[2], (box[1] + box[3]) / 2),
            (bank_left_x, target_y),
        )

    view_specs = (
        ((900, 735), "#DDF3E8", "#32966D", 31, "Reconstructed view 1"),
        ((1170, 735), "#D9F1F4", "#24939E", 47, "Reconstructed view 2"),
        ((1430, 735), "#EAE5F8", "#7557B4", 63, "Reconstructed view 3"),
    )
    bank_bottom = ((bank_box[0] + bank_box[2]) / 2, bank_box[3])
    for center, fill, outline, seed, label in view_specs:
        arrow(draw, bank_bottom, (center[0], center[1] - 70))
        particle_cloud(
            draw,
            center,
            (88, 57),
            fill=fill,
            outline=outline,
            seed=seed,
            count=18,
        )
        centered_text(
            draw,
            (center[0], 827),
            label,
            font=VIEW_FONT,
        )

    return image


def main() -> None:
    output_path = (
        Path(__file__).resolve().parents[1]
        / "teacher_logit_reco"
        / "presentation_assets"
        / "reconstructor_architecture.jpg"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    build_diagram().save(output_path, format="JPEG", quality=96, subsampling=0)
    print(output_path)


if __name__ == "__main__":
    main()
