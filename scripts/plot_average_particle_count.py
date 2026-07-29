from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


STAGES = ("Offline", "After dropping", "After merging")
COUNTS = (39.5, 36.6, 34.1)
COLORS = ("#2563EB", "#D97706", "#0F766E")

WIDTH = 1600
HEIGHT = 950
PLOT_LEFT = 150
PLOT_RIGHT = 1530
PLOT_TOP = 150
PLOT_BOTTOM = 735
Y_MAX = 45.0


def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    font_name = "arialbd.ttf" if bold else "arial.ttf"
    font_path = Path("C:/Windows/Fonts") / font_name
    if font_path.exists():
        return ImageFont.truetype(str(font_path), size)
    return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)


def centered_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    value: str,
    *,
    font: ImageFont.FreeTypeFont,
    fill: str = "#111827",
) -> None:
    box = draw.textbbox((0, 0), value, font=font)
    width = box[2] - box[0]
    height = box[3] - box[1]
    draw.text(
        (xy[0] - width / 2, xy[1] - height / 2 - box[1]),
        value,
        font=font,
        fill=fill,
    )


def y_position(value: float) -> float:
    plot_height = PLOT_BOTTOM - PLOT_TOP
    return PLOT_BOTTOM - (value / Y_MAX) * plot_height


def build_chart() -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), "#FFFFFF")
    draw = ImageDraw.Draw(image)

    title_font = load_font(46, bold=True)
    axis_font = load_font(29)
    axis_title_font = load_font(31, bold=True)
    stage_font = load_font(30, bold=True)
    value_font = load_font(38, bold=True)
    footer_font = load_font(34, bold=True)

    centered_text(
        draw,
        (WIDTH / 2, 72),
        "Average Particle Count Through Pseudo-HLT Processing",
        font=title_font,
    )

    for tick in range(0, 46, 5):
        y = y_position(float(tick))
        draw.line((PLOT_LEFT, y, PLOT_RIGHT, y), fill="#D1D5DB", width=2)
        tick_label = str(tick)
        box = draw.textbbox((0, 0), tick_label, font=axis_font)
        draw.text(
            (PLOT_LEFT - 25 - (box[2] - box[0]), y - (box[3] - box[1]) / 2 - box[1]),
            tick_label,
            font=axis_font,
            fill="#111827",
        )

    draw.line(
        (PLOT_LEFT, PLOT_TOP, PLOT_LEFT, PLOT_BOTTOM),
        fill="#111827",
        width=4,
    )
    draw.line(
        (PLOT_LEFT, PLOT_BOTTOM, PLOT_RIGHT, PLOT_BOTTOM),
        fill="#111827",
        width=4,
    )

    axis_label = "Average particles per jet"
    axis_box = axis_title_font.getbbox(axis_label)
    label_width = axis_box[2] - axis_box[0]
    label_height = axis_box[3] - axis_box[1]
    label_image = Image.new("RGBA", (label_width + 20, label_height + 20), (0, 0, 0, 0))
    label_draw = ImageDraw.Draw(label_image)
    label_draw.text(
        (10 - axis_box[0], 10 - axis_box[1]),
        axis_label,
        font=axis_title_font,
        fill="#111827",
    )
    label_image = label_image.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    image.paste(
        label_image,
        (
            25,
            int((PLOT_TOP + PLOT_BOTTOM - label_image.height) / 2),
        ),
        label_image,
    )

    centers = (390, 835, 1280)
    bar_width = 260
    for center, stage, count, color in zip(centers, STAGES, COUNTS, COLORS):
        top = y_position(count)
        draw.rounded_rectangle(
            (center - bar_width / 2, top, center + bar_width / 2, PLOT_BOTTOM),
            radius=5,
            fill=color,
        )
        centered_text(
            draw,
            (center, top - 28),
            f"{count:.1f}",
            font=value_font,
        )
        centered_text(
            draw,
            (center, PLOT_BOTTOM + 55),
            stage,
            font=stage_font,
        )

    centered_text(
        draw,
        (WIDTH / 2, 875),
        "Total reduction: 5.4 particles (13.7%)",
        font=footer_font,
    )
    return image


def main() -> None:
    output_path = (
        Path(__file__).resolve().parents[1]
        / "figures"
        / "average_particle_count_processing.jpg"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    build_chart().save(output_path, format="JPEG", quality=95, subsampling=0)
    print(output_path)


if __name__ == "__main__":
    main()
