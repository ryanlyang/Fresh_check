from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    font_name = "arialbd.ttf" if bold else "arial.ttf"
    font_path = Path("C:/Windows/Fonts") / font_name
    if font_path.exists():
        return ImageFont.truetype(str(font_path), size)
    fallback = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(fallback, size)


def centered_text(
    draw: ImageDraw.ImageDraw,
    center: tuple[float, float],
    value: str,
    *,
    font: ImageFont.FreeTypeFont,
    fill: str,
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


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source = (
        repo_root
        / "teacher_logit_reco"
        / "presentation_assets"
        / "five_models_fusion.jpg"
    )
    output = source.with_name("five_models_fusion_genhigh.jpg")

    image = Image.open(source).convert("RGB")
    draw = ImageDraw.Draw(image)

    card_fill = "#D8F2F3"
    draw.rectangle((123, 530, 349, 592), fill=card_fill)

    centered_text(
        draw,
        (236, 550),
        "m2_genhigh",
        font=load_font(22, bold=True),
        fill="#1F2937",
    )
    centered_text(
        draw,
        (236, 581),
        "more generation",
        font=load_font(17),
        fill="#536174",
    )

    image.save(output, format="JPEG", quality=96, subsampling=0)
    print(output)


if __name__ == "__main__":
    main()
