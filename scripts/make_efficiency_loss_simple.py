"""Generate a two-particle worked example for pseudo-HLT efficiency loss.

The diagram is authored as SVG with the Python standard library, then rendered
to JPEG with headless Microsoft Edge and Windows System.Drawing. No Python
graphics package is required.
"""

from __future__ import annotations

import argparse
import html
import math
import os
import subprocess
from pathlib import Path


WIDTH = 1600
HEIGHT = 900
BG = "#f7f3ed"
INK = "#18222d"
MUTED = "#68737e"
BLUE = "#4389e8"
BLUE_LIGHT = "#dceafa"
ORANGE = "#ee6b3b"
ORANGE_LIGHT = "#f8d9cb"
TEAL = "#258f8b"
TEAL_LIGHT = "#d9efec"
GOLD = "#d69b2d"
WHITE = "#ffffff"


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _text(
    x: float,
    y: float,
    body: str,
    *,
    cls: str,
    anchor: str = "middle",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'class="{_esc(cls)}">{_esc(body)}</text>'
    )


def _line_arrow(x1: float, y1: float, x2: float, y2: float) -> str:
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" '
        f'y2="{y2:.1f}" class="arrow" marker-end="url(#arrow)" />'
    )


def _pill(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill: str,
    stroke: str,
) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" '
        f'height="{height:.1f}" rx="18" fill="{fill}" stroke="{stroke}" '
        'stroke-width="2.2" />'
    )


def _neighborhood(
    *,
    cx: float,
    cy: float,
    crowded: bool,
    accent: str,
    light: str,
) -> list[str]:
    out = [
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="125" fill="{light}" '
        f'fill-opacity="0.48" stroke="{accent}" stroke-width="2.3" '
        'stroke-dasharray="10 8" />'
    ]

    if crowded:
        inside = [(-68, 22), (-42, -66), (34, 58), (73, -31), (12, -88)]
        outside = [(-157, -52), (160, 62)]
    else:
        inside = []
        outside = [(-155, 35), (149, 72), (135, -96), (-133, -105)]

    for dx, dy in inside:
        out.append(
            f'<circle cx="{cx + dx:.1f}" cy="{cy + dy:.1f}" r="14" '
            f'fill="{BLUE}" stroke="{INK}" stroke-width="1.8" />'
        )
    for dx, dy in outside:
        out.append(
            f'<circle cx="{cx + dx:.1f}" cy="{cy + dy:.1f}" r="11" '
            f'fill="{BLUE}" fill-opacity="0.45" stroke="{INK}" '
            'stroke-width="1.3" />'
        )

    out.extend(
        [
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="23" fill="{accent}" '
            f'stroke="{INK}" stroke-width="2.3" />',
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="34" fill="none" '
            f'stroke="{GOLD}" stroke-width="4" />',
            _text(cx, cy - 91, "R = 0.04", cls="tiny"),
        ]
    )
    return out


def _curve(
    *,
    left: float,
    top: float,
    width: float,
    height: float,
    percentile: float,
    multiplier: float,
    accent: str,
) -> list[str]:
    out = [
        f'<line x1="{left:.1f}" y1="{top + height:.1f}" '
        f'x2="{left + width:.1f}" y2="{top + height:.1f}" class="axis" />',
        f'<line x1="{left:.1f}" y1="{top:.1f}" x2="{left:.1f}" '
        f'y2="{top + height:.1f}" class="axis" />',
        f'<rect x="{left:.1f}" y="{top:.1f}" width="{width * 0.10:.1f}" '
        f'height="{height:.1f}" fill="{GOLD}" opacity="0.13" />',
    ]

    commands = []
    for step in range(81):
        q = step / 80.0
        value = 0.65 + 1.35 * (1.0 - q) ** 1.35
        x = left + q * width
        y = top + height - ((value - 0.5) / 1.65) * height
        commands.append(("M" if step == 0 else "L") + f" {x:.1f} {y:.1f}")
    out.append(
        f'<path d="{" ".join(commands)}" fill="none" stroke="{ORANGE}" '
        'stroke-width="5.5" stroke-linecap="round" />'
    )

    marker_x = left + percentile * width
    value = 0.65 + 1.35 * (1.0 - percentile) ** 1.35
    marker_y = top + height - ((value - 0.5) / 1.65) * height
    out.extend(
        [
            f'<line x1="{marker_x:.1f}" y1="{marker_y:.1f}" '
            f'x2="{marker_x:.1f}" y2="{top + height:.1f}" '
            f'stroke="{accent}" stroke-width="2" stroke-dasharray="5 5" />',
            f'<circle cx="{marker_x:.1f}" cy="{marker_y:.1f}" r="9" '
            f'fill="{accent}" stroke="{WHITE}" stroke-width="3" />',
            _text(left, top + height + 27, "low", cls="micro", anchor="start"),
            _text(
                left + width,
                top + height + 27,
                "high",
                cls="micro",
                anchor="end",
            ),
            _text(
                marker_x,
                top - 17,
                f"{round(percentile * 100):d}th percentile",
                cls="tiny",
            ),
            _text(marker_x, marker_y - 20, f"x{multiplier:.2g}", cls="factor"),
        ]
    )
    return out


def _worked_row(
    *,
    y: float,
    title: str,
    crowded: bool,
    percentile: float,
    multiplier: float,
    neighbor_count: int,
    final_probability: float,
    accent: str,
    light: str,
) -> list[str]:
    row_top = y - 170
    curve_left = 650.0
    curve_top = y - 95
    out = [
        f'<rect x="48" y="{row_top:.1f}" width="1504" height="340" rx="24" '
        f'fill="{WHITE}" stroke="{accent}" stroke-width="2.0" opacity="0.98" />',
        _text(83, y - 128, title, cls="row-title", anchor="start"),
        *_neighborhood(
            cx=220,
            cy=y + 18,
            crowded=crowded,
            accent=accent,
            light=light,
        ),
        _text(
            220,
            y + 157,
            f"{neighbor_count} neighbors",
            cls="neighbor-count",
        ),
        _line_arrow(372, y + 18, 435, y + 18),
        _pill(448, y - 37, 150, 110, fill=light, stroke=accent),
        _text(523, y - 3, "base", cls="tiny"),
        _text(523, y + 41, "2.5%", cls="number"),
        _line_arrow(605, y + 18, 637, y + 18),
        *_curve(
            left=curve_left,
            top=curve_top,
            width=300,
            height=150,
            percentile=percentile,
            multiplier=multiplier,
            accent=accent,
        ),
        _line_arrow(973, y + 18, 1018, y + 18),
        _pill(1032, y - 37, 182, 110, fill=light, stroke=accent),
        _text(1123, y - 4, f"{neighbor_count} x 0.2 pp", cls="tiny"),
        _text(
            1123,
            y + 41,
            f"+{neighbor_count * 0.2:.1f} pp",
            cls="number-small",
        ),
        _line_arrow(1222, y + 18, 1261, y + 18),
        _pill(1275, y - 52, 225, 140, fill=accent, stroke=accent),
        _text(1387.5, y - 7, "drop chance", cls="final-label"),
        _text(1387.5, y + 51, f"{final_probability:.2f}%", cls="final-number"),
    ]
    return out


def build_svg(output: Path) -> None:
    top_multiplier = 1.90
    top_neighbors = 5
    top_final = 2.5 * top_multiplier + top_neighbors * 0.2

    bottom_multiplier = 0.86
    bottom_neighbors = 0
    bottom_final = 2.5 * bottom_multiplier + bottom_neighbors * 0.2

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
        f'height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        "<defs>",
        '<marker id="arrow" markerWidth="12" markerHeight="12" refX="10" '
        'refY="6" orient="auto"><path d="M 0 0 L 12 6 L 0 12 z" '
        f'fill="{MUTED}" /></marker>',
        "<style>",
        f"""
        .title {{ font: 700 46px Arial, sans-serif; fill: {INK}; }}
        .row-title {{ font: 700 28px Arial, sans-serif; fill: {INK}; }}
        .tiny {{ font: 600 18px Arial, sans-serif; fill: {MUTED}; }}
        .micro {{ font: 600 15px Arial, sans-serif; fill: {MUTED}; }}
        .neighbor-count {{ font: 700 21px Arial, sans-serif; fill: {INK}; }}
        .number {{ font: 700 35px Arial, sans-serif; fill: {INK}; }}
        .number-small {{ font: 700 29px Arial, sans-serif; fill: {INK}; }}
        .factor {{ font: 700 23px Arial, sans-serif; fill: {ORANGE}; }}
        .final-label {{ font: 600 20px Arial, sans-serif; fill: {WHITE}; }}
        .final-number {{ font: 700 46px Arial, sans-serif; fill: {WHITE}; }}
        .arrow {{ stroke: {MUTED}; stroke-width: 3; }}
        .axis {{ stroke: {INK}; stroke-width: 2; }}
        """,
        "</style>",
        "</defs>",
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{BG}" />',
        _text(58, 72, "One particle, one drop probability", cls="title", anchor="start"),
        *_worked_row(
            y=265,
            title="Low pT + crowded",
            crowded=True,
            percentile=0.08,
            multiplier=top_multiplier,
            neighbor_count=top_neighbors,
            final_probability=top_final,
            accent=ORANGE,
            light=ORANGE_LIGHT,
        ),
        *_worked_row(
            y=640,
            title="Higher pT + isolated",
            crowded=False,
            percentile=0.75,
            multiplier=bottom_multiplier,
            neighbor_count=bottom_neighbors,
            final_probability=bottom_final,
            accent=TEAL,
            light=TEAL_LIGHT,
        ),
        "</svg>",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts), encoding="utf-8")


def _find_edge() -> Path:
    candidates = [
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Microsoft Edge was not found; cannot rasterize SVG")


def _render_jpg(svg_path: Path, jpg_path: Path, quality: int = 95) -> None:
    edge = _find_edge()
    png_path = jpg_path.with_suffix(".render.png")
    jpg_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                str(edge),
                "--headless",
                "--disable-gpu",
                "--hide-scrollbars",
                f"--screenshot={png_path.resolve()}",
                f"--window-size={WIDTH},{HEIGHT}",
                svg_path.resolve().as_uri(),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        ps_script = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
$source = [System.Drawing.Image]::FromFile($env:EFFICIENCY_SOURCE_PNG)
try {
    $codec = [System.Drawing.Imaging.ImageCodecInfo]::GetImageEncoders() |
        Where-Object { $_.MimeType -eq 'image/jpeg' }
    $parameters = New-Object System.Drawing.Imaging.EncoderParameters(1)
    $parameters.Param[0] = New-Object System.Drawing.Imaging.EncoderParameter(
        [System.Drawing.Imaging.Encoder]::Quality,
        [long]$env:EFFICIENCY_JPG_QUALITY
    )
    $source.Save($env:EFFICIENCY_DEST_JPG, $codec, $parameters)
} finally {
    $source.Dispose()
}
"""
        ps_env = os.environ.copy()
        ps_env["EFFICIENCY_SOURCE_PNG"] = str(png_path.resolve())
        ps_env["EFFICIENCY_DEST_JPG"] = str(jpg_path.resolve())
        ps_env["EFFICIENCY_JPG_QUALITY"] = str(quality)
        subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                ps_script,
            ],
            check=True,
            env=ps_env,
        )
    finally:
        png_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "teacher_logit_reco/presentation_assets/"
            "pseudo_hlt_efficiency_loss_two_particles.jpg"
        ),
    )
    parser.add_argument("--quality", type=int, default=95)
    args = parser.parse_args()

    if args.output.suffix.lower() not in {".jpg", ".jpeg"}:
        raise ValueError("--output must use a .jpg or .jpeg extension")
    svg_path = args.output.with_suffix(".svg")
    build_svg(svg_path)
    _render_jpg(svg_path, args.output, quality=args.quality)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
