"""Generate a presentation graphic for the pseudo-HLT v2 degradation profile.

The script uses only Python plus the repository's HLT parameter scaler. It
intentionally writes SVG so it is easy to inspect, version, and drop into slides.
"""

from __future__ import annotations

import argparse
import html
import math
import random
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jetclass_fixed_hlt import scaled_fixed_hlt_v2_realistic_params


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _fmt(value: float, digits: int = 4) -> str:
    if abs(value) >= 10:
        return f"{value:.2f}"
    if abs(value) >= 1:
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def _text(x: float, y: float, body: str, *, cls: str = "", anchor: str = "start") -> str:
    cls_attr = f' class="{_esc(cls)}"' if cls else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}"{cls_attr}>{_esc(body)}</text>'


def _rect(x: float, y: float, w: float, h: float, *, cls: str, rx: float = 8.0) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx:.1f}" class="{_esc(cls)}" />'


def _line(x1: float, y1: float, x2: float, y2: float, *, cls: str = "arrow") -> str:
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'class="{_esc(cls)}" marker-end="url(#arrow)" />'
    )


def _particle_cloud(
    *,
    x: float,
    y: float,
    title: str,
    seed: int,
    degraded: bool,
) -> list[str]:
    rng = random.Random(seed)
    out: list[str] = []
    out.append(_rect(x, y, 230, 170, cls="cloud"))
    out.append(_text(x + 115, y + 26, title, cls="cloud-title", anchor="middle"))
    cx = x + 115
    cy = y + 94
    out.append(f'<ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="82" ry="52" class="jet-ellipse" />')

    points = []
    for idx in range(42):
        r = 1.0 - rng.random() ** 2
        theta = rng.random() * 2.0 * math.pi
        px = cx + math.cos(theta) * r * rng.uniform(12, 82)
        py = cy + math.sin(theta) * r * rng.uniform(8, 50)
        pt = rng.uniform(2.0, 9.0)
        points.append((idx, px, py, pt))

    if degraded:
        keep = []
        for idx, px, py, pt in points:
            if pt < 3.0 and idx % 2 == 0:
                continue
            if idx % 11 == 0:
                continue
            dx = rng.uniform(-5.0, 5.0)
            dy = rng.uniform(-4.0, 4.0)
            keep.append((idx, px + dx, py + dy, pt * rng.uniform(0.72, 1.18)))
        merged = keep[:]
        for idx, px, py, pt in keep[::13]:
            merged.append((idx + 1000, px + 6.0, py - 4.0, pt * 1.35))
        points = merged

    for idx, px, py, pt in points:
        radius = max(2.2, min(7.2, pt * 0.72))
        cls = "particle hlt-particle" if degraded else "particle offline-particle"
        opacity = 0.55 if degraded and idx % 7 == 0 else 0.9
        out.append(
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{radius:.1f}" '
            f'class="{cls}" opacity="{opacity:.2f}" />'
        )
    if degraded:
        out.append(_text(x + 115, y + 152, "fewer, smeared, merged tokens", cls="caption", anchor="middle"))
    else:
        out.append(_text(x + 115, y + 152, "offline-like constituent set", cls="caption", anchor="middle"))
    return out


def _pipeline_box(x: float, y: float, title: str, lines: Iterable[str]) -> list[str]:
    out = [_rect(x, y, 165, 124, cls="step-box")]
    out.append(_text(x + 82.5, y + 28, title, cls="step-title", anchor="middle"))
    yy = y + 55
    for line in lines:
        out.append(_text(x + 16, yy, line, cls="step-line"))
        yy += 20
    return out


def build_svg(strength: float) -> str:
    params = scaled_fixed_hlt_v2_realistic_params(strength)
    width, height = 1320, 820
    rows = [
        ("pT threshold", f"{_fmt(params.hlt_pt_threshold)} GeV"),
        ("Merge radius", f"{_fmt(params.merge_radius, 5)}"),
        ("Merge probability", _fmt(params.merge_probability)),
        ("Barrel plateau", _fmt(params.eff_plateau_barrel)),
        ("Endcap plateau", _fmt(params.eff_plateau_endcap)),
        ("Barrel pT50 / width", f"{_fmt(params.eff_turnon_pt_barrel)} / {_fmt(params.eff_width_pt_barrel)}"),
        ("Endcap pT50 / width", f"{_fmt(params.eff_turnon_pt_endcap)} / {_fmt(params.eff_width_pt_endcap)}"),
        ("Density loss scale", _fmt(params.density_loss_scale)),
        ("Smear scale", _fmt(params.smear_scale)),
        ("Reassignment scale", _fmt(params.reassign_scale)),
    ]

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<marker id="arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">',
        '<path d="M 0 0 L 10 5 L 0 10 z" class="arrow-head" />',
        "</marker>",
        "<style>",
        """
        .bg { fill: #f7f3ea; }
        .title { font: 700 32px Arial, sans-serif; fill: #17202a; }
        .subtitle { font: 18px Arial, sans-serif; fill: #425466; }
        .section-label { font: 700 15px Arial, sans-serif; letter-spacing: .08em; fill: #566573; }
        .cloud { fill: #ffffff; stroke: #243447; stroke-width: 1.4; }
        .cloud-title { font: 700 18px Arial, sans-serif; fill: #17202a; }
        .caption { font: 13px Arial, sans-serif; fill: #53616f; }
        .jet-ellipse { fill: #dbeafe; stroke: #6b8fbf; stroke-width: 1; opacity: .45; }
        .particle { stroke: #17202a; stroke-width: .55; }
        .offline-particle { fill: #2f80ed; }
        .hlt-particle { fill: #e4572e; }
        .step-box { fill: #fffaf2; stroke: #c08a3e; stroke-width: 1.2; }
        .step-title { font: 700 15px Arial, sans-serif; fill: #17202a; }
        .step-line { font: 13px Arial, sans-serif; fill: #36454f; }
        .arrow { stroke: #566573; stroke-width: 2.2; fill: none; }
        .arrow-head { fill: #566573; }
        .callout { fill: #17202a; }
        .callout-text { font: 700 18px Arial, sans-serif; fill: #ffffff; }
        .callout-small { font: 14px Arial, sans-serif; fill: #eef4ff; }
        .table { fill: #ffffff; stroke: #85929e; stroke-width: 1.2; }
        .table-head { fill: #243447; }
        .table-head-text { font: 700 14px Arial, sans-serif; fill: #ffffff; }
        .table-key { font: 13px Arial, sans-serif; fill: #243447; }
        .table-val { font: 700 13px Arial, sans-serif; fill: #17202a; }
        .note { font: 14px Arial, sans-serif; fill: #425466; }
        """,
        "</style>",
        "</defs>",
        '<rect x="0" y="0" width="1320" height="820" class="bg" />',
        _text(55, 62, "Pseudo-HLT v2 View at Strength 2.5", cls="title"),
        _text(55, 94, "A deterministic stress-test degradation: offline JetClass particles are converted into a matched HLT-like particle set.", cls="subtitle"),
        _text(55, 142, "CONSTITUENT VIEW", cls="section-label"),
    ]

    svg.extend(_particle_cloud(x=55, y=165, title="Offline jet", seed=12, degraded=False))
    svg.extend(_particle_cloud(x=1035, y=165, title="Pseudo-HLT jet", seed=12, degraded=True))
    svg.append(_line(285, 250, 340, 250))

    step_xs = [350, 525, 700, 875]
    steps = [
        ("1. Threshold", ["drop very soft", "constituents", f"pT < {_fmt(params.hlt_pt_threshold)} GeV"]),
        ("2. Merge", ["nearby particles", f"DeltaR < {_fmt(params.merge_radius, 5)}", f"prob = {_fmt(params.merge_probability)}"]),
        ("3. Efficiency", ["pT turn-on", "eta plateau", "density loss"]),
        ("4. Smear", ["pT and axis", "rare tails", "local reassignment"]),
    ]
    for x, (title, lines) in zip(step_xs, steps):
        svg.extend(_pipeline_box(x, 188, title, lines))
        svg.append(_line(x + 165, 250, x + 175, 250))
    svg.append(_line(1040, 250, 1035, 250))

    svg.extend(
        [
            '<rect x="55" y="390" width="415" height="120" rx="10" class="callout" />',
            _text(80, 426, "Strength convention", cls="callout-text"),
            _text(80, 456, "0.0 = exact offline identity", cls="callout-small"),
            _text(80, 481, "1.0 = mild realistic target point", cls="callout-small"),
            _text(80, 506, "2.5 = amplified stress-test degradation", cls="callout-small"),
            _text(525, 407, "EFFECTIVE PARAMETERS AT STRENGTH 2.5", cls="section-label"),
            _rect(525, 425, 740, 315, cls="table", rx=8),
            '<rect x="525" y="425" width="740" height="36" rx="8" class="table-head" />',
            _text(548, 449, "Parameter", cls="table-head-text"),
            _text(900, 449, "Effective value", cls="table-head-text"),
        ]
    )

    y = 486
    for idx, (key, value) in enumerate(rows):
        if idx % 2 == 0:
            svg.append(f'<rect x="526" y="{y - 17:.1f}" width="738" height="26" fill="#f4f7fa" />')
        svg.append(_text(548, y, key, cls="table-key"))
        svg.append(_text(900, y, value, cls="table-val"))
        y += 26

    svg.extend(
        [
            _text(55, 580, "Slide phrasing", cls="section-label"),
            _text(55, 612, "We do not change the label or the underlying physics event.", cls="note"),
            _text(55, 638, "We only create a second detector-resolution view paired one-to-one with the offline jet.", cls="note"),
            _text(55, 664, "At training time, offline and pseudo-HLT can both be used; at inference time, only pseudo-HLT is available.", cls="note"),
            _text(55, 707, "Best short name: fixed_hlt_v2_realistic, strength=2.5", cls="note"),
            _text(55, 733, "Use this as a stress test, not as a claim about an exact CMS/ATLAS HLT response.", cls="note"),
            "</svg>",
        ]
    )
    return "\n".join(svg)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strength", type=float, default=2.5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("teacher_logit_reco/presentation_assets/hlt_v2_strength_2p5_pipeline.svg"),
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_svg(args.strength), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
