"""Draw a simple 16:9 slide explaining ParT's learned pairwise bias.

Run: python figures/slides/draw_pairwise_bias_slide.py

Requires Matplotlib; exports beside this script by default. Outputs are schematic,
not event data or measured attention; positions and widths carry no physics
claim. The pair-feature questions are examples, not an exhaustive input list.
This script does not modify the manuscript or existing EdgeValue figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


SLIDE_ROOT = Path(__file__).resolve().parent
INK = "#203343"
MUTED = "#647582"
BLUE = "#31688e"
TEAL = "#178075"
ORANGE = "#b45b27"
FAINT = "#d9e2e8"


def label(ax, x, y, text, size=20, color=INK, **kwargs):
    return ax.text(x, y, text, ha="center", va="center", fontsize=size,
                   color=color, zorder=6, **kwargs)


def box(ax, x, y, width, height, *, fill="white", edge=FAINT):
    ax.add_patch(FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0,rounding_size=0.16",
        facecolor=fill, edgecolor=edge, linewidth=1.4, zorder=1,
    ))


def arrow(ax, start, end, *, color=INK, width=2, shrink=0):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=20,
        linewidth=width, color=color, shrinkA=shrink, shrinkB=shrink,
        zorder=3,
    ))


def particle(ax, x, y, name):
    ax.add_patch(Circle((x, y), 0.25, facecolor="white",
                        edgecolor=BLUE, linewidth=2, zorder=4))
    label(ax, x, y, name, 19, BLUE)


def build_figure():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "svg.hashsalt": "part-pairwise-bias-slide-v1",
    })
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, 16), ylim=(0, 9), aspect="equal")
    ax.set_axis_off()

    label(ax, 8, 8.24,
          "ParT: look at the relationship, not just the particles",
          29, fontweight="bold")
    label(ax, 8, 7.60,
          "Pairwise physics helps decide how much attention one particle gives another.",
          20, MUTED)

    for x, width in ((0.55, 4.30), (5.55, 4.30), (10.55, 4.90)):
        box(ax, x, 2.25, width, 4.65, fill="#f7f9fb")
    arrow(ax, (4.97, 4.63), (5.43, 4.63), color=MUTED)
    arrow(ax, (9.97, 4.63), (10.43, 4.63), color=MUTED)

    # Pair geometry here is arbitrary: no distance-to-weight rule is depicted.
    label(ax, 2.70, 6.42, "1. Describe the pair", 22, fontweight="bold")
    ax.plot((1.55, 3.85), (5.65, 5.65), color=FAINT,
            linewidth=2, linestyle="--", zorder=2)
    particle(ax, 1.55, 5.65, "$i$")
    particle(ax, 3.85, 5.65, "$j$")
    label(ax, 2.70, 4.87, "Examples of pair information", 15, MUTED)
    label(ax, 2.70, 4.20, "How far apart?", 20)
    label(ax, 2.70, 3.57, "How is momentum shared?", 19)
    label(ax, 2.70, 2.94, "What is their combined mass?", 18)

    label(ax, 7.70, 6.42, "2. Learn the bias", 22, fontweight="bold")
    box(ax, 5.82, 4.76, 3.76, 1.05, fill="#e8f2ef", edge="#c2d9d1")
    label(ax, 7.70, 5.285, "Small learned network", 19, TEAL,
          fontweight="bold")
    arrow(ax, (7.70, 4.57), (7.70, 4.03), color=TEAL)
    label(ax, 7.70, 3.59, "Positive or negative", 20, TEAL)
    label(ax, 7.70, 3.13, "score adjustment", 20, TEAL)
    label(ax, 7.70, 2.59, "Learned from training examples", 14, MUTED)

    label(ax, 13.00, 6.42, "3. Adjust attention", 22, fontweight="bold")
    for y, weight, title in ((5.15, 1.5, "Less attention"),
                              (3.72, 5.0, "More attention")):
        label(ax, 13.00, y + 0.53, title, 18, ORANGE)
        # j sends information to i; i's attention determines the weighting.
        arrow(ax, (14.38, y), (11.62, y), color=ORANGE,
              width=weight, shrink=21)
        particle(ax, 11.62, y, "$i$")
        particle(ax, 14.38, y, "$j$")
    label(ax, 13.00, 2.69, "Combined with the usual attention score", 14, MUTED)

    return fig


def validate_layout(fig):
    """Verify all labels are inside the canvas and do not overlap each other."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = [text for ax in fig.axes for text in ax.texts]
    extents = [(text, text.get_window_extent(renderer)) for text in labels]
    for index, (text, extent) in enumerate(extents):
        if not (0 <= extent.x0 <= extent.x1 <= fig.bbox.width
                and 0 <= extent.y0 <= extent.y1 <= fig.bbox.height):
            raise ValueError(f"Label exceeds canvas: {text.get_text()}")
        for other, other_extent in extents[index + 1:]:
            if extent.overlaps(other_extent):
                raise ValueError(
                    f"Labels overlap: {text.get_text()} / {other.get_text()}"
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=SLIDE_ROOT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    validate_layout(fig)
    for extension in ("png", "svg", "pdf"):
        destination = args.output_dir / f"part_pairwise_bias_slide.{extension}"
        metadata = {"Creator": Path(__file__).name}
        if extension == "pdf":
            metadata.update(CreationDate=None, ModDate=None)
        elif extension == "svg":
            metadata["Date"] = None
        fig.savefig(destination, dpi=160, facecolor="white", metadata=metadata)
        print(destination)
    plt.close(fig)


if __name__ == "__main__":
    main()
