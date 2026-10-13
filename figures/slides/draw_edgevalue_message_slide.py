"""Draw a standalone slide explaining how an EdgeValue message is constructed.

Run: python figures/slides/draw_edgevalue_message_slide.py

Requires Matplotlib and the adjacent draw_pairwise_bias_slide.py helpers.
Exports beside this script; does not modify the manuscript or paper figures.
All geometry is schematic, not measured data. The flow illustrates one pair
in one attention head/layer. The learned pair mapping abstracts the shared
pair encoder and layer/head-specific linear value projection. The attention
weight is supplied by the existing attention computation, not by the vector
addition shown here. Aggregation over senders follows attention weighting.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from draw_pairwise_bias_slide import (
    BLUE, MUTED, ORANGE, TEAL, arrow, box, label, particle,
    plt, validate_layout,
)


SLIDE_ROOT = Path(__file__).resolve().parent


def encode_pair(ax):
    label(ax, 2.70, 6.48, "1. Encode the pair", 22, fontweight="bold")
    particle(ax, 1.52, 5.65, "$i$")
    particle(ax, 3.88, 5.65, "$j$")
    ax.plot((1.82, 3.58), (5.65, 5.65), color=MUTED,
            linestyle="--", linewidth=1.4)
    label(ax, 2.70, 4.90, "Pair physics features", 19)
    label(ax, 2.70, 4.43, "Separation, momentum sharing,\npair mass, ...", 15, MUTED)
    arrow(ax, (2.70, 4.03), (2.70, 3.70), color=ORANGE)
    box(ax, 0.94, 2.76, 3.52, 0.88, fill="#fff0e3", edge="#e2baa0")
    label(ax, 2.70, 3.20, "Learned pair mapping", 18, ORANGE,
          fontweight="bold")
    arrow(ax, (2.70, 2.60), (2.70, 2.17), color=ORANGE)
    label(ax, 2.70, 1.85, "$e_{ij}$", 28, ORANGE)
    label(ax, 2.70, 1.40, "Relationship vector", 17, ORANGE)


def build_message(ax):
    label(ax, 7.70, 6.48, "2. Add to the value", 22, fontweight="bold")
    box(ax, 5.95, 5.06, 3.50, 0.98, fill="#eaf0f5", edge="#bdcfdf")
    label(ax, 7.70, 5.74, "Information about particle $j$", 16, BLUE)
    label(ax, 7.70, 5.31, "$v_j$", 25, BLUE)
    label(ax, 7.70, 4.65, "+", 30)
    box(ax, 5.95, 3.22, 3.50, 0.98, fill="#fff0e3", edge="#e2baa0")
    label(ax, 7.70, 3.90, "Information about the pair", 16, ORANGE)
    label(ax, 7.70, 3.47, "$e_{ij}$", 25, ORANGE)
    arrow(ax, (7.70, 3.04), (7.70, 2.53), color=MUTED)
    label(ax, 7.70, 2.11, "$v_j + e_{ij}$", 28)
    label(ax, 7.70, 1.46, "Message content", 17, MUTED)


def deliver_message(ax):
    label(ax, 13.00, 6.48, "3. Weight and send", 22, fontweight="bold")
    label(ax, 13.00, 5.57, "Apply attention weight $a_{ij}$", 20, TEAL)
    box(ax, 11.15, 4.25, 3.70, 0.94, fill="#e8f2ef", edge="#c2d9d1")
    label(ax, 13.00, 4.72, r"$a_{ij}\,(v_j + e_{ij})$", 28, TEAL)
    arrow(ax, (13.00, 4.02), (13.00, 3.26), color=TEAL, width=2.5)
    particle(ax, 13.00, 2.89, "$i$")
    label(ax, 13.00, 2.29, "Receiver $i$", 20)
    label(ax, 13.00, 1.56, "Sums these contributions\nfrom all sender particles", 16, MUTED)


def build_figure():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "svg.hashsalt": "edgevalue-message-slide-v2",
    })
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, 16), ylim=(0, 9), aspect="equal")
    ax.set_axis_off()
    label(ax, 8, 8.25, "How EdgeValue builds a message",
          29, fontweight="bold")
    label(ax, 8, 7.60,
          "Add learned pair information to the particle value, then apply the attention weight.",
          18, MUTED)
    for x, width in ((0.55, 4.30), (5.55, 4.30), (10.55, 4.90)):
        box(ax, x, 1.04, width, 5.93, fill="#f7f9fb")
    arrow(ax, (4.97, 4.05), (5.43, 4.05), color=MUTED)
    arrow(ax, (9.97, 4.05), (10.43, 4.05), color=MUTED)
    encode_pair(ax)
    build_message(ax)
    deliver_message(ax)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=SLIDE_ROOT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    validate_layout(fig)
    for extension in ("png", "svg", "pdf"):
        destination = args.output_dir / f"edgevalue_message_slide.{extension}"
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
