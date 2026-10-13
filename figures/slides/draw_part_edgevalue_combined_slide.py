"""Draw the combined ParT + EdgeValue mechanism for a presentation slide.

Schematic for one attention head/layer, not measured data. Pair features
feed learned mappings to both a value vector and a scalar score bias.
The ordinary query-key score is s_ij; softmax is over sender particles j.
The same resulting weight multiplies both terms of the enriched message.
This abstracts the shared pair stem and layer/head-specific projections;
it does not imply that these branches must execute sequentially.

Run: python figures/slides/draw_part_edgevalue_combined_slide.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from draw_pairwise_bias_slide import (
    BLUE, MUTED, ORANGE, TEAL, arrow, box, label, particle,
    plt, validate_layout,
)


SLIDE_ROOT = Path(__file__).resolve().parent


def build_figure():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "svg.hashsalt": "part-edgevalue-combined-slide-v1",
    })
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, 16), ylim=(0, 9), aspect="equal")
    ax.set_axis_off()

    label(ax, 8, 8.25, "ParT + EdgeValue: richer messages, learned strength",
          28, fontweight="bold")
    label(ax, 8, 7.60,
          "Pair information shapes both what is sent and how strongly it is weighted.",
          19, MUTED)

    # Pair features branch into two learned transformations.
    box(ax, 0.45, 3.10, 3.20, 2.60, fill="#f0f4f8")
    particle(ax, 1.35, 5.15, "$i$")
    particle(ax, 2.75, 5.15, "$j$")
    ax.plot((1.65, 2.45), (5.15, 5.15), color=MUTED,
            linewidth=1.4, linestyle="--")
    label(ax, 2.05, 4.46, "Pair physics features", 18, fontweight="bold")
    label(ax, 2.05, 3.73, "Separation, pair mass,\nmomentum sharing, ...", 16, MUTED)

    ax.plot((3.65, 4.05, 4.05), (4.65, 4.65, 5.75),
            color=ORANGE, linewidth=2)
    arrow(ax, (4.05, 5.75), (4.55, 5.75), color=ORANGE)
    ax.plot((3.65, 4.05, 4.05), (4.05, 4.05, 2.75),
            color=TEAL, linewidth=2)
    arrow(ax, (4.05, 2.75), (4.55, 2.75), color=TEAL)

    # Content branch: the pair mapping produces a vector added to v_j.
    box(ax, 4.65, 4.75, 5.95, 2.22, fill="#fff0e3", edge="#e2baa0")
    label(ax, 7.625, 6.51, "EdgeValue: enrich the message", 21, ORANGE,
          fontweight="bold")
    label(ax, 7.625, 5.95, "Particle value + learned pair vector", 17)
    label(ax, 7.625, 5.28, "$v_j + e_{ij}$", 31, ORANGE)

    # Weight branch: the pair bias adjusts QK logits, not values directly.
    box(ax, 4.65, 1.43, 5.95, 2.80, fill="#e8f2ef", edge="#c2d9d1")
    label(ax, 7.625, 3.78, "ParT: adjust attention strength", 21, TEAL,
          fontweight="bold")
    label(ax, 7.625, 3.19, "Usual QK score + learned pair bias", 17)
    label(ax, 7.625, 2.52, r"$a_{ij}=\mathrm{softmax}_{j}(s_{ij}+b_{ij})$", 25, TEAL)
    label(ax, 7.625, 1.83, "Normalized across sender particles", 16, MUTED)

    # Both branches meet: a_ij multiplies the entire enriched message.
    arrow(ax, (10.73, 5.72), (11.77, 5.10), color=ORANGE, width=2.5)
    arrow(ax, (10.73, 2.73), (11.77, 4.20), color=TEAL, width=2.5)
    box(ax, 11.88, 3.82, 3.68, 2.28, fill="#f0f4f8", edge="#bdcfdf")
    label(ax, 13.72, 5.48, "Weight the\nwhole message", 20, fontweight="bold")
    label(ax, 13.72, 4.47, r"$a_{ij}\,(v_j+e_{ij})$", 26, BLUE)
    arrow(ax, (13.72, 3.66), (13.72, 2.94), color=BLUE, width=2.5)
    particle(ax, 13.72, 2.58, "$i$")
    label(ax, 13.72, 1.97, "Receiver $i$", 19, BLUE)
    label(ax, 13.72, 1.26, "Sums weighted messages\nfrom all senders", 16, MUTED)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=SLIDE_ROOT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    validate_layout(fig)
    for extension in ("png", "svg", "pdf"):
        destination = args.output_dir / f"part_edgevalue_combined_slide.{extension}"
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
