"""Draw the ParT-EV architecture and its added value-message route.

    python "paper_write/EdgeValue_Paper(1)/tools/draw_edgevalue_architecture.py"

Original Matplotlib vector diagram, with PDF/SVG and PNG preview exports.
The lower panel depicts evaluation-time attention for one head. It uses the
equivalent efficient operation U(sum_j a_ij z_ij), not materialized per-head
pair-value tensors. Surrounding residual/normalization/FFN paths are omitted.
No evaluation measurements or source results are read or changed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath


PAPER_ROOT = Path(__file__).resolve().parents[1]
INK = "#243746"
MUTED = "#63727f"
BLUE_FILL = "#eaf0f5"
GREEN_FILL = "#edf3ef"
PARTICLE_FILL = "#eef0f8"
CLASS_FILL = "#eee6d8"
ORANGE = "#b4511d"
ORANGE_FILL = "#fff0e3"
FAINT = "#c9d3dc"


def label(ax, x, y, text, size=12, color=INK, *, ha="center", **kwargs):
    return ax.text(x, y, text, ha=ha, va="center", fontsize=size,
                   color=color, zorder=6, **kwargs)


def box(ax, x, y, width, height, text="", *, fill=BLUE_FILL,
        edge=INK, size=12, color=INK, zorder=3):
    ax.add_patch(FancyBboxPatch((x, y), width, height,
                               boxstyle="round,pad=0,rounding_size=0.10",
                               facecolor=fill, edgecolor=edge, linewidth=1.25,
                               zorder=zorder))
    if text:
        label(ax, x + width / 2, y + height / 2, text, size, color)


def wire(ax, points, *, color=INK, arrow=True, dashed=False, width=1.35):
    """Route an explicit orthogonal path; only the terminal segment has an arrow."""
    path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1))
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>" if arrow else "-",
                                mutation_scale=11.5, linewidth=width, color=color,
                                linestyle=(0, (3, 3)) if dashed else "solid", zorder=2))


def junction(ax, x, y, color=INK):
    ax.plot([x], [y], marker="o", markersize=3.1, color=color, zorder=4)


def plus(ax, x, y, radius=0.21):
    bounds = ax.get_position()
    fig_width, fig_height = ax.figure.get_size_inches()
    x_unit = bounds.width * fig_width / (ax.get_xlim()[1] - ax.get_xlim()[0])
    y_unit = bounds.height * fig_height / (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.add_patch(Ellipse((x, y), 2 * radius, 2 * radius * x_unit / y_unit,
                         facecolor="white", edgecolor=INK, linewidth=1.25, zorder=4))
    label(ax, x, y, "+", 15)


def overview(ax):
    label(ax, 0.15, 9.51, "(a) ParT-EV for jet tagging", 14, ha="left", fontweight="bold")
    label(ax, 0.64, 8.025, "Particle\nfeatures", 12)
    box(ax, 1.38, 7.575, 1.30, 0.90, "Particle\nembedding", fill=GREEN_FILL, size=11.5)
    wire(ax, [(1.12, 8.025), (1.38, 8.025)])
    wire(ax, [(2.68, 8.025), (3.20, 8.025)])
    for x in (3.20, 5.20, 7.20):
        box(ax, x, 7.475, 1.45, 1.10, fill=PARTICLE_FILL)
        label(ax, x + 0.725, 8.14, "Particle\nattention", 12)
        label(ax, x + 0.725, 7.73, "+ EdgeValue", 10.5, ORANGE, fontweight="bold")
    wire(ax, [(4.65, 8.025), (5.20, 8.025)])
    wire(ax, [(6.65, 8.025), (7.20, 8.025)], dashed=True)
    wire(ax, [(8.65, 8.025), (9.20, 8.025)])
    wire(ax, [(3.20, 8.72), (3.20, 8.88), (8.65, 8.88), (8.65, 8.72)], arrow=False)
    label(ax, 5.925, 9.075, "8 particle-attention blocks", 11.5)

    box(ax, 9.20, 7.475, 1.55, 1.10, "Class\nattention\n" + r"$\times\,2$", fill=CLASS_FILL, size=11.5)
    label(ax, 9.975, 9.08, "Class token", 11, color="#775320")
    wire(ax, [(9.975, 8.88), (9.975, 8.575)], color="#775320")
    box(ax, 11.20, 7.625, 0.90, 0.80, "Linear", fill=GREEN_FILL, size=11.5)
    box(ax, 12.55, 7.625, 1.15, 0.80, "Softmax", fill="white", size=11.5)
    wire(ax, [(10.75, 8.025), (11.20, 8.025)])
    wire(ax, [(12.10, 8.025), (12.55, 8.025)])
    wire(ax, [(13.70, 8.025), (13.97, 8.025)])
    label(ax, 13.10, 7.25, "Jet classes", 11, MUTED)
    label(ax, 11.30, 6.53, "Unchanged class readout", 11, MUTED)

    label(ax, 0.64, 6.50, "Pair\nfeatures", 12)
    box(ax, 1.38, 6.05, 1.30, 0.90, "Pair\nembedding", fill=GREEN_FILL, size=11.5)
    wire(ax, [(1.12, 6.50), (1.38, 6.50)])
    wire(ax, [(2.68, 6.50), (7.925, 6.50)], arrow=False)
    for x in (3.925, 5.925, 7.925):
        junction(ax, x, 6.50)
        wire(ax, [(x, 6.50), (x, 7.475)])
    label(ax, 5.50, 6.11, "Shared pair stem $z_{ij}$: bias + value messages", 11)


def attention_detail(ax):
    label(ax, 0.15, 5.31, "(b) Particle attention: one head shown", 14,
          ha="left", fontweight="bold")
    label(ax, 13.83, 5.31, "Orange = added EdgeValue route", 11, ORANGE, ha="right")

    label(ax, 0.85, 3.86, r"$q_i,\ k_j$", 15)
    label(ax, 0.85, 3.32, "Projected\nparticle states", 10.5, MUTED)
    box(ax, 2.0, 3.43, 1.80, 0.86, r"$q_i^{\mathsf{T}}k_j/\sqrt{d_h}$", size=14)
    wire(ax, [(1.41, 3.86), (2.00, 3.86)])
    plus(ax, 4.48, 3.86)
    wire(ax, [(3.80, 3.86), (4.27, 3.86)])
    box(ax, 5.17, 3.43, 1.40, 0.86, "Masked\nsoftmax", size=11.5)
    wire(ax, [(4.69, 3.86), (5.17, 3.86)])
    wire(ax, [(6.57, 3.86), (8.18, 3.86)])
    label(ax, 7.34, 4.13, "$a_{ij}$", 14)
    junction(ax, 7.30, 3.86)

    box(ax, 8.18, 3.43, 2.34, 0.86, r"$\sum_j a_{ij}v_j$", size=16)
    label(ax, 9.35, 4.64, "Values $v_j$", 12)
    wire(ax, [(9.35, 4.47), (9.35, 4.29)])
    wire(ax, [(10.52, 3.86), (11.46, 3.86), (11.46, 2.985)])

    label(ax, 0.85, 1.23, "$z_{ij}$", 16)
    label(ax, 0.85, 0.78, "Pair stem", 10.5, MUTED)
    wire(ax, [(1.37, 1.23), (2.19, 1.23)], arrow=False)
    junction(ax, 2.19, 1.23)
    box(ax, 3.25, 1.90, 2.46, 0.80, r"Bias $b_{ij}=\beta(z_{ij})$", size=11.5)
    wire(ax, [(2.19, 1.23), (2.19, 2.30), (3.25, 2.30)])
    wire(ax, [(4.48, 2.70), (4.48, 3.635)])

    # Efficient new route: weighted relation sum, then a layer/head-specific map.
    box(ax, 5.16, 0.77, 2.42, 0.92, r"$\sum_j a_{ij}z_{ij}$",
        fill=ORANGE_FILL, edge=ORANGE, color=ORANGE, size=16)
    box(ax, 8.18, 0.77, 2.34, 0.92, "Value projection\n$U_{\\ell h}$",
        fill=ORANGE_FILL, edge=ORANGE, color=ORANGE, size=12)
    wire(ax, [(2.19, 1.23), (5.16, 1.23)], color=ORANGE, width=1.8)
    wire(ax, [(7.30, 3.86), (7.30, 2.77), (6.37, 2.77), (6.37, 1.69)],
         color=ORANGE, width=1.8)
    wire(ax, [(7.58, 1.23), (8.18, 1.23)], color=ORANGE, width=1.8)
    label(ax, 8.28, 0.47, "New pair-conditioned value contribution", 11, ORANGE)
    wire(ax, [(10.52, 1.23), (11.46, 1.23), (11.46, 2.535)], color=ORANGE, width=1.8)
    label(ax, 11.04, 0.91, r"$\delta_i$", 14, ORANGE)
    plus(ax, 11.46, 2.76)
    box(ax, 12.13, 2.16, 1.55, 1.20, "Concat heads\n+ output\nprojection", size=10.5)
    wire(ax, [(11.67, 2.76), (12.13, 2.76)])
    wire(ax, [(13.68, 2.76), (13.93, 2.76)])


def build_figure():
    plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                         "svg.hashsalt": "part-ev-architecture-v2"})
    fig, ax = plt.subplots(figsize=(10.8, 6.7))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.015)
    ax.set_xlim(0, 14.1)
    ax.set_ylim(0.10, 9.79)
    ax.set_axis_off()
    overview(ax)
    attention_detail(ax)
    return fig


def validate_layout(fig):
    """Fail if any text extends beyond the canvas, including exported math."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        bounds = ax.get_window_extent(renderer)
        for text in ax.texts:
            extent = text.get_window_extent(renderer)
            if (extent.x0 < bounds.x0 or extent.x1 > bounds.x1
                    or extent.y0 < bounds.y0 or extent.y1 > bounds.y1):
                raise ValueError(f"Text extends beyond panel: {text.get_text()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PAPER_ROOT / "figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    validate_layout(fig)
    for extension in ("pdf", "svg", "png"):
        destination = args.output_dir / f"edgevalue_architecture.{extension}"
        metadata = {"Creator": "draw_edgevalue_architecture.py"}
        if extension == "pdf":
            metadata.update(CreationDate=None, ModDate=None)
        elif extension == "svg":
            metadata["Date"] = None
        fig.savefig(destination, dpi=220, facecolor="white", metadata=metadata)
        print(destination)
    plt.close(fig)


if __name__ == "__main__":
    main()
