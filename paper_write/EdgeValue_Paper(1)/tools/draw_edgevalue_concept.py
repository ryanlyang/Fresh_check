"""Draw the paper's editable, conceptual ParT / ParT-EV comparison.

Run with a Python environment containing Matplotlib:
    python "paper_write/EdgeValue_Paper(1)/tools/draw_edgevalue_concept.py"

Exports vector PDF/SVG and a PNG preview. Particle positions, arrow widths,
and colours illustrate the mechanism only; they are not measured event data
or attention values. No training artifacts or evaluation results are read.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch


PAPER_ROOT = Path(__file__).resolve().parents[1]
INK = "#203343"
MUTED = "#647582"
BLUE = "#31688e"
ORANGE = "#b45b27"
TEAL = "#178075"
FAINT = "#cbd5dc"


def label(ax, x, y, text, size=12, color=INK, **kwargs):
    return ax.text(x, y, text, ha="center", va="center", fontsize=size,
                   color=color, zorder=7, **kwargs)


def node(ax, x, y, symbol, *, context=False):
    box = ax.get_position()
    width, height = ax.figure.get_size_inches()
    aspect = (box.width * width) / (box.height * height)
    ax.add_patch(Ellipse((x, y), 0.080, 0.080 * aspect,
                         facecolor=BLUE if context else "white",
                         edgecolor=BLUE if context else INK, linewidth=1.6,
                         zorder=5))
    label(ax, x, y, symbol, 15, "white" if context else INK)


def content_card(ax, x, receiver, *, edgevalue, color):
    """Show pre-weight content separately from the scalar attention weight."""
    width = 0.40
    y = 0.280
    ax.add_patch(FancyBboxPatch((x, y), width, 0.075,
                               boxstyle="round,pad=0.006,rounding_size=0.012",
                               facecolor="white", edgecolor=FAINT, linewidth=1))
    label(ax, x + width / 2, 0.382, rf"To query ${receiver}$", 11, MUTED)
    if edgevalue:
        label(ax, x + 0.100, y + 0.0375, r"$v_j$", 16, BLUE)
        label(ax, x + 0.200, y + 0.0375, "+", 15)
        label(ax, x + 0.303, y + 0.0375, rf"$e_{{{receiver}j}}$", 16, color)
    else:
        label(ax, x + width / 2, y + 0.0375, r"$v_j$", 16, BLUE)


def panel(ax, *, edgevalue):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.add_patch(FancyBboxPatch((0.005, 0.012), 0.990, 0.975,
                               boxstyle="round,pad=0,rounding_size=0.02",
                               facecolor="#f8fafb", edgecolor="#dce3e8",
                               linewidth=1.1))
    label(ax, 0.50, 0.948, "(b) ParT-EV" if edgevalue else "(a) ParT", 17,
          fontweight="bold")
    label(ax, 0.50, 0.902,
          "Pair information also changes message content" if edgevalue
          else "Pair information changes attention weights", 10.5)

    # Both panels show exactly the same schematic jet constituent arrangement.
    ax.add_patch(Ellipse((0.50, 0.662), 0.87, 0.424, facecolor="white",
                         edgecolor=FAINT, linewidth=1.0, linestyle=(0, (3, 3))))
    others = [(0.17, 0.67), (0.25, 0.78), (0.40, 0.70), (0.66, 0.79),
              (0.75, 0.67), (0.37, 0.54), (0.60, 0.55), (0.53, 0.62),
              (0.33, 0.62), (0.68, 0.59)]
    ax.scatter([x for x, _ in others], [y for _, y in others],
               s=[13, 18, 12, 20, 15, 12, 20, 13, 10, 12],
               color=FAINT, zorder=2)
    origin = (0.50, 0.800)
    queries = (("i", 0.20, ORANGE), ("k", 0.80, TEAL))
    for receiver, x, edge_color in queries:
        # Equal line widths deliberately avoid implying observed weights.
        arrow = FancyArrowPatch(origin, (x, 0.522), arrowstyle="-|>",
                                mutation_scale=17, shrinkA=12, shrinkB=15,
                                linewidth=2.0,
                                color=edge_color if edgevalue else BLUE,
                                zorder=3)
        ax.add_patch(arrow)
        label(ax, 0.265 if receiver == "i" else 0.735, 0.718,
              rf"$a_{{{receiver}j}}$", 13,
              bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
        node(ax, x, 0.522, rf"${receiver}$")
        content_card(ax, 0.065 if receiver == "i" else 0.535,
                     receiver, edgevalue=edgevalue, color=edge_color)
    node(ax, *origin, "$j$", context=True)
    label(ax, 0.50, 0.855, "Context particle", 10.5, MUTED)
    label(ax, 0.50, 0.432, "MESSAGE CONTENT BEFORE WEIGHTING", 9.5, MUTED,
          fontweight="bold")

    label(ax, 0.50, 0.233,
          "Pair-specific addition to the same context value" if edgevalue
          else "The same context value for either receiver", 10.1)
    label(ax, 0.50, 0.185, "Contribution to query $i$", 10.5, MUTED)
    label(ax, 0.50, 0.139,
          r"$a_{ij}\left(v_j+e_{ij}\right)$" if edgevalue
          else r"$a_{ij}\,v_j$", 18)
    label(ax, 0.50, 0.065,
          r"$e_{ij}=U_{\ell h}z_{ij}$  (learned pair-dependent vector)" if edgevalue
          else "The attention weights can still differ: $a_{ij}$, $a_{kj}$", 10.5)


def build_figure():
    plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                         "svg.hashsalt": "part-ev-concept-v1"})
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 5.8))
    fig.subplots_adjust(left=0.012, right=0.988, top=0.985, bottom=0.065, wspace=0.035)
    panel(axes[0], edgevalue=False)
    panel(axes[1], edgevalue=True)
    fig.text(0.5, 0.027,
             "Schematic jet constituents  |  One attention head and layer  |  Only two interactions shown",
             ha="center", va="center", fontsize=10, color=MUTED)
    return fig


def validate_layout(fig):
    """Fail if a label escapes its panel or the figure canvas."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        bounds = ax.get_window_extent(renderer)
        for text in ax.texts:
            extent = text.get_window_extent(renderer)
            if (extent.x0 < bounds.x0 or extent.x1 > bounds.x1
                    or extent.y0 < bounds.y0 or extent.y1 > bounds.y1):
                raise ValueError(f"Text extends beyond panel: {text.get_text()}")
    for text in fig.texts:
        extent = text.get_window_extent(renderer)
        if not (0 <= extent.x0 <= extent.x1 <= fig.bbox.width
                and 0 <= extent.y0 <= extent.y1 <= fig.bbox.height):
            raise ValueError(f"Text extends beyond figure: {text.get_text()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PAPER_ROOT / "figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    validate_layout(fig)
    for extension in ("pdf", "svg", "png"):
        destination = args.output_dir / f"edgevalue_concept.{extension}"
        metadata = {"Creator": "draw_edgevalue_concept.py"}
        if extension == "pdf":
            metadata.update(CreationDate=None, ModDate=None)
        elif extension == "svg":
            metadata["Date"] = None
        fig.savefig(destination, dpi=220, facecolor="white", metadata=metadata)
        print(destination)
    plt.close(fig)


if __name__ == "__main__":
    main()
