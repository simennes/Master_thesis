"""Single source of truth for thesis figure style and colour palette.

All thesis figures should share one look so the Results chapter reads as a
coherent whole. Import from this module instead of redefining the style:

    from thesis_style import configure_thesis_style, style_axes
    from thesis_style import TRAIT_COLORS, SEMANTIC_COLORS, PALETTE

`scripts/` is on ``sys.path`` when a plot script is run directly, and the
figures notebook adds it explicitly, so the bare ``import thesis_style`` works
in both contexts.

House style
-----------
* Serif text (Times New Roman) with STIX math, matching the thesis body.
* Seaborn ``whitegrid`` with the top/right spines removed and a faint grid.
* Qualitative cycle: seaborn ``"colorblind"`` (``PALETTE``).
* Colour by trait whenever the three traits appear in separate panels or
  series, using the fixed ``TRAIT_COLORS``: body mass = blue, tarsus = green,
  wing = orange.
* Use the accent pair ``SEMANTIC_COLORS["observed"]`` / ``["adjusted"]`` only
  when two series share one panel and must be told apart by colour (for
  example male vs. female overlaid in the same axes).
* Reference lines (means, identities, annotations) use
  ``SEMANTIC_COLORS["reference"]``.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Qualitative cycle for categorical figures (islands, methods, ...).
PALETTE = "colorblind"

# Fixed colours for the three traits, used across every figure that shows them.
TRAIT_COLORS = {
    "Body mass": "#4C78A8",      # blue
    "Tarsus length": "#59A14F",  # green
    "Wing length": "#F28E2B",    # orange
}

# Accent pair for two series sharing one panel (e.g. male vs. female).
SEMANTIC_COLORS = {
    "observed": "#4C78A8",   # first series (blue)
    "adjusted": "#E45756",   # second series (red)
    "reference": "#333333",  # mean / identity / annotation lines
    "context": "#C7C7C7",    # de-emphasised background points
}

# Figure widths (inches) for the single-column thesis text block.
FULL_WIDTH = 6.7
MAIN_WIDTH = 0.95 * FULL_WIDTH
HALF_WIDTH = 3.3


def configure_thesis_style() -> None:
    """Apply the shared seaborn/matplotlib style. Call once before plotting."""
    sns.set_theme(
        context="paper",
        style="whitegrid",
        palette=PALETTE,
        font="Times New Roman",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.55,
        },
    )
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "normal",
            "axes.titlepad": 8,
            "axes.labelsize": 12.5,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10.4,
            "legend.title_fontsize": 11,
            "figure.titlesize": 15,
            "figure.titleweight": "normal",
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.8,
            "mathtext.fontset": "stix",
            "text.usetex": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axes(ax: plt.Axes) -> None:
    """Apply the shared per-axes grid and spine treatment."""
    ax.grid(True, alpha=0.22, linewidth=0.55)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)
