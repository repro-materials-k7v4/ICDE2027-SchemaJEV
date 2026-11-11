"""Shared visual style for every experiment figure in the paper.

All figures are sized in inches to the IEEEtran two-column geometry and are
meant to be included at their natural width (\\columnwidth or \\textwidth), so
font sizes in the PDF match the body text exactly:

* body text 10 pt, captions 8 pt  ->  figure labels 8 pt, ticks/legends 7 pt
* Times-compatible TrueType serif (Times New Roman -> Liberation Serif -> STIX)
* thin axes (0.6 pt), 1.2 pt data lines, 0.8 pt error bars

Colour semantics are fixed across the whole paper:

* decision type   choice = blue, binary = orange, score = green
* split           ID = filled marker / solid line, OOD = hollow marker / dashed line
* our method      dark red accent;  references (LoRA, bare) = greys
"""
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
TAB_DIR = ROOT / "tables"

# Display name of the proposed method. Change it here to rename every figure.
METHOD = "PCA-T2L"

# IEEEtran geometry (inches)
COL_W = 3.50
TEXT_W = 7.16

# ---------------------------------------------------------------- palette ----
INK = "#1F1F1F"
GREY_DARK = "#4D4D4D"
GREY = "#8C8C8C"
GREY_LIGHT = "#D9D9D9"
GRID = "#E9E9E9"
OURS = "#B2182B"

TYPE_COLOR = {
    "choice": "#2F6CAB",  # blue
    "binary": "#D9822B",  # orange
    "score": "#2E8B6E",   # green
}
TYPE_MARKER = {"choice": "o", "binary": "s", "score": "D"}
TYPE_NAME = {"choice": "Choice", "binary": "Binary", "score": "Score"}
TYPES = ("choice", "binary", "score")

# Tints used for secondary elements of the same hue (e.g. OOD bars)
TYPE_TINT = {
    "choice": "#A9C4E2",
    "binary": "#F0C79C",
    "score": "#A5D3C4",
}

_FONT_DIRS = (
    "/usr/share/texmf/fonts/opentype/public/tex-gyre",
    "/usr/share/texlive/texmf-dist/fonts/opentype/public/tex-gyre",
    "/usr/local/texlive/texmf-dist/fonts/opentype/public/tex-gyre",
    "/Library/TeX/Root/texmf-dist/fonts/opentype/public/tex-gyre",
)


def _register_fonts():
    for directory in _FONT_DIRS:
        folder = Path(directory)
        if folder.is_dir():
            for font in folder.glob("texgyretermes-*.otf"):
                font_manager.fontManager.addfont(str(font))


def apply():
    mpl.rcParams.update({
        "font.family": "serif",
        # TrueType Times-metric fonts only: OpenType/CFF fonts embed badly with
        # matplotlib and are flagged by IEEE PDF eXpress.
        "font.serif": ["Times New Roman", "Liberation Serif", "Tinos", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "axes.edgecolor": GREY_DARK,
        "axes.labelcolor": INK,
        "axes.titlepad": 3.0,
        "axes.labelpad": 2.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "xtick.color": GREY_DARK,
        "ytick.color": GREY_DARK,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.pad": 1.8,
        "ytick.major.pad": 1.8,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "lines.linewidth": 1.2,
        "lines.markersize": 3.6,
        "lines.markeredgewidth": 0.8,
        "errorbar.capsize": 1.6,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.4,
        "legend.columnspacing": 1.0,
        "legend.borderaxespad": 0.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
        "figure.dpi": 150,
    })


def panel_label(ax, text, x=-0.02, y=1.02):
    """Bold '(a)'-style label followed by a plain panel title, left aligned."""
    ax.set_title(text, loc="left", x=x, y=y, fontsize=8)


def save(fig, stem):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.pdf")
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300)
    plt.close(fig)
