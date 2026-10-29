"""Shared figure style."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
RESULTS_DIR = os.path.join(os.path.dirname(FIG_DIR), "results")

plt.rcParams.update({
    "font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
    "figure.figsize": (5.5, 3.8), "savefig.dpi": 200, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"))
    plt.close(fig)


def results_path(name):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, name)
