"""Draw the bundled logistic example using the current lsNGC package.

From the repository root, run ``python docs/figures/make_figure.py`` after
``python -m pip install -e '.[demo]'``. The first realization's final 200 samples
and the notebook parameters (d=1, k_f=2, k_g=2, KMeans seed 123) are fixed.
This is a new illustration from the bundled data, not an image from the paper.
"""

from __future__ import annotations

import os
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lsngc import lsNGC
from lsngc.simulate import load_logistic3


def main() -> None:
    """Save true adjacency and recovered affinity with explicit edge direction."""
    series, truth = load_logistic3(realization=0, n_samples=200)
    affinity, _ = lsNGC(series, ar_order=1, k_f=2, k_g=2, normalize=1, seed=123)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "text.color": "#142a38",
        "axes.labelcolor": "#142a38",
        "xtick.color": "#405667",
        "ytick.color": "#405667",
        "figure.facecolor": "#ffffff",
        "savefig.facecolor": "#ffffff",
    })
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 4.7), layout="constrained")
    for axis, matrix, title, upper in zip(
        axes,
        (truth, affinity),
        ("True adjacency", "Recovered lsNGC affinity"),
        (1.0, float(affinity.max())),
    ):
        scale = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=upper)
        axis.set_title(title, fontsize=15, weight="bold", pad=14)
        axis.set_xlabel("Target node", labelpad=9)
        axis.set_ylabel("Source node", labelpad=9)
        axis.set_xticks(np.arange(3), labels=("1", "2", "3"))
        axis.set_yticks(np.arange(3), labels=("1", "2", "3"))
        axis.tick_params(length=0, pad=7)
        axis.set_xticks(np.arange(-0.5, 3, 1), minor=True)
        axis.set_yticks(np.arange(-0.5, 3, 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=2)
        axis.tick_params(which="minor", bottom=False, left=False)
        for spine in axis.spines.values():
            spine.set_visible(False)
        for source in range(3):
            for target in range(3):
                value = matrix[source, target]
                label = f"{value:.0f}" if axis is axes[0] else f"{value:.3f}"
                axis.text(
                    target, source, label, ha="center", va="center",
                    fontsize=13, color="white" if value > upper * 0.58 else "#142a38",
                )
        colorbar = figure.colorbar(scale, ax=axis, fraction=0.046, pad=0.04)
        colorbar.outline.set_visible(False)
        if axis is axes[0]:
            colorbar.set_ticks((0, 1))
    destination = Path(__file__).with_name("lsngc_logistic3.png")
    figure.savefig(destination, dpi=160, metadata={"Software": "lsNGC figure generator"})
    plt.close(figure)
    print(destination)


if __name__ == "__main__":
    main()
