"""Matplotlib figures used by evaluation and the report (all drawn from real result files)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SHORT = {"included": "incl", "not included": "not incl", "excluded": "excl", "not excluded": "not excl",
         "not enough information": "NEI", "not applicable": "N/A", "invalid": "invalid"}


def confusion_png(conf: dict, path: Path, title: str = "", figsize: tuple | None = None) -> None:
    labels = conf["labels"]
    m = np.array(conf["matrix"])
    fig, ax = plt.subplots(figsize=figsize or (0.75 * len(labels) + 1.8, 0.65 * (len(labels) - 1) + 1.6))
    ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(len(labels)), [SHORT.get(x, x) for x in labels], rotation=35, ha="right")
    ax.set_yticks(range(m.shape[0]), [SHORT.get(x, x) for x in labels[: m.shape[0]]])
    vmax = m.max() if m.size else 1
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, int(m[i, j]), ha="center", va="center",
                    color="white" if m[i, j] > vmax / 2 else "black", fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("gold (expert)")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
