"""Figures for the baseline results."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

from ror.config import GENERATORS, RESULTS

FIGURES = RESULTS / "figures"

NAMES = {
    "stable_diffusion_v_1_4": "SD v1.4\n(training)",
    "stable_diffusion_v_1_5": "SD v1.5",
    "wukong": "Wukong",
    "Midjourney": "Midjourney",
    "ADM": "ADM",
    "glide": "GLIDE",
    "VQDM": "VQDM",
    "BigGAN": "BigGAN",
}
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, TEXT, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
BAR_WIDTH = 0.36

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "text.color": TEXT,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": AXIS,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 1,
        "axes.axisbelow": True,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
    }
)


def accuracy_columns(series, title, path):
    """Grouped columns of accuracy per test generator. series: {label: (color, per-seed frame)}."""
    order = list(NAMES)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    offsets = (np.arange(len(series)) - (len(series) - 1) / 2) * (BAR_WIDTH + 0.03)
    for offset, (label, (color, frame)) in zip(offsets, series.items()):
        per_seed = frame.pivot(index="generator", columns="seed", values="acc").loc[order]
        mean, std = per_seed.mean(axis=1), per_seed.std(axis=1).fillna(0)
        x = np.arange(len(order)) + offset
        ax.bar(x, mean, BAR_WIDTH, color=color, label=label, yerr=std if std.any() else None,
               error_kw={"ecolor": TEXT, "elinewidth": 1, "capsize": 2})
        for xi, value in zip(x, mean):
            ax.text(xi, value + 1.5, f"{value:.0f}", ha="center", va="bottom", fontsize=9, color=MUTED)
    ax.axhline(50, color=AXIS, linewidth=1)
    ax.text(1.0, 50, " chance", transform=ax.get_yaxis_transform(), va="center", fontsize=9, color=MUTED)
    ax.set_xticks(np.arange(len(order)), [NAMES[g] for g in order])
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title, loc="left", fontsize=13, pad=34)
    ax.legend(frameon=False, loc="lower left", ncols=len(series), bbox_to_anchor=(0, 1.0), borderaxespad=0)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def roc_small_multiples(probs, title, path):
    """One ROC panel per test generator."""
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.8), sharex=True, sharey=True)
    for ax, generator in zip(axes.flat, NAMES):
        fpr, tpr, _ = roc_curve(probs[f"{generator}/label"], probs[f"{generator}/resize"])
        ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1)
        ax.plot(fpr, tpr, color=ORANGE, linewidth=2)
        ax.set_title(NAMES[generator].replace("\n", " "), fontsize=11, loc="left")
        ax.text(0.97, 0.05, f"AUC {auc(fpr, tpr):.3f}", ha="right", fontsize=10, color=TEXT)
        ax.set_aspect("equal")
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.grid(True, axis="both")
    fig.supxlabel("False positive rate (real images called AI)", fontsize=11, color=MUTED)
    fig.supylabel("True positive rate (AI images caught)", fontsize=11, color=MUTED)
    fig.suptitle(title, x=0.02, ha="left", fontsize=13)
    fig.tight_layout(h_pad=2)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(RESULTS / "baseline.csv")
    assert set(results["generator"]) == set(GENERATORS)

    def pick(model, protocol):
        return results[(results["model"] == model) & (results["protocol"] == protocol)]

    accuracy_columns(
        {
            "Released weights (trained on ProGAN)": (BLUE, pick("released_progan", "resize")),
            "Our baseline (retrained on SD v1.4)": (ORANGE, pick("retrained_sd14", "resize")),
        },
        "UnivFD accuracy on each GenImage generator",
        FIGURES / "baseline_accuracy.png",
    )
    accuracy_columns(
        {
            "Resize to 256, then crop 224": (ORANGE, pick("retrained_sd14", "resize")),
            "Crop 224 only (repo's validate.py)": (AQUA, pick("retrained_sd14", "crop")),
        },
        "Same model, two test preprocessing protocols",
        FIGURES / "baseline_resize_vs_crop.png",
    )
    roc_small_multiples(
        np.load(RESULTS / "probs_retrained_sd14_seed0.npz"),
        "ROC curves of our baseline (seed 0) on each GenImage generator",
        FIGURES / "baseline_roc.png",
    )
    print(f"figures written to {FIGURES}")


if __name__ == "__main__":
    main()
