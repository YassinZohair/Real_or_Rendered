"""Milestone 2 figures: every model compared with B. Run through python -m ror.plots m2.

Damaged-image numbers for B and B+U2 come from all 2,000 + 2,000 images, the others from the
1,000 + 1,000 subset; the paired version of each comparison is in ror.analysis.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ror import collect
from ror.config import RESULTS, TRAIN_GENERATOR
from ror.plots import (
    AXIS,
    FIGURES,
    MUTED,
    SURFACE,
    TEXT,
    accuracy_columns,
    robustness_curve,
)

# One colour per model, fixed by name, in every figure (a missing model never shifts the others).
# The reference palette's first six slots, in this order, pass the colour-vision checks
# (validate_palette.js: CVD separation >= 9, normal vision >= 19.6).
COLORS = {"B": "#2a78d6", "B+U2": "#eb6834", "B+U1": "#1baf7a", "B+U1+U2": "#eda100", "B+U3": "#e87ba4",
          "Full (U1+U2+U3)": "#008300"}
REAL_GREY, AI_GREY = "#a9a79f", "#3a3936"


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=200)
    plt.close(fig)


def summary_bars(results, colors):
    """Clean test images: mean accuracy and AUC over the 8 generators, one bar per model, seed spread."""
    clean = results[results["protocol"] == "resize"]
    per_seed = clean.groupby(["model", "seed"], sort=False)[["acc", "auc"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, metric, title in zip(axes, ["acc", "auc"], ["Accuracy (cutoff 0.5)", "ROC-AUC (ranking, no cutoff)"]):
        stats = per_seed[metric].groupby("model", sort=False).agg(["mean", "std"])
        x = np.arange(len(stats))
        ax.bar(x, stats["mean"], 0.62, color=[colors[m] for m in stats.index], yerr=stats["std"].fillna(0),
               error_kw={"ecolor": TEXT, "elinewidth": 1, "capsize": 3})
        for xi, value in zip(x, stats["mean"]):
            ax.text(xi, value + 1.2, f"{value:.1f}", ha="center", va="bottom", fontsize=10, color=MUTED)
        ax.set_xticks(x, [m.replace(" (", "\n(") for m in stats.index], fontsize=9)
        low = np.floor(stats["mean"].min() / 5) * 5 - 10
        ax.set_ylim(low, 100)
        ax.set_title(title, loc="left", fontsize=12)
        ax.set_ylabel("Mean over 8 generators (%)")
    fig.suptitle("Clean test images, every model (bars: mean of 3 seeds; whiskers: seed spread)",
                 x=0.01, ha="left", fontsize=13)
    _save(fig, "m2_summary.png")


def real_fake(results, labels):
    """The cutoff story: how often real and AI images are called correctly, per model."""
    clean = results[results["protocol"] == "resize"].groupby("model", sort=False)[["real_acc", "fake_acc"]].mean()
    clean = clean.loc[labels]
    x = np.arange(len(clean))
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for offset, column, color, name in [(-0.19, "real_acc", REAL_GREY, "Real images called real"),
                                        (0.19, "fake_acc", AI_GREY, "AI images called AI")]:
        ax.bar(x + offset, clean[column], 0.36, color=color, label=name)
        for xi, value in zip(x + offset, clean[column]):
            ax.text(xi, value + 1.2, f"{value:.0f}", ha="center", va="bottom", fontsize=10, color=MUTED)
    ax.set_xticks(x, [m.replace(" (", "\n(") for m in clean.index], fontsize=9)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Accuracy, mean over 8 generators (%)")
    ax.set_title("Where the errors are: real vs AI images, clean test set", loc="left", fontsize=13, pad=30)
    ax.legend(frameon=False, loc="lower left", ncols=2, bbox_to_anchor=(0, 1.0), borderaxespad=0)
    _save(fig, "m2_real_vs_ai.png")


def training_curves():
    """Validation accuracy during training: U1 runs by step, U3 runs by epoch. Step 0 = the starting model."""
    # (legend label, results name, model colour, line style): colour = model, dashes = concat fusion
    panels = [("U1 (LoRA): validation accuracy by training step", "step",
               [("B+U1", f"u1_r{collect.U1_RANK}", "B+U1", "-"),
                ("B+U1+U2", f"u1u2_r{collect.U1_RANK}", "B+U1+U2", "-")]),
              ("U3 (noise branch): validation accuracy by epoch", "epoch",
               [("B+U3, gate", "u3_gate", "B+U3", "-"), ("B+U3, concat", "u3_concat", "B+U3", "--"),
                ("Full, gate", "full_gate", "Full (U1+U2+U3)", "-"),
                ("Full, concat", "full_concat", "Full (U1+U2+U3)", "--")])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (title, x_name, runs) in zip(axes, panels):
        for label, name, model, style in runs:
            for i, log in enumerate(collect.training_logs(name).values()):  # one line per seed
                ax.plot(log[x_name], log["val_acc"] * 100, color=COLORS[model], linestyle=style,
                        linewidth=2 if i == 0 else 1, label=label if i == 0 else None, marker="o", markersize=4)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlabel(x_name)
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.set_ylabel("Validation accuracy, SD v1.4 (%)")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False, fontsize=9)
    _save(fig, "m2_training.png")


def score_distributions(labels):
    """P(AI) histograms: real images, SD v1.4 fakes (seen in training) and fakes from unseen generators."""
    fig, axes = plt.subplots(1, len(labels), figsize=(5.5 * len(labels), 4), sharey=True)
    bins = np.linspace(0, 1, 41)
    for ax, label in zip(np.atleast_1d(axes), labels):
        preds = collect.predictions(label, "resize")
        groups = {"Real images": [], "SD v1.4 fakes (training generator)": [], "Fakes from unseen generators": []}
        for generator, (_, y, probs) in preds.items():
            p = probs.mean(0)
            groups["Real images"].append(p[y == 0])
            key = "SD v1.4 fakes (training generator)" if generator == TRAIN_GENERATOR else "Fakes from unseen generators"
            groups[key].append(p[y == 1])
        # Unseen fakes wear the panel's model colour; real images and SD fakes are neutral greys.
        for (name, values), color in zip(groups.items(), [REAL_GREY, AI_GREY, COLORS[label]]):
            ax.hist(np.concatenate(values), bins=bins, density=True, histtype="step", linewidth=2, color=color, label=name)
        ax.axvline(0.5, color=TEXT, linewidth=1, linestyle="--")
        ax.text(0.5, 1.0, " cutoff 0.5", transform=ax.get_xaxis_transform(), fontsize=9, color=MUTED, va="top")
        ax.set_title(label, loc="left", fontsize=12)
        ax.set_xlabel("P(AI), mean of 3 seeds")
        ax.set_yscale("log")
    np.atleast_1d(axes)[0].set_ylabel("Density (log scale)")
    handles, names = np.atleast_1d(axes)[0].get_legend_handles_labels()
    names[2] = "Fakes from unseen generators (model colour)"
    fig.suptitle("P(AI) of real images, training-generator fakes and unseen fakes", x=0.01, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.legend(handles, names, frameon=False, fontsize=9, loc="upper left", ncols=3, bbox_to_anchor=(0.01, 0.93))
    fig.savefig(FIGURES / "m2_scores.png", dpi=200)
    plt.close(fig)


def gate_use():
    """How much the gated models use the noise branch as test images get more damaged."""
    frames = [pd.read_csv(p) for name in ["u3_gate", "full_gate"] for p in sorted(RESULTS.glob(f"{name}_seed[0-9].csv"))]
    if not frames:
        return
    gates = pd.concat(frames).groupby(["model", "protocol"], sort=False)["mean_gate"].mean().unstack("model")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(gates))
    for name, model, label in [("u3_gate", "B+U3", "B+U3 (gate)"), ("full_gate", "Full (U1+U2+U3)", "Full (gate)")]:
        if name in gates:
            ax.plot(x, gates[name], color=COLORS[model], linewidth=2, marker="o", markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=2, label=label)
    ax.set_xticks(x, [p.replace("resize", "clean") for p in gates.index])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean gate value (0 = noise ignored)")
    ax.set_title("How much the learned gate trusts the noise branch", loc="left", fontsize=13)
    ax.legend(frameon=False)
    _save(fig, "m2_gate.png")


def significance_forest():
    """Each model minus B with 95% bootstrap CIs, clean and shared images (needs python -m ror.analysis)."""
    path = RESULTS / "analysis_significance.csv"
    if not path.exists():
        return
    sig = pd.read_csv(path)
    sig = sig[(sig["generators"] == "all 8") & sig["protocol"].isin(["resize", "jpeg30", "shared"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    labels = list(dict.fromkeys(sig["model"]))
    offsets = {"resize": -0.22, "jpeg30": 0, "shared": 0.22}
    # Conditions get greys and marker shapes: the hues already mean models in every other figure.
    shades = {"resize": AI_GREY, "jpeg30": MUTED, "shared": REAL_GREY}
    markers = {"resize": "o", "jpeg30": "s", "shared": "^"}
    for ax, metric in zip(axes, ["acc", "auc"]):
        for protocol, offset in offsets.items():
            rows = sig[sig["protocol"] == protocol].set_index("model").loc[labels]
            y = np.arange(len(labels)) + offset
            err = [rows[f"{metric}_diff"] - rows[f"{metric}_diff_lo"], rows[f"{metric}_diff_hi"] - rows[f"{metric}_diff"]]
            ax.errorbar(rows[f"{metric}_diff"], y, xerr=err, fmt=markers[protocol], color=shades[protocol], markersize=6,
                        capsize=3, label={"resize": "clean", "jpeg30": "JPEG 30", "shared": "shared"}[protocol])
        ax.axvline(0, color=AXIS, linewidth=1)
        ax.set_yticks(np.arange(len(labels)), labels)
        ax.invert_yaxis()
        ax.set_xlabel(f"{'Accuracy' if metric == 'acc' else 'ROC-AUC'} minus B (percentage points)")
        ax.grid(True, axis="x")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Difference from B, 95% bootstrap intervals (an interval clear of 0 = a real difference)",
                 x=0.01, ha="left", fontsize=13)
    _save(fig, "m2_significance.png")


def m2_figures():
    results = collect.results()
    labels = list(dict.fromkeys(results["model"]))
    colors = COLORS
    pick = lambda label, protocol=None: results[(results["model"] == label)  # noqa: E731
                                                & ((results["protocol"] == protocol) if protocol else True)]

    summary_bars(results, colors)
    real_fake(results, labels)
    for metric in ["acc", "auc"]:
        robustness_curve({m: (colors[m], pick(m)) for m in labels},
                         "Every model as test images are compressed and shared", FIGURES / f"m2_robustness_{metric}.png",
                         metric=metric)
    main_models = [m for m in labels if m in ("B", "B+U1", "B+U3", "Full (U1+U2+U3)")]
    accuracy_columns({m: (colors[m], pick(m, "resize")) for m in main_models},
                     "ROC-AUC on each generator, clean test images", FIGURES / "m2_auc_per_generator.png", metric="auc")
    training_curves()
    score_distributions([m for m in ["B", "B+U1"] if m in labels])
    gate_use()
    significance_forest()
    print("models:", ", ".join(f"{m} = {collect.models()[m]}" for m in labels))
