"""Every model's results under one set of names, for the report, the statistics and the figures.

Results were written by different scripts with different file layouts; this module is the one place
that knows them. Display names follow the proposal: B, then the updates added to it.
"""
import numpy as np
import pandas as pd

from ror.config import FEATURES, GENERATORS, RESULTS, SEEDS, TRAIN_GENERATOR, TRAIN_TAG

U1_RANK = 8
UNSEEN = [g for g in GENERATORS if g != TRAIN_GENERATOR]


def _u3_name(variant):
    """The fusion chosen by validation accuracy (never by test results), e.g. "u3_gate"."""
    scores = {}
    for fusion in ["concat", "gate"]:
        logs = [RESULTS / f"trainlog_{variant}_{fusion}_seed{s}.csv" for s in SEEDS]
        if all(log.exists() for log in logs):
            scores[fusion] = np.mean([pd.read_csv(log)["val_acc"].iloc[1:].max() for log in logs])
    return f"{variant}_{max(scores, key=scores.get)}" if scores else None


def models(both_fusions=False):
    """Display name -> results model name, for the models that have results."""
    names = {"B": "retrained_sd14", "B+U2": "u2v2", "B+U1": f"u1_r{U1_RANK}", "B+U1+U2": f"u1u2_r{U1_RANK}"}
    if both_fusions:
        for variant, label in [("u3", "B+U3"), ("full", "Full")]:
            for fusion in ["concat", "gate"]:
                names[f"{label} ({fusion})"] = f"{variant}_{fusion}"
    else:
        names |= {"B+U3": _u3_name("u3"), "Full (U1+U2+U3)": _u3_name("full")}
    return {label: name for label, name in names.items() if name and _files(name)}


def _files(name):
    if name in ("retrained_sd14", "u2v2"):
        return [RESULTS / "u2.csv"] if (RESULTS / "u2.csv").exists() else []
    return sorted(RESULTS.glob(f"{name}_seed[0-9].csv")) + sorted(RESULTS.glob(f"{name}_seed[0-9]_degraded.csv"))


def results(both_fusions=False):
    """Per-generator metrics: model (display name), seed, generator, protocol, acc, real_acc, fake_acc, ap, auc."""
    frames = []
    for label, name in models(both_fusions).items():
        frame = pd.concat([pd.read_csv(path) for path in _files(name)])
        frames.append(frame[frame["model"] == name].assign(model=label))
    return pd.concat(frames, ignore_index=True)


def predictions(label, protocol, both_fusions=False):
    """{generator: (rows, labels, probs [seeds, images])}: every seed's P(AI) for the same test images."""
    name = models(both_fusions)[label]
    out = {}
    for generator in GENERATORS:
        per_seed = [_seed_predictions(name, seed, generator, protocol) for seed in SEEDS]
        rows, labels, _ = per_seed[0]
        for r, _, _ in per_seed[1:]:
            assert (r == rows).all(), f"{label}: seeds were tested on different images"
        out[generator] = (rows, labels, np.stack([p for _, _, p in per_seed]))
    return out


def _seed_predictions(name, seed, generator, protocol):
    clean = protocol == "resize"
    if name in ("retrained_sd14", "u2v2"):  # ror.u2: all 2,000 + 2,000 images, in the frozen feature files' order
        probs = np.load(RESULTS / f"probs_u2_{name}_seed{seed}.npz")
        rows = np.load(FEATURES / f"test_{generator}.npz")["row"]
        return rows, probs[f"{generator}/label"], probs[f"{generator}/{protocol}"]
    if name.startswith("u1"):  # ror.u1: rows are in the run's own feature files
        run = f"{name}_seed{seed}" + ("" if clean else "_degraded")
        probs = np.load(RESULTS / f"probs_{run}_{name}_seed{seed}.npz")
        rows = np.load(FEATURES / f"test_{generator}{TRAIN_TAG}_{name}_seed{seed}{'' if clean else '_degraded'}.npz")["row"]
        return rows, probs[f"{generator}/label"], probs[f"{generator}/{protocol}"]
    probs = np.load(RESULTS / f"probs_{name}_seed{seed}.npz")  # ror.u3 stores rows itself
    return probs[f"{generator}/{protocol}_row"], probs[f"{generator}/{protocol}_label"], probs[f"{generator}/{protocol}"]


def training_logs(name):
    """{seed: training log} for a U1 or U3 run, e.g. name="u1_r8"."""
    return {s: pd.read_csv(RESULTS / f"trainlog_{name}_seed{s}.csv") for s in SEEDS
            if (RESULTS / f"trainlog_{name}_seed{s}.csv").exists()}
