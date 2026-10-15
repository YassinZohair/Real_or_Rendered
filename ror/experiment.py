"""Steps every experiment shares: load cached features, train heads, evaluate them, save and summarize."""
import numpy as np
import pandas as pd
import torch

from ror.config import CHECKPOINTS, GENERATORS, RESULTS, SAMPLE_SEED, TRAIN_GENERATOR, VAL_FRACTION
from ror.features import feature_path
from ror.head import predict, train_head
from ror.metrics import score


def load_train_val(tag, train_view, val_view):
    """Split by image, so every copy of a validation image stays out of training."""
    cached = np.load(feature_path("train", TRAIN_GENERATOR, tag))
    order = np.random.default_rng(SAMPLE_SEED).permutation(len(cached["label"]))
    n_val = int(len(order) * VAL_FRACTION)
    val_idx, train_idx = order[:n_val], order[n_val:]

    copies = cached[train_view][train_idx]
    train = (copies.reshape(-1, copies.shape[-1]), np.repeat(cached["label"][train_idx], copies.shape[1]))
    val = (cached[val_view][val_idx], cached["label"][val_idx])
    print(f"train: {len(train[1])} feature vectors, val: {len(val[1])} images")
    return train, val


def train_and_save(train, val, model_name, seed):
    head = train_head(train, val, seed)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), CHECKPOINTS / f"{model_name}_seed{seed}.pth")
    return head


def evaluate(head, test_sets):
    """test_sets maps a feature-file tag to the views to test. One row per (generator, view)."""
    rows, probs = [], {}
    for generator in GENERATORS:
        for tag, views in test_sets.items():
            cached = np.load(feature_path("test", generator, tag))
            labels = probs.setdefault(f"{generator}/label", cached["label"])
            assert (labels == cached["label"]).all(), "test feature files hold different images"
            for view in views:
                p = predict(head, cached[view])
                probs[f"{generator}/{view}"] = p
                rows.append({"generator": generator, "protocol": view} | score(labels, p))
    return rows, probs


def run(name, heads, test_sets):
    """Evaluates every head; writes results/<name>.csv and per-image probabilities; prints a summary."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for (model_name, seed), head in heads.items():
        head_rows, probs = evaluate(head, test_sets)
        rows += [{"model": model_name, "seed": seed} | row for row in head_rows]
        np.savez(RESULTS / f"probs_{name}_{model_name}_seed{seed}.npz", **probs)
    results = pd.DataFrame(rows)
    results.to_csv(RESULTS / f"{name}.csv", index=False)
    summarize(results)


def summarize(results):
    metrics = ["acc", "real_acc", "fake_acc", "ap", "auc"]
    by_generator = results.groupby(["protocol", "model", "generator"], sort=False)[metrics].mean()
    over_generators = results.groupby(["protocol", "model", "seed"], sort=False)[metrics].mean()
    summary = over_generators.groupby(["protocol", "model"], sort=False).agg(
        {"acc": ["mean", "std"], "real_acc": "mean", "fake_acc": "mean", "auc": "mean"}
    )
    pd.set_option("display.width", 200)
    print(by_generator.round(1).to_string())
    print("\nMean over the 8 generators (acc mean and std across seeds):")
    print(summary.round(2).to_string())
