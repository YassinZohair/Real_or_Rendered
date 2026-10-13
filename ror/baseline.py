"""Baseline results on GenImage: the released ProGAN-trained layer, and our layer retrained on SD v1.4."""
import numpy as np
import pandas as pd
import torch

from ror.config import (
    CHECKPOINTS,
    GENERATORS,
    RELEASED_FC,
    RESULTS,
    SAMPLE_SEED,
    SEEDS,
    TRAIN_GENERATOR,
    VAL_FRACTION,
)
from ror.features import TEST_VIEWS, feature_path
from ror.head import new_head, predict, train_head
from ror.metrics import score


def load_train_val():
    """Split by image, so all augmented views of a validation image stay out of training."""
    cached = np.load(feature_path("train", TRAIN_GENERATOR))
    order = np.random.default_rng(SAMPLE_SEED).permutation(len(cached["label"]))
    n_val = int(len(order) * VAL_FRACTION)
    val_idx, train_idx = order[:n_val], order[n_val:]

    aug = cached["aug"][train_idx]
    views_per_image = aug.shape[1]
    train = (aug.reshape(-1, aug.shape[-1]), np.repeat(cached["label"][train_idx], views_per_image))
    val = (cached["resize"][val_idx], cached["label"][val_idx])
    return train, val


def released_head():
    head = new_head()
    head.load_state_dict(torch.load(RELEASED_FC, map_location="cpu"))
    return head.cuda()


def evaluate(head, model_name, seed):
    """One row per (test generator, protocol), plus the per-image probabilities."""
    rows, probs = [], {}
    for generator in GENERATORS:
        cached = np.load(feature_path("test", generator))
        for protocol in TEST_VIEWS:
            p = predict(head, cached[protocol])
            probs[f"{generator}/{protocol}"] = p
            rows.append(
                {"model": model_name, "seed": seed, "protocol": protocol, "generator": generator}
                | score(cached["label"], p)
            )
    return rows, probs


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    train, val = load_train_val()
    print(f"train: {len(train[1])} feature vectors, val: {len(val[1])} images")

    heads = {("released_progan", 0): released_head()}
    for seed in SEEDS:
        heads[("retrained_sd14", seed)] = train_head(train, val, seed)

    rows = []
    for (model_name, seed), head in heads.items():
        head_rows, probs = evaluate(head, model_name, seed)
        rows += head_rows
        np.savez(RESULTS / f"probs_{model_name}_seed{seed}.npz", **probs)
        torch.save(head.state_dict(), CHECKPOINTS / f"{model_name}_seed{seed}.pth")

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS / "baseline.csv", index=False)

    metrics = ["acc", "real_acc", "fake_acc", "ap", "auc"]
    by_generator = results.groupby(["protocol", "model", "generator"], sort=False)[metrics].mean()
    mean_over_generators = results.groupby(["protocol", "model", "seed"])["acc"].mean()
    summary = mean_over_generators.groupby(["protocol", "model"]).agg(["mean", "std"])
    pd.set_option("display.width", 200)
    print(by_generator.round(1).to_string())
    print("\nMean accuracy over the 8 generators (mean and std across seeds):")
    print(summary.round(2).to_string())


if __name__ == "__main__":
    main()
