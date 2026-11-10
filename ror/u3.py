"""U3 (noise branch + fusion) on frozen CLIP features, and the full model (U1 + U2 + U3).

    python -m ror.patches                                  crops for B+U3 and the test sets (once per session)
    python -m ror.u3 train --fusion concat --seed 0        B+U3
    python -m ror.u3 train --fusion gate --seed 0          B+U3 with the learned gate
    python -m ror.patches full                             crops for the full model's training
    python -m ror.u3 train --fusion gate --seed 0 --full   full model: U3 on top of the trained B+U1+U2

B+U3 extends B: CLIP frozen, B's head, B's training augmentation. The full model extends B+U1+U2 of the
same seed: its LoRA-adapted CLIP (now frozen), its head, U2 v2 damage. In both, the noise head starts
at zero, so training starts from exactly the model being extended and U3 is the only change.
The full model is trained in two stages (U1+U2 first, then U3 on its frozen features) so that it
fits on a T4 and every stage can be checked on its own.
"""
import argparse
import math

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils import parametrize
from tqdm import tqdm

from ror import patches
from ror.config import CHECKPOINTS, GENERATORS, RESULTS, TRAIN_GENERATOR, TRAIN_TAG
from ror.device import DEVICE, autocast
from ror.experiment import summarize, train_val_split
from ror.features import CLEAN_TAG, DEGRADED_TAG, Job, cache, feature_path
from ror.metrics import score
from ror.noise import FusionModel
from ror.views import DEGRADED_TEST_VIEWS

EPOCHS = 10
BATCH_SIZE = 128
NOISE_LR = 1e-3  # branch trained from scratch
HEAD_LR = 1e-4  # the CLIP head is already trained
WEIGHT_DECAY = 1e-4
FULL_RANK = 8  # the U1 rank chosen by the sweep


def setup(variant, seed):
    """Where each model's CLIP features, crops and starting head come from."""
    if variant == "u3":
        return {
            "clip_train": (feature_path("train", TRAIN_GENERATOR, CLEAN_TAG), "aug", "resize"),
            "patch_train": ("_u3", "aug", "clean"),
            "clip_test": CLEAN_TAG,
            "head": torch.load(CHECKPOINTS / f"retrained_sd14_seed{seed}.pth", map_location="cpu"),
        }
    run = f"u1u2_r{FULL_RANK}_seed{seed}"
    return {
        "clip_train": (feature_path("train", TRAIN_GENERATOR, f"_{run}"), "u2v2_aug", "jpeg95"),
        "patch_train": ("_full", "u2v2_aug", "jpeg95"),
        "clip_test": f"{TRAIN_TAG}_{run}",
        "head": torch.load(CHECKPOINTS / f"{run}.pt", map_location="cpu")["head"],
    }


def adapted_train_features(seed):
    """The trained B+U1+U2 model's CLIP features for the training images, extracted once."""
    from ror.u1 import load_u1

    run = f"u1u2_r{FULL_RANK}_seed{seed}"
    job = Job("train", [TRAIN_GENERATOR], {"u2v2_aug": 4, "jpeg95": 1}, None, f"_{run}")
    if feature_path("train", TRAIN_GENERATOR, job.tag).exists():
        return
    model = load_u1(CHECKPOINTS / f"{run}.pt")
    with parametrize.cached():
        cache(model, job, TRAIN_GENERATOR)


def _positions(have, want):
    """Index of each wanted image row in an array of cached rows (both sorted)."""
    idx = np.searchsorted(have, want)
    assert (have[np.minimum(idx, len(have) - 1)] == want).all(), "cached sets hold different images"
    return idx


def load_training(cfg):
    clip_path, clip_train_view, clip_val_view = cfg["clip_train"]
    patch_tag, patch_train_view, patch_val_view = cfg["patch_train"]
    clip = np.load(clip_path)
    rows, crops = patches.load("train", TRAIN_GENERATOR, patch_tag, patch_train_view)
    _, val_crops = patches.load("train", TRAIN_GENERATOR, patch_tag, patch_val_view)
    assert (rows == clip["row"]).all(), "CLIP features and crops hold different training images"
    train_idx, val_idx = train_val_split(len(rows))
    labels = torch.as_tensor(clip["label"]).float()
    return {
        "train": (torch.as_tensor(clip[clip_train_view][train_idx]), torch.as_tensor(np.asarray(crops[train_idx])),
                  labels[train_idx]),
        "val": (torch.as_tensor(clip[clip_val_view][val_idx]), torch.as_tensor(np.asarray(val_crops[val_idx])),
                labels[val_idx]),
    }


@torch.no_grad()
def predict(model, clip_features, crops, batch_size=256):
    model.eval()
    probs, gates = [], []
    for start in range(0, len(crops), batch_size):
        c = torch.as_tensor(clip_features[start : start + batch_size]).float().to(DEVICE)
        x = torch.from_numpy(np.array(crops[start : start + batch_size])).to(DEVICE)
        with autocast():
            logits, g = model(c, x)
        probs.append(torch.sigmoid(logits.float()).cpu())
        gates.append(g.float().cpu())
    return torch.cat(probs).numpy(), torch.cat(gates).numpy()


def val_accuracy(model, val):
    probs, _ = predict(model, val[0], val[1])
    return float(((probs > 0.5) == (val[2].numpy() > 0.5)).mean())


def train(model, sets, seed, name):
    """Each step pairs a random one of the 4 augmented CLIP copies with a random one of the 4 crops."""
    torch.manual_seed(seed)
    clip, crops, labels = sets["train"]
    steps_per_epoch = len(labels) // BATCH_SIZE
    total = EPOCHS * steps_per_epoch
    optimizer = torch.optim.AdamW(model.param_groups(NOISE_LR, HEAD_LR, WEIGHT_DECAY))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 0.5 * (1 + math.cos(math.pi * s / total)))
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=DEVICE.type == "cuda")
    loss_fn = nn.BCEWithLogitsLoss()

    log = [{"epoch": 0, "train_loss": float("nan"), "val_acc": val_accuracy(model, sets["val"])}]
    print(f"epoch 0 (= starting model): val acc {log[0]['val_acc']:.4f}")
    best = None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for batch in tqdm(torch.randperm(len(labels))[: steps_per_epoch * BATCH_SIZE].split(BATCH_SIZE),
                          desc=f"epoch {epoch}", leave=False):
            pick_clip = torch.randint(clip.shape[1], (len(batch),))
            pick_crop = torch.randint(crops.shape[1], (len(batch),))
            c = clip[batch, pick_clip].to(DEVICE)
            x = crops[batch, pick_crop].to(DEVICE)
            with autocast():
                logits, _ = model(c, x)
            loss = loss_fn(logits.float(), labels[batch].to(DEVICE))
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(loss.item())
        acc = val_accuracy(model, sets["val"])
        log.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_acc": acc})
        print(f"epoch {epoch}: train loss {log[-1]['train_loss']:.4f}, val acc {acc:.4f}")
        if best is None or acc > best["val_acc"]:
            best = {"val_acc": acc, "epoch": epoch, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}

    model.load_state_dict(best["state"])
    print(f"kept epoch {best['epoch']} (val acc {best['val_acc']:.4f})")
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    torch.save({"fusion": model.fusion} | best, CHECKPOINTS / f"{name}_seed{seed}.pt")
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log).to_csv(RESULTS / f"trainlog_{name}_seed{seed}.csv", index=False)
    return model


def evaluate(model, cfg, name, seed):
    """Clean test images (all 2,000 + 2,000) and damaged copies (the 1,000 + 1,000 subset)."""
    rows, probs = [], {}
    sets = [(CLEAN_TAG, ["clean"], "resize", cfg["clip_test"]),
            (DEGRADED_TAG, DEGRADED_TEST_VIEWS, None, cfg["clip_test"] + DEGRADED_TAG)]
    for generator in GENERATORS:
        for patch_tag, views, clip_view, clip_tag in sets:
            clip = np.load(feature_path("test", generator, clip_tag))
            for view in views:
                crop_rows, crops = patches.load("test", generator, patch_tag, view)
                take = _positions(clip["row"], crop_rows)
                protocol = clip_view or view
                p, g = predict(model, clip[protocol][take], crops)
                labels = clip["label"][take]
                rows.append({"model": name, "seed": seed, "generator": generator, "protocol": protocol}
                            | score(labels, p) | {"mean_gate": float(g.mean())})
                probs |= {f"{generator}/{protocol}": p, f"{generator}/{protocol}_label": labels,
                          f"{generator}/{protocol}_row": crop_rows, f"{generator}/{protocol}_gate": g}
    results = pd.DataFrame(rows)
    results.to_csv(RESULTS / f"{name}_seed{seed}.csv", index=False)
    np.savez(RESULTS / f"probs_{name}_seed{seed}.npz", **probs)
    summarize(results)
    if model.fusion == "gate":
        print("\nmean gate (how much the noise branch is used), by test condition:")
        print(results.groupby("protocol", sort=False)["mean_gate"].mean().round(3).to_string())


def report():
    """Every model so far, mean over the 8 generators, clean and damaged.

    B and U2 were scored on all 2,000 + 2,000 damaged images, the others on the 1,000 + 1,000 subset,
    so damaged rows compare closely but not image for image (ror.analysis does the paired version).
    """
    keep = {"baseline.csv": ["retrained_sd14"], "u2.csv": ["u2v2"]}
    frames = []
    for file, models in keep.items():
        if (RESULTS / file).exists():
            frame = pd.read_csv(RESULTS / file)
            frames.append(frame[frame["model"].isin(models)])
    for pattern in ["u1_r8_seed*.csv", "u1_r8_seed*_degraded.csv", "u1u2_r8_seed*.csv", "u3_*_seed*.csv", "full_*_seed*.csv"]:
        frames += [pd.read_csv(path) for path in sorted(RESULTS.glob(pattern))]
    results = pd.concat(frames)
    metrics = ["acc", "real_acc", "fake_acc", "auc"]
    per_seed = results.groupby(["protocol", "model", "seed"], sort=False)[metrics].mean()
    table = per_seed.groupby(["protocol", "model"], sort=False).agg(
        {"acc": ["mean", "std"], "real_acc": "mean", "fake_acc": "mean", "auc": ["mean", "std"]}
    )
    pd.set_option("display.width", 200)
    print(table.round(2).to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["train", "report"])
    parser.add_argument("--fusion", choices=["concat", "gate"], default="gate")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--full", action="store_true", help="U3 on top of the trained B+U1+U2 (the full model)")
    args = parser.parse_args()
    if args.command == "report":
        return report()

    variant = "full" if args.full else "u3"
    name = f"{variant}_{args.fusion}"
    if args.full:
        adapted_train_features(args.seed)
    cfg = setup(variant, args.seed)
    model = FusionModel(args.fusion, cfg["head"]).to(DEVICE)
    checkpoint = CHECKPOINTS / f"{name}_seed{args.seed}.pt"
    if checkpoint.exists():  # trained already: evaluate only
        model.load_state_dict(torch.load(checkpoint, map_location="cpu")["state"])
    else:
        model = train(model, load_training(cfg), args.seed, name)
    evaluate(model, cfg, name, args.seed)


if __name__ == "__main__":
    main()
