"""U1: LoRA on CLIP's attention, trained end to end with the linear head on SD v1.4 images.

    python -m ror.u1 check                               zero-init faithfulness and gradient check
    python -m ror.u1 train --rank 8 --seed 0             U1 (the repo's blur+JPEG augmentation)
    python -m ror.u1 train --rank 8 --seed 0 --u2        B+U1+U2 (U2 v2 damage, equal format)
    python -m ror.u1 train ... --damaged                 also test on JPEG / shared copies
    python -m ror.u1 report                              mean over generators, U1 runs vs baseline

The head starts from the trained linear head of the same seed and data (B's head for U1, U2 v2's head
for B+U1+U2) and the adapters start at zero, so training starts from exactly that frozen-CLIP model and
U1 is the only change (linear probe, then fine-tune: Kumar et al. 2022,
which keeps fine-tuning from distorting the pretrained features as much as a random head would).

Unlike B and U2, gradients flow through all 24 CLIP layers, so features cannot be cached for
training. After training, the adapted model's test features are cached under the run's own tag,
and the existing evaluation code scores them.
"""
import argparse
import math
import os
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils import parametrize
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ror import data, lora
from ror.config import CHECKPOINTS, GENERATORS, RESULTS, TEST_PER_CLASS, TRAIN_GENERATOR, TRAIN_TAG
from ror.device import DEVICE, NUM_WORKERS, autocast
from ror.experiment import run, summarize, train_val_split
from ror.features import DEGRADED_TAG, Job, cache, feature_path, load_univfd
from ror.head import predict
from ror.views import DEGRADED_TEST_VIEWS, VIEWS

BATCH_SIZE = 32
LORA_LR = 2e-4
HEAD_LR = 1e-4  # the repo's learning rate for the head
WARMUP_FRACTION = 0.05
EPOCHS = 1
VALIDATIONS_PER_EPOCH = 4
# Damaged copies are scored on 1,000 + 1,000 images per generator (a subset of the clean 2,000 + 2,000),
# because each adapted model has to run CLIP again on every image.
DAMAGED_PER_CLASS = 1000
GRADIENT_CHECKPOINTING = os.environ.get("ROR_CHECKPOINTING", "1") == "1"

# variant -> (training view, validation view, starting head). Views match how B and U2 were trained and
# validated; the starting head is the frozen-CLIP model trained on the same images (B, or B+U2 v2).
VARIANTS = {"u1": ("aug", "resize", "retrained_sd14"), "u1u2": ("u2v2_aug", "jpeg95", "u2v2")}
BASELINE_HEAD = "retrained_sd14"


class Labelled(Dataset):
    """One view of each image, with its label."""

    def __init__(self, rows, labels, view):
        self.images = data.ViewDataset("train", TRAIN_GENERATOR, rows, [VIEWS[view]])
        self.labels = torch.as_tensor(labels[rows]).float()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.images[i][0], self.labels[i]


def build(rank, head_state):
    """The UnivFD model with zero-initialised adapters and the given head."""
    model = lora.add_lora(lora.drop_text_tower(load_univfd()), rank)
    model.fc.load_state_dict(head_state)
    if GRADIENT_CHECKPOINTING:
        lora.use_gradient_checkpointing(model)
    return model


def checkpoint_path(name, seed):
    return CHECKPOINTS / f"{name}_seed{seed}.pt"


def starting_head(variant, seed):
    return torch.load(CHECKPOINTS / f"{VARIANTS[variant][2]}_seed{seed}.pth", map_location="cpu")


def load_u1(path):
    """A trained U1 model from its checkpoint, ready for inference."""
    saved = torch.load(path, map_location="cpu")
    model = build(saved["rank"], saved["head"])
    lora.load_lora_state(model, saved["lora"])
    return model.eval()


def snapshot(model):
    return {"lora": lora.lora_state(model), "head": {k: v.detach().cpu() for k, v in model.fc.state_dict().items()}}


def restore(model, state):
    lora.load_lora_state(model, state["lora"])
    model.fc.load_state_dict(state["head"])


@torch.no_grad()
def val_accuracy(model, loader):
    correct = 0
    with parametrize.cached():
        for x, y in loader:
            with autocast():
                logits = model(x.to(DEVICE)).squeeze(1)
            correct += ((logits.float().cpu() > 0) == (y > 0.5)).sum().item()
    return correct / len(loader.dataset)


def warmup_cosine(total_steps):
    warmup = max(1, int(total_steps * WARMUP_FRACTION))

    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))

    return factor


def train(model, variant, name, seed):
    """One epoch of LoRA + head training. Resumes from the last validation if a session dropped."""
    torch.manual_seed(seed)
    random.seed(seed)
    train_view, val_view, _ = VARIANTS[variant]
    labels, _ = data.read_index("train", TRAIN_GENERATOR)
    train_rows, val_rows = train_val_split(len(labels))
    train_set = Labelled(train_rows, labels, train_view)
    val_loader = DataLoader(Labelled(val_rows, labels, val_view), batch_size=64, num_workers=NUM_WORKERS)

    steps_per_epoch = len(train_set) // BATCH_SIZE
    total_steps = steps_per_epoch * EPOCHS
    val_every = max(1, steps_per_epoch // VALIDATIONS_PER_EPOCH)
    optimizer = torch.optim.AdamW(
        [{"params": lora.lora_parameters(model), "lr": LORA_LR}, {"params": model.fc.parameters(), "lr": HEAD_LR}],
        weight_decay=0.0,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine(total_steps))
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=DEVICE.type == "cuda")
    loss_fn = nn.BCEWithLogitsLoss()

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    partial = CHECKPOINTS / f"{name}_seed{seed}.partial.pt"
    if partial.exists():
        state = torch.load(partial, map_location="cpu", weights_only=False)
        restore(model, state["current"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        step, best, log = state["step"], state["best"], state["log"]
        print(f"resuming at step {step} of {total_steps}")
    else:
        step, best = 0, None
        # Must equal the starting head's validation accuracy: the adapters are still zero.
        log = [{"step": 0, "train_loss": float("nan"), "val_acc": val_accuracy(model, val_loader)}]
        print(f"step 0 (= starting model): val acc {log[0]['val_acc']:.4f}")

    print(f"{name} seed {seed}: {len(train_set)} images, {total_steps} steps of {BATCH_SIZE}, validating every {val_every}")
    losses = []
    while step < total_steps:
        # The order of each epoch is fixed by the seed, so a resumed run continues the same order.
        epoch, done = divmod(step, steps_per_epoch)
        order = torch.randperm(len(train_set), generator=torch.Generator().manual_seed(seed * 1000 + epoch))
        order = order[done * BATCH_SIZE : steps_per_epoch * BATCH_SIZE].tolist()
        loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=order, num_workers=NUM_WORKERS, drop_last=True)
        for x, y in tqdm(loader, initial=done, total=steps_per_epoch, desc=f"epoch {epoch + 1}"):
            with autocast():
                logits = model(x.to(DEVICE)).squeeze(1)
            loss = loss_fn(logits.float(), y.to(DEVICE))
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(loss.item())
            step += 1

            if step % val_every == 0 or step == total_steps:
                acc = val_accuracy(model, val_loader)
                log.append({"step": step, "train_loss": float(np.mean(losses)), "val_acc": acc})
                losses = []
                print(f"step {step}: train loss {log[-1]['train_loss']:.4f}, val acc {acc:.4f}")
                if best is None or acc > best["val_acc"]:
                    best = {"val_acc": acc, "step": step} | snapshot(model)
                torch.save(
                    {
                        "current": snapshot(model),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                        "step": step,
                        "best": best,
                        "log": log,
                    },
                    partial,
                )

    restore(model, best)
    print(f"kept step {best['step']} of {total_steps} (val acc {best['val_acc']:.4f})")
    torch.save({"rank": lora.rank(model), "variant": variant, "seed": seed} | best, checkpoint_path(name, seed))
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log).to_csv(RESULTS / f"trainlog_{name}_seed{seed}.csv", index=False)
    partial.unlink()
    return model.eval()


def test_jobs(name, seed, damaged):
    """results name -> the adapted model's own test features, tagged with the training size (they depend on it).

    One results file per job, because the damaged subset holds different images from the clean set.
    """
    tag = f"{TRAIN_TAG}_{name}_seed{seed}"
    jobs = {f"{name}_seed{seed}": Job("test", GENERATORS, {"resize": 1}, TEST_PER_CLASS, tag)}
    if damaged:
        views = {view: 1 for view in DEGRADED_TEST_VIEWS}
        jobs[f"{name}_seed{seed}{DEGRADED_TAG}"] = Job("test", GENERATORS, views, DAMAGED_PER_CLASS, tag + DEGRADED_TAG)
    return jobs


def evaluate(model, name, seed, damaged):
    with parametrize.cached():
        for results_name, job in test_jobs(name, seed, damaged).items():
            for generator in job.generators:
                cache(model, job, generator)
            run(results_name, {(name, seed): model.fc}, {job.tag: list(job.views)})


@torch.no_grad()
def check(rank=8, n_images=64):
    """The adapted model at initialisation must reproduce the cached baseline, and train only the adapters."""
    cached = np.load(feature_path("test", TRAIN_GENERATOR))
    pick = np.random.default_rng(0).choice(len(cached["row"]), n_images, replace=False)
    dataset = data.ViewDataset("test", TRAIN_GENERATOR, cached["row"][pick], views=[VIEWS["resize"]])
    images = torch.stack([dataset[i][0] for i in range(n_images)]).to(DEVICE)

    model = build(rank, starting_head("u1", 0)).eval()
    with autocast():
        feats = torch.cat([model.model.encode_image(b) for b in images.split(16)]).float()
    diff = np.abs(feats.cpu().numpy() - cached["resize"][pick]).max()
    p_adapted, p_cached = predict(model.fc, feats), predict(model.fc, cached["resize"][pick])
    print(f"zero-init: max |feature difference| {diff:.5f}, max |probability difference| "
          f"{np.abs(p_adapted - p_cached).max():.5f}, same decision {((p_adapted > 0.5) == (p_cached > 0.5)).mean():.1%}")

    n_lora = sum(p.numel() for p in lora.lora_parameters(model))
    n_vision = sum(p.numel() for p in model.model.visual.parameters())
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"rank {rank}: {n_lora:,} adapter parameters = {n_lora / n_vision:.2%} of CLIP's vision encoder "
          f"({n_vision:,}); trainable tensors: {len(trainable)} (adapters + head)")

    with torch.enable_grad():
        with autocast():
            loss = model(images[:8]).float().mean()
        loss.backward()
    frozen_with_grad = [n for n, p in model.named_parameters() if not p.requires_grad and p.grad is not None]
    adapter_grads = [a.b_q.grad.abs().sum().item() > 0 for a in lora.adapters(model)]
    print(f"gradients: B matrices with a gradient in {sum(adapter_grads)}/{len(adapter_grads)} layers, "
          f"frozen tensors with a gradient: {len(frozen_with_grad)}")


def report():
    """Mean over the 8 generators on clean test images: every U1 run next to the baseline head."""
    frames = [pd.read_csv(path) for variant in VARIANTS for path in sorted(RESULTS.glob(f"{variant}_r*_seed*.csv"))]
    baseline = RESULTS / "baseline.csv"
    if baseline.exists():
        frames.append(pd.read_csv(baseline))
    results = pd.concat(frames)
    is_u1 = results["model"].str.startswith(tuple(VARIANTS))
    summarize(results[(results["protocol"] == "resize") & (is_u1 | (results["model"] == BASELINE_HEAD))])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["check", "train", "report"])
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--u2", action="store_true", help="train on U2 v2 damage (the B+U1+U2 model)")
    parser.add_argument("--damaged", action="store_true", help="also test on the damaged copies")
    args = parser.parse_args()

    if args.command == "check":
        check(args.rank)
    elif args.command == "report":
        report()
    else:
        variant = "u1u2" if args.u2 else "u1"
        name = f"{variant}_r{args.rank}"
        if checkpoint_path(name, args.seed).exists():  # trained already: only the missing evaluations run
            model = load_u1(checkpoint_path(name, args.seed))
        else:
            model = train(build(args.rank, starting_head(variant, args.seed)), variant, name, args.seed)
        evaluate(model, name, args.seed, args.damaged)


if __name__ == "__main__":
    main()
