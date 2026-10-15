"""The trainable part of UnivFD: one linear layer on cached CLIP features."""
import copy

import torch
from torch import nn

from ror.config import FEATURE_DIM

# The repo's training settings (networks/trainer.py, options/train_options.py, train.py).
LR = 1e-4
MIN_LR = 1e-6
BATCH_SIZE = 256
MAX_EPOCHS = 100
PATIENCE = 5
MIN_IMPROVEMENT = 0.001
INIT_STD = 0.02


def new_head():
    return nn.Linear(FEATURE_DIM, 1)


def load_head(path):
    head = new_head()
    head.load_state_dict(torch.load(path, map_location="cpu"))
    return head.cuda()


@torch.no_grad()
def predict(head, x):
    """Probability that each image is AI-generated."""
    device = head.weight.device
    return torch.sigmoid(head(torch.as_tensor(x, device=device)).squeeze(1)).cpu().numpy()


def accuracy(head, x, y):
    return float(((predict(head, x) > 0.5) == y).mean())


def train_head(train, val, seed):
    """Early stopping on validation accuracy; on a plateau the learning rate drops 10x until MIN_LR."""
    torch.manual_seed(seed)
    x = torch.as_tensor(train[0]).cuda()
    y = torch.as_tensor(train[1]).float().cuda()
    head = new_head().cuda()
    nn.init.normal_(head.weight, 0.0, INIT_STD)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=0.0)
    loss_fn = nn.BCEWithLogitsLoss()

    best_acc, best_state, stale, lr = -1.0, None, 0, LR
    for epoch in range(MAX_EPOCHS):
        for batch in torch.randperm(len(x), device=x.device).split(BATCH_SIZE):
            loss = loss_fn(head(x[batch]).squeeze(1), y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc = accuracy(head, *val)
        if acc > best_acc + MIN_IMPROVEMENT:
            best_acc, best_state, stale = acc, copy.deepcopy(head.state_dict()), 0
        else:
            stale += 1
        if stale >= PATIENCE:
            lr /= 10
            if lr < MIN_LR:
                break
            for group in optimizer.param_groups:
                group["lr"] = lr
            stale = 0

    print(f"seed {seed}: stopped after epoch {epoch + 1}, best val acc {best_acc:.4f}")
    head.load_state_dict(best_state)
    return head
