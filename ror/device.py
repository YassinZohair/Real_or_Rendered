"""Where tensors live: the Colab GPU (cuda), an Apple Silicon GPU (mps), or the CPU.

Override with the ROR_DEVICE environment variable, e.g. ROR_DEVICE=cpu.
"""
import contextlib
import os
import sys

import torch


def _pick():
    if os.environ.get("ROR_DEVICE"):
        return torch.device(os.environ["ROR_DEVICE"])
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _pick()

# DataLoader workers. On Linux (Colab) workers are forked and can share our lambda-based views;
# macOS and Windows start workers by pickling, which lambdas do not survive, so load in the main process.
NUM_WORKERS = int(os.environ.get("ROR_WORKERS", os.cpu_count() if sys.platform == "linux" else 0))


def autocast():
    """Half precision on GPUs (about 2x faster on a T4), full precision on the CPU."""
    if DEVICE.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    if DEVICE.type == "mps":
        try:  # MPS autocast only exists in recent PyTorch versions
            return torch.autocast("mps", dtype=torch.float16)
        except (RuntimeError, ValueError):
            pass
    return contextlib.nullcontext()
