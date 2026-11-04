"""U1: low-rank adapters (LoRA, Hu et al. 2022) on the query and value projections of CLIP's vision encoder.

CLIP's attention uses PyTorch's nn.MultiheadAttention, which stores q, k and v as one fused
[3 * width, width] matrix. The adapter is added to that matrix as a parametrization:

    W_qkv  ->  W_qkv + scale * [B_q A_q ; 0 ; B_v A_v]

so the repo's attention code runs unchanged. B starts at zero, so an adapted model is exactly
the frozen model until training moves it. Only A and B (and the linear head) are trained.
"""
import math

import torch
from torch import nn
from torch.nn.utils import parametrize
from torch.utils.checkpoint import checkpoint

# scale = ALPHA_PER_RANK, the same for every rank, so the rank sweep changes capacity only.
ALPHA_PER_RANK = 2.0


class QVAdapter(nn.Module):
    def __init__(self, width, rank):
        super().__init__()
        self.width = width
        self.scale = ALPHA_PER_RANK
        self.a_q = nn.Parameter(torch.empty(rank, width))
        self.a_v = nn.Parameter(torch.empty(rank, width))
        self.b_q = nn.Parameter(torch.zeros(width, rank))
        self.b_v = nn.Parameter(torch.zeros(width, rank))
        for a in (self.a_q, self.a_v):  # PEFT's default initialisation for A
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))

    def forward(self, w_qkv):
        """Called by the parametrization with the frozen fused weight; returns the adapted one."""
        zeros = torch.zeros(self.width, self.width, dtype=w_qkv.dtype, device=w_qkv.device)
        delta = torch.cat([self.b_q @ self.a_q, zeros, self.b_v @ self.a_v])
        return w_qkv + self.scale * delta


def vision_blocks(univfd):
    return list(univfd.model.visual.transformer.resblocks.children())


def add_lora(univfd, rank):
    """Freezes every CLIP weight and adds an adapter to each of the 24 vision attention layers."""
    for p in univfd.model.parameters():
        p.requires_grad_(False)
    for block in vision_blocks(univfd):
        width = block.attn.embed_dim
        adapter = QVAdapter(width, rank).to(block.attn.in_proj_weight.device)
        parametrize.register_parametrization(block.attn, "in_proj_weight", adapter)
    return univfd


def adapters(univfd):
    return [block.attn.parametrizations.in_proj_weight[0] for block in vision_blocks(univfd)]


def lora_parameters(univfd):
    return [p for adapter in adapters(univfd) for p in adapter.parameters()]


def lora_state(univfd):
    return [{k: v.detach().cpu() for k, v in adapter.state_dict().items()} for adapter in adapters(univfd)]


def load_lora_state(univfd, state):
    for adapter, weights in zip(adapters(univfd), state, strict=True):
        adapter.load_state_dict(weights)


def drop_text_tower(univfd):
    """UnivFD only uses the image encoder. Freeing the text encoder saves about 0.5 GB of GPU memory."""
    clip = univfd.model
    for name in ("transformer", "token_embedding", "ln_final"):
        delattr(clip, name)
    for name in ("positional_embedding", "text_projection"):
        delattr(clip, name)
    return univfd


def use_gradient_checkpointing(univfd):
    """Recompute each block's activations in the backward pass instead of storing them.

    Training ViT-L/14 at batch 32 needs about 10 to 12 GB of activations without it, too close to a
    T4's 15 GB. With it, memory is small and training is about 30% slower.
    """
    for block in vision_blocks(univfd):
        block.forward = _checkpointed(block.forward)
    return univfd


def _checkpointed(forward):
    def run(x):
        return checkpoint(forward, x, use_reentrant=False) if torch.is_grad_enabled() else forward(x)

    return run


def rank(univfd):
    return adapters(univfd)[0].a_q.shape[0]
