"""U3's networks: a noise branch that reads high-frequency residuals, fused with CLIP's 768 features.

The branch never sees image content directly. Two filter banks remove content and keep noise:
- SRM: three fixed high-pass filters from steganalysis (Fridrich & Kodovsky 2012), applied to each
  colour channel, with the usual truncation of large residuals (strong edges) to +-3.
- A constrained convolution (after Bayar & Stamm 2018): learned 5x5 filters forced to be
  prediction-error filters (centre -1, other taps sum to 1), so they can only learn high-pass filters.
A small ResNet-style CNN trained from scratch then turns the residuals into NOISE_DIM numbers.
It never downsamples before the first residual block, because pooling early averages the noise away.
"""
import torch
from torch import nn
from torch.nn import functional as F

from ror.config import FEATURE_DIM

NOISE_DIM = 128
SRM_TRUNCATION = 3.0

_SRM = [
    [[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]],  # 2nd order, /4
    [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]],  # /12
    [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],  # 1st order, /2
]
_SRM_SCALE = [4.0, 12.0, 2.0]


def _mirror_pad(x, size):
    """Zero padding would put a sharp edge around every crop; mirroring keeps the border noise-like."""
    return F.pad(x, (size // 2,) * 4, mode="reflect")


class SRM(nn.Module):
    """Fixed: 3 filters x 3 colour channels = 9 residual maps."""

    def __init__(self):
        super().__init__()
        kernels = torch.stack([torch.tensor(k, dtype=torch.float32) / s for k, s in zip(_SRM, _SRM_SCALE)])
        self.register_buffer("weight", kernels.unsqueeze(1).repeat(3, 1, 1, 1))  # [9, 1, 5, 5], 3 per channel

    def forward(self, x):
        residuals = F.conv2d(_mirror_pad(x, 5), self.weight.to(x.dtype), groups=3)
        return residuals.clamp(-SRM_TRUNCATION, SRM_TRUNCATION)


class ConstrainedConv(nn.Module):
    """Learned high-pass filters: each 5x5 kernel has centre -1 and its other taps sum to 1.

    The other taps are a softmax over learned scores, so each filter is "centre pixel minus a learned
    weighted average of its neighbours". This keeps Bayar & Stamm's constraint without their
    renormalisation step, which divides by a sum that can drift towards zero.
    """

    def __init__(self, channels=3, filters=3, size=5):
        super().__init__()
        self.scores = nn.Parameter(torch.randn(filters, channels, size * size) * 0.1)
        self.size = size
        centre = torch.zeros(size * size, dtype=torch.bool)
        centre[size * size // 2] = True
        self.register_buffer("centre", centre)

    def kernel(self):
        weights = torch.softmax(self.scores.masked_fill(self.centre, float("-inf")), dim=-1)
        weights = weights - self.centre.float()
        return weights.view(*self.scores.shape[:2], self.size, self.size)

    def forward(self, x):
        return F.conv2d(_mirror_pad(x, self.size), self.kernel().to(x.dtype))


class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.skip = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(y)) + self.skip(x))


class NoiseCNN(nn.Module):
    """uint8 crops [n, 3, h, w] -> [n, NOISE_DIM]."""

    def __init__(self):
        super().__init__()
        self.srm = SRM()
        self.constrained = ConstrainedConv()
        self.stem = nn.Sequential(nn.Conv2d(9 + 3, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU())
        self.blocks = nn.Sequential(ResBlock(32, 64, 2), ResBlock(64, 128, 2), ResBlock(128, NOISE_DIM, 2))

    def forward(self, crops):
        x = crops.float()  # pixel values 0-255, the scale the SRM truncation is defined on
        x = torch.cat([self.srm(x), self.constrained(x)], dim=1)
        return self.blocks(self.stem(x)).mean(dim=(2, 3))


class FusionModel(nn.Module):
    """P(AI) from CLIP features plus the noise branch.

    concat: logit = clip_head(clip) + noise_head(noise), i.e. one linear layer on [clip; noise].
    gate:   logit = clip_head(clip) + g * noise_head(noise), g = sigmoid(w . [clip; noise] + b) per image,
            so the model can learn to ignore the noise when it is unreliable (e.g. after compression).
    clip_head starts as the given trained head and noise_head starts at zero, so before training the
    model is exactly the frozen-CLIP model it extends.
    """

    def __init__(self, fusion, clip_head_state):
        super().__init__()
        self.fusion = fusion
        self.clip_head = nn.Linear(FEATURE_DIM, 1)
        self.clip_head.load_state_dict(clip_head_state)
        self.noise = NoiseCNN()
        self.noise_head = nn.Linear(NOISE_DIM, 1)
        nn.init.zeros_(self.noise_head.weight)
        nn.init.zeros_(self.noise_head.bias)
        if fusion == "gate":
            self.gate = nn.Linear(FEATURE_DIM + NOISE_DIM, 1)
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)

    def forward(self, clip_features, crops):
        """Returns (logits [n], gate values [n]; the gate is 1 for concat)."""
        noise = self.noise(crops)
        clip_logit = self.clip_head(clip_features).squeeze(1)
        noise_logit = self.noise_head(noise).squeeze(1)
        if self.fusion == "gate":
            g = torch.sigmoid(self.gate(torch.cat([clip_features, noise.to(clip_features.dtype)], 1))).squeeze(1)
        else:
            g = torch.ones_like(clip_logit)
        return clip_logit + g * noise_logit, g

    def param_groups(self, noise_lr, head_lr, weight_decay):
        """The branch is trained from scratch (higher learning rate); the CLIP head is already trained."""
        new = [p for name, p in self.named_parameters() if not name.startswith("clip_head")]
        return [
            {"params": new, "lr": noise_lr, "weight_decay": weight_decay},
            {"params": self.clip_head.parameters(), "lr": head_lr, "weight_decay": 0.0},
        ]
