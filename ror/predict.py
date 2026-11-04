"""Score single images: the probability that each one is AI-generated.

    python -m ror.predict photo.jpg [more.png ...] [--head PATH] [--view resize|crop]

The full pipeline for one image, with no cache: decode, the test view (resize to 256, centre crop 224,
CLIP normalisation), CLIP ViT-L/14 -> 768 numbers, a trained linear head -> sigmoid -> P(AI).
--head takes a baseline or U2 head (.pth), or a U1 checkpoint (.pt: LoRA adapters plus their head).
"""
import argparse
from pathlib import Path

import torch
from PIL import Image

from ror.config import CHECKPOINTS
from ror.device import DEVICE, autocast
from ror.features import load_univfd
from ror.head import load_head, predict
from ror.views import VIEWS

DEFAULT_HEAD = CHECKPOINTS / "retrained_sd14_seed0.pth"
THRESHOLD = 0.5


class Detector:
    """Loads CLIP and one head once, then scores any number of PIL images."""

    def __init__(self, head_path=DEFAULT_HEAD, view="resize"):
        if head_path.suffix == ".pt":
            from ror.u1 import load_u1

            self.clip = load_u1(head_path)
            self.head = self.clip.fc
        else:
            self.clip = load_univfd()
            self.head = load_head(head_path)
        self.view = VIEWS[view]

    @torch.no_grad()
    def __call__(self, img):
        x = self.view(img.convert("RGB")).unsqueeze(0).to(DEVICE)  # [1, 3, 224, 224]
        with autocast():
            features = self.clip.model.encode_image(x)  # [1, 768]
        return float(predict(self.head, features.float())[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--head", type=Path, default=DEFAULT_HEAD, help="a trained head (.pth) or U1 checkpoint (.pt)")
    parser.add_argument("--view", default="resize", choices=["resize", "crop"], help="test preprocessing")
    args = parser.parse_args()

    detector = Detector(args.head, args.view)
    print(f"head: {args.head.name}, view: {args.view}, device: {DEVICE}")
    for path in args.images:
        p = detector(Image.open(path))
        print(f"{path.name}: P(AI) = {p:.3f} -> {'AI-generated' if p > THRESHOLD else 'real'}")


if __name__ == "__main__":
    main()
