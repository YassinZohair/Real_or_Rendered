"""Real-world image damage for U2: what images go through when they are shared online or re-photographed."""
import random
from io import BytesIO

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def jpeg(img, quality):
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def shrink(img, factor):
    w, h = img.size
    return img.resize((max(1, round(w * factor)), max(1, round(h * factor))), Image.BICUBIC)


def rescale(img, factor):
    """Shrink and enlarge back: detail is lost but the size is unchanged."""
    return shrink(img, factor).resize(img.size, Image.BICUBIC)


def downsize(img, short_side):
    """Shrink so the shorter side is at most short_side, as sharing apps do. Never enlarges."""
    factor = short_side / min(img.size)
    return shrink(img, factor) if factor < 1 else img


def blur(img, sigma):
    return img.filter(ImageFilter.GaussianBlur(sigma))


def noise(img, sigma):
    # Drawn from Python's random module, which PyTorch seeds separately in every DataLoader worker.
    rng = np.random.default_rng(random.getrandbits(32))
    pixels = np.asarray(img, dtype=np.float32) + rng.normal(0, sigma, (img.size[1], img.size[0], 3))
    return Image.fromarray(pixels.clip(0, 255).astype(np.uint8))


def colour(img, strength):
    for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        img = enhancer(img).enhance(1 + random.uniform(-strength, strength))
    return img


def rephotograph(img):
    """A rough stand-in for photographing a screen: slight tilt, brightness curve, softening and sensor noise."""
    w, h = img.size
    img = img.rotate(random.uniform(-3, 3), resample=Image.BICUBIC)
    img = img.crop((round(w * 0.05), round(h * 0.05), round(w * 0.95), round(h * 0.95)))
    gamma = random.uniform(0.8, 1.2)
    img = img.point(lambda v: 255 * (v / 255) ** gamma)
    img = blur(img, random.uniform(0.5, 1.2))
    return noise(img, random.uniform(2, 6))


def share_once(img, platform_downsize):
    """One trip through a sharing app: possibly downsized, filtered or noisy, usually recompressed.

    With platform_downsize the image can stay small, so JPEG damage lands at the scale the
    detector sees. Without it (U2's first version) every shrink is undone before compression.
    """
    if platform_downsize and random.random() < 0.4:
        img = downsize(img, random.randint(128, 384))
    elif random.random() < 0.4:
        img = rescale(img, random.uniform(0.25, 0.9))
    if random.random() < 0.3:
        img = blur(img, random.uniform(0.1, 2.0))
    if random.random() < 0.3:
        img = colour(img, 0.15)
    if random.random() < 0.2:
        img = noise(img, random.uniform(1, 8))
    if random.random() < 0.7:
        img = jpeg(img, random.randint(30, 95))
    return img


def real_world(img, platform_downsize):
    """U2's training damage. A fifth of images stay untouched so the head still sees clean ones."""
    if random.random() < 0.2:
        return img
    if random.random() < 0.1:
        img = rephotograph(img)
    for _ in range(random.randint(1, 3)):
        img = share_once(img, platform_downsize)
    return img
