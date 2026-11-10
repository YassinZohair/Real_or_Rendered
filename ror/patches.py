"""U3's input: crops at each image's own resolution, decoded once and cached as uint8 arrays.

Resizing interpolates pixels and smooths away the high-frequency noise U3 looks for, so the noise
branch reads a PATCH x PATCH crop with no resizing. 128 px is the size of the smallest test generator
(BigGAN). Smaller images (rare; or half-size "shared" copies of small generators) are mirror-padded.

Damage is applied to the whole image first, as for CLIP, then the crop is taken. Training crops get the
same random augmentation as the CLIP side of their model (B's blur+JPEG, or U2 v2's damage).

    python -m ror.patches          crops for B+U3 training, and the clean and damaged test sets
    python -m ror.patches full     crops for the full model's training (U2 v2 damage)
"""
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from ror import data, degrade
from ror.config import (
    DAMAGED_PER_CLASS,
    EQUAL_FORMAT_QUALITY,
    GENERATORS,
    JPEG_QUALITIES,
    PATCHES,
    SHARED_QUALITY,
    SHARED_SCALE,
    TEST_PER_CLASS,
    TRAIN_GENERATOR,
    TRAIN_TAG,
)
from ror.device import NUM_WORKERS
from ror.features import Job
from ror.views import DEGRADED_TEST_VIEWS, UNIVFD_OPT, data_augment

PATCH = 128


def _pad_to_patch(img):
    w, h = img.size
    if w >= PATCH and h >= PATCH:
        return img
    pixels = np.asarray(img)
    pad_h, pad_w = max(0, PATCH - h), max(0, PATCH - w)
    pad = ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0))
    return transforms.functional.to_pil_image(np.pad(pixels, pad, mode="symmetric"))


def _to_uint8(img):
    return torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1)  # [3, PATCH, PATCH]


def _patch_view(damage, training):
    crop = [transforms.RandomCrop(PATCH), transforms.RandomHorizontalFlip()] if training else [transforms.CenterCrop(PATCH)]
    return transforms.Compose([transforms.Lambda(damage), transforms.Lambda(_pad_to_patch), *crop, _to_uint8])


def _u2v2_damage(img):
    return degrade.real_world(degrade.jpeg(img, EQUAL_FORMAT_QUALITY), platform_downsize=True)


PATCH_VIEWS = {
    "aug": _patch_view(lambda img: data_augment(img, UNIVFD_OPT), training=True),
    "u2v2_aug": _patch_view(_u2v2_damage, training=True),
    "clean": _patch_view(lambda img: img, training=False),
    "shared": _patch_view(lambda img: degrade.jpeg(degrade.shrink(img, SHARED_SCALE), SHARED_QUALITY), False),
}
for _quality in JPEG_QUALITIES:
    PATCH_VIEWS[f"jpeg{_quality}"] = _patch_view(lambda img, q=_quality: degrade.jpeg(img, q), training=False)

# Crop sets. Train sets hold 4 augmented crops per image plus the validation view.
JOBS = {
    "u3": Job("train", [TRAIN_GENERATOR], {"aug": 4, "clean": 1}, None, "_u3"),
    "full": Job("train", [TRAIN_GENERATOR], {"u2v2_aug": 4, "jpeg95": 1}, None, "_full"),
    "test": Job("test", GENERATORS, {"clean": 1}, TEST_PER_CLASS, ""),
    "test_damaged": Job("test", GENERATORS, {view: 1 for view in DEGRADED_TEST_VIEWS}, DAMAGED_PER_CLASS, "_degraded"),
}
COMMANDS = {"u3": ["u3", "test", "test_damaged"], "full": ["full", "test", "test_damaged"]}


def _base(split, generator, tag):
    size_tag = TRAIN_TAG if split == "train" else ""
    return PATCHES / f"{split}_{generator}{size_tag}{tag}"


def patch_path(split, generator, tag, view):
    return _base(split, generator, tag).with_name(_base(split, generator, tag).name + f"_{view}.npy")


def rows_path(split, generator, tag):
    return _base(split, generator, tag).with_name(_base(split, generator, tag).name + "_rows.npy")


def cache(job, generator, batch_size=64):
    """Writes one uint8 array per view ([n, copies, 3, P, P], or [n, 3, P, P] for one copy) and the rows."""
    if rows_path(job.split, generator, job.tag).exists():
        print(f"{rows_path(job.split, generator, job.tag).name}: cached, skipping")
        return
    labels, _ = data.read_index(job.split, generator)
    rows = np.arange(len(labels)) if job.per_class is None else data.balanced_sample(labels, job.per_class)
    names = [name for name, count in job.views.items() for _ in range(count)]
    dataset = data.ViewDataset(job.split, generator, rows, views=[PATCH_VIEWS[name] for name in names])
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=NUM_WORKERS)

    PATCHES.mkdir(parents=True, exist_ok=True)
    out, start = {}, 0
    for view, count in job.views.items():
        shape = (len(rows), count, 3, PATCH, PATCH) if count > 1 else (len(rows), 3, PATCH, PATCH)
        tmp = patch_path(job.split, generator, job.tag, view).with_suffix(".tmp.npy")
        out[view] = (np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=shape), start, count)
        start += count

    done = 0
    for batch in tqdm(loader, desc=f"{job.split} {generator}{job.tag}"):
        for array, first, count in out.values():
            block = batch[:, first : first + count].numpy()
            array[done : done + len(batch)] = block if count > 1 else block[:, 0]
        done += len(batch)

    # Arrays are renamed into place only when complete, and the rows file last, so a dropped session
    # never leaves a set that looks finished.
    for view, (array, _, _) in out.items():
        array.flush()
        os.replace(patch_path(job.split, generator, job.tag, view).with_suffix(".tmp.npy"),
                   patch_path(job.split, generator, job.tag, view))
    np.save(rows_path(job.split, generator, job.tag), rows)


def load(split, generator, tag, view):
    """(rows, crops) with the crops memory-mapped from disk."""
    return np.load(rows_path(split, generator, tag)), np.load(patch_path(split, generator, tag, view), mmap_mode="r")


def main(command):
    torch.manual_seed(0)
    for name in COMMANDS[command]:
        job = JOBS[name]
        for generator in job.generators:
            cache(job, generator)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "u3")
