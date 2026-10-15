"""Run images through the frozen CLIP encoder once and cache the features."""
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ror import data
from ror.config import (
    CLIP_ARCH,
    FEATURES,
    GENERATORS,
    RELEASED_FC,
    SAMPLE_SEED,
    TEST_PER_CLASS,
    TRAIN_GENERATOR,
)
from ror.views import DEGRADED_TEST_VIEWS, VIEWS

# Feature files are named <split>_<generator><tag>.npz.
CLEAN_TAG, U2_TAG, DEGRADED_TAG = "", "_u2", "_degraded"


@dataclass
class Job:
    split: str
    generators: list
    views: dict  # view name -> copies per image
    per_class: int | None  # None keeps every image
    tag: str


JOBS = {
    "baseline": [
        Job("train", [TRAIN_GENERATOR], {"aug": 4, "resize": 1}, None, CLEAN_TAG),
        Job("test", GENERATORS, {"resize": 1, "crop": 1}, TEST_PER_CLASS, CLEAN_TAG),
    ],
    "u2": [
        Job("train", [TRAIN_GENERATOR], {"u2_aug": 4, "jpeg95": 1}, None, U2_TAG),
        Job("test", GENERATORS, {view: 1 for view in DEGRADED_TEST_VIEWS}, TEST_PER_CLASS, DEGRADED_TAG),
    ],
}


def load_univfd():
    """The repo's model: frozen CLIP ViT-L/14 plus the released linear layer."""
    from models import get_model  # resolved through the sys.path entry added in ror.views

    model = get_model(CLIP_ARCH)
    model.fc.load_state_dict(torch.load(RELEASED_FC, map_location="cpu"))
    return model.eval().cuda()


def feature_path(split, generator, tag=CLEAN_TAG):
    return FEATURES / f"{split}_{generator}{tag}.npz"


@torch.no_grad()
def extract(model, dataset_args, view_counts, batch_size=64):
    """Returns {view: [n, 768]} for single views and {view: [n, count, 768]} for repeated ones."""
    names = [name for name, count in view_counts.items() for _ in range(count)]
    dataset = data.ViewDataset(*dataset_args, views=[VIEWS[name] for name in names])
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=os.cpu_count())
    chunks = []
    for x in tqdm(loader):
        n, v = x.shape[:2]
        with torch.autocast("cuda", dtype=torch.float16):
            feats = model.model.encode_image(x.flatten(0, 1).cuda())
        chunks.append(feats.float().view(n, v, -1).cpu())
    feats = torch.cat(chunks).numpy()

    out, start = {}, 0
    for name, count in view_counts.items():
        block = feats[:, start : start + count]
        out[name] = block[:, 0] if count == 1 else block
        start += count
    return out


def cache(model, job, generator):
    path = feature_path(job.split, generator, job.tag)
    if path.exists():
        print(f"{path.name}: cached, skipping")
        return
    labels, image_paths = data.read_index(job.split, generator)
    rows = np.arange(len(labels)) if job.per_class is None else data.balanced_sample(labels, job.per_class)
    print(f"{path.name}: {len(rows)} images")
    feats = extract(model, (job.split, generator, rows), job.views)

    # Written to a temp name first so an interrupted session never leaves a half file on Drive.
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, row=rows, label=labels[rows], image_path=image_paths[rows], **feats)
    tmp.replace(path)


def main(experiment):
    FEATURES.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SAMPLE_SEED)
    model = load_univfd()
    for job in JOBS[experiment]:
        for generator in job.generators:
            cache(model, job, generator)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "baseline")
