"""GenImage access through the Hugging Face Arrow mirror."""
from io import BytesIO

import numpy as np
import pyarrow as pa
import torch
from huggingface_hub import snapshot_download
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from ror.config import DATA, HF_REPO, N_TRAIN_SHARDS, SAMPLE_SEED, TRAIN_GENERATOR

ImageFile.LOAD_TRUNCATED_IMAGES = True

REAL, FAKE = 0, 1


def download():
    train = [f"data/train/{TRAIN_GENERATOR}/data-{i:05d}-of-*.arrow" for i in range(N_TRAIN_SHARDS)]
    snapshot_download(
        HF_REPO,
        repo_type="dataset",
        local_dir=DATA,
        allow_patterns=train + ["data/test/*/*.arrow"],
        max_workers=8,
    )


def shard_paths(split, generator):
    paths = sorted((DATA / "data" / split / generator).glob("*.arrow"))
    assert paths, f"no shards for {split}/{generator}; run python -m ror.data first"
    return paths


def open_table(paths):
    return pa.concat_tables([pa.ipc.open_stream(pa.memory_map(str(p))).read_all() for p in paths])


def read_index(split, generator):
    """Labels and original file paths for every row, without touching the image bytes."""
    table = open_table(shard_paths(split, generator)).select(["label", "image_path"])
    labels = table.column("label").to_numpy()
    image_paths = np.array(table.column("image_path").to_pylist())
    is_ai = np.char.find(image_paths, "/ai/") >= 0
    assert ((labels == FAKE) == is_ai).all(), "label column disagrees with the ai/nature folders"
    return labels, image_paths


def balanced_sample(labels, per_class):
    """Test shards are sorted real-then-fake, so sample rows at random instead of taking a prefix."""
    rng = np.random.default_rng(SAMPLE_SEED)
    rows = [rng.choice(np.flatnonzero(labels == c), per_class, replace=False) for c in (REAL, FAKE)]
    return np.sort(np.concatenate(rows))


class ViewDataset(Dataset):
    """Decodes each image once and returns one tensor per requested view, stacked."""

    def __init__(self, split, generator, rows, views):
        self.paths = shard_paths(split, generator)
        self.rows = rows
        self.views = views
        self._images = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        # Opened lazily so each DataLoader worker maps the shards itself.
        if self._images is None:
            self._images = open_table(self.paths).column("image")
        raw = self._images[int(self.rows[i])].as_py()
        img = Image.open(BytesIO(raw)).convert("RGB")
        return torch.stack([view(img) for view in self.views])


if __name__ == "__main__":
    download()
