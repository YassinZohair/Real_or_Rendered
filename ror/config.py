import os
from pathlib import Path


def _path(variable, default):
    """A path that can be overridden with an environment variable, so the code also runs outside Colab."""
    return Path(os.environ.get(variable, default))


# How many of the 189 SD v1.4 training shards to use: 6 = Milestone 1 (10,505 images), 26 = about 44,000.
N_TRAIN_SHARDS = int(os.environ.get("ROR_TRAIN_SHARDS", 6))
# Training-size tag. Train features, checkpoints and results from different training sizes must never mix,
# because cached files are skipped when they exist. The Milestone 1 size keeps its original, untagged names.
TRAIN_TAG = "" if N_TRAIN_SHARDS == 6 else f"_{N_TRAIN_SHARDS}shards"

OUT = _path("ROR_OUT", "/content/drive/MyDrive/aml_real_or_rendered")
FEATURES = OUT / "features"  # test features do not depend on the training size, so they are shared
RESULTS = OUT / f"results{TRAIN_TAG}"
CHECKPOINTS = OUT / f"checkpoints{TRAIN_TAG}"

DATA = _path("ROR_DATA", "/content/data")
UNIVFD = _path("ROR_UNIVFD", "/content/UniversalFakeDetect")
RELEASED_FC = UNIVFD / "pretrained_weights" / "fc_weights.pth"

HF_REPO = "nebula/GenImage-arrow"
TRAIN_GENERATOR = "stable_diffusion_v_1_4"
GENERATORS = [
    "ADM",
    "BigGAN",
    "glide",
    "Midjourney",
    "stable_diffusion_v_1_4",
    "stable_diffusion_v_1_5",
    "VQDM",
    "wukong",
]

TEST_PER_CLASS = 2000
VAL_FRACTION = 0.1
SEEDS = [0, 1, 2]
SAMPLE_SEED = 0

# U2: every training image is re-saved as JPEG at this quality, so file format cannot give away the label.
EQUAL_FORMAT_QUALITY = 95
# Robustness tests: JPEG qualities, and a "shared" copy shrunk by SHARED_SCALE and saved at SHARED_QUALITY.
JPEG_QUALITIES = [95, 75, 50, 30]
SHARED_SCALE = 0.5
SHARED_QUALITY = 75

CLIP_ARCH = "CLIP:ViT-L/14"
FEATURE_DIM = 768
