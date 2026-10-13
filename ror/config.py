import os
from pathlib import Path

OUT = Path(os.environ.get("ROR_OUT", "/content/drive/MyDrive/aml_real_or_rendered"))
FEATURES = OUT / "features"
RESULTS = OUT / "results"
CHECKPOINTS = OUT / "checkpoints"

DATA = Path("/content/data")
UNIVFD = Path("/content/UniversalFakeDetect")
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

N_TRAIN_SHARDS = 6
TEST_PER_CLASS = 2000
VAL_FRACTION = 0.1
SEEDS = [0, 1, 2]
SAMPLE_SEED = 0

CLIP_ARCH = "CLIP:ViT-L/14"
FEATURE_DIM = 768
