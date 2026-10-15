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

# U2: every training image is re-saved as JPEG at this quality, so file format cannot give away the label.
EQUAL_FORMAT_QUALITY = 95
# Robustness tests: JPEG qualities, and a "shared" copy shrunk by SHARED_SCALE and saved at SHARED_QUALITY.
JPEG_QUALITIES = [95, 75, 50, 30]
SHARED_SCALE = 0.5
SHARED_QUALITY = 75

CLIP_ARCH = "CLIP:ViT-L/14"
FEATURE_DIM = 768
