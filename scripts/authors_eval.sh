#!/bin/bash
# Runs the authors' validate.py, unmodified, with their released weights on their released
# diffusion test set, so the numbers can be compared with the UnivFD paper.
set -e
cd /content/UniversalFakeDetect

if [ ! -d diffusion_datasets ]; then
    gdown 1FXlGIRh_Ud3cScMgSVDbEWmPDmjcrm1t -O diffusion_datasets.zip
    unzip -q diffusion_datasets.zip
fi

# dataset_paths.py also lists the 19 GB GAN test set, which we do not download.
python - <<'PY'
from dataset_paths import DATASET_PATHS
kept = [d for d in DATASET_PATHS if d["real_path"].startswith("./diffusion_datasets")]
open("dataset_paths.py", "w").write(f"DATASET_PATHS = {kept!r}\n")
PY

python validate.py --arch=CLIP:ViT-L/14 --ckpt=pretrained_weights/fc_weights.pth --result_folder=clip_vitl14
echo "--- AP"; cat clip_vitl14/ap.txt
echo "--- real acc / fake acc / acc (threshold 0.5)"; cat clip_vitl14/acc0.txt
