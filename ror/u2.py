"""U2: heads trained on equal-format, real-world-damaged images, compared with the baseline head
on clean test images and on damaged copies of the same test images.

u2 is the first version; u2v2 adds platform-style downsizing before compression (see degrade.share_once).
"""
from ror.config import CHECKPOINTS, SEEDS
from ror.experiment import load_train_val, run, train_and_save
from ror.features import CLEAN_TAG, DEGRADED_TAG, U2_TAG, U2V2_TAG
from ror.head import load_head
from ror.views import DEGRADED_TEST_VIEWS

VERSIONS = {"u2": (U2_TAG, "u2_aug"), "u2v2": (U2V2_TAG, "u2v2_aug")}


def main():
    heads = {
        ("retrained_sd14", seed): load_head(CHECKPOINTS / f"retrained_sd14_seed{seed}.pth") for seed in SEEDS
    }
    for model_name, (tag, train_view) in VERSIONS.items():
        train, val = load_train_val(tag, train_view, "jpeg95")
        for seed in SEEDS:
            heads[(model_name, seed)] = train_and_save(train, val, model_name, seed)
    run("u2", heads, {CLEAN_TAG: ["resize"], DEGRADED_TAG: DEGRADED_TEST_VIEWS})


if __name__ == "__main__":
    main()
