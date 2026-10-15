"""U2: a head trained on equal-format, real-world-damaged images, compared with the baseline head
on clean test images and on damaged copies of the same test images."""
from ror.config import CHECKPOINTS, SEEDS
from ror.experiment import load_train_val, run, train_and_save
from ror.features import CLEAN_TAG, DEGRADED_TAG, U2_TAG
from ror.head import load_head
from ror.views import DEGRADED_TEST_VIEWS


def main():
    train, val = load_train_val(U2_TAG, "u2_aug", "jpeg95")
    heads = {
        ("retrained_sd14", seed): load_head(CHECKPOINTS / f"retrained_sd14_seed{seed}.pth") for seed in SEEDS
    }
    for seed in SEEDS:
        heads[("u2", seed)] = train_and_save(train, val, "u2", seed)
    run("u2", heads, {CLEAN_TAG: ["resize"], DEGRADED_TAG: DEGRADED_TEST_VIEWS})


if __name__ == "__main__":
    main()
