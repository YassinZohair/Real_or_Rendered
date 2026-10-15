"""Baseline results on GenImage: the released ProGAN-trained layer, and our layer retrained on SD v1.4."""
from ror.config import RELEASED_FC, SEEDS
from ror.experiment import load_train_val, run, train_and_save
from ror.features import CLEAN_TAG
from ror.head import load_head


def main():
    train, val = load_train_val(CLEAN_TAG, "aug", "resize")
    heads = {("released_progan", 0): load_head(RELEASED_FC)}
    for seed in SEEDS:
        heads[("retrained_sd14", seed)] = train_and_save(train, val, "retrained_sd14", seed)
    run("baseline", heads, {CLEAN_TAG: ["resize", "crop"]})


if __name__ == "__main__":
    main()
