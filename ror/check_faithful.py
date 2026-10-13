"""Check that cached features plus the released layer match the UnivFD repo's model run end to end."""
import numpy as np
import torch

from ror import data
from ror.baseline import released_head
from ror.config import TRAIN_GENERATOR
from ror.features import feature_path, load_univfd
from ror.head import predict
from ror.views import VIEWS

N_IMAGES = 200


@torch.no_grad()
def main():
    cached = np.load(feature_path("test", TRAIN_GENERATOR))
    pick = np.random.default_rng(0).choice(len(cached["row"]), N_IMAGES, replace=False)

    # validate.py's own preprocessing is the "crop" view, here in full precision.
    dataset = data.ViewDataset("test", TRAIN_GENERATOR, cached["row"][pick], views=[VIEWS["crop"]])
    images = torch.stack([dataset[i][0] for i in range(N_IMAGES)]).cuda()
    model = load_univfd()
    end_to_end = torch.cat([torch.sigmoid(model(b)).squeeze(1) for b in images.split(50)]).cpu().numpy()

    from_cache = predict(released_head(), cached["crop"][pick])

    print(f"max |probability difference| over {N_IMAGES} images: {np.abs(end_to_end - from_cache).max():.5f}")
    print(f"same decision at 0.5: {((end_to_end > 0.5) == (from_cache > 0.5)).mean() * 100:.1f}%")


if __name__ == "__main__":
    main()
