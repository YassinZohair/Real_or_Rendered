"""Image-to-tensor pipelines. Resizing and augmentation are the UnivFD repo's own functions."""
import sys
from types import SimpleNamespace

from torchvision import transforms

from ror import degrade
from ror.config import EQUAL_FORMAT_QUALITY, JPEG_QUALITIES, SHARED_QUALITY, SHARED_SCALE, UNIVFD

sys.path.insert(0, str(UNIVFD))
from data.datasets import MEAN, STD, custom_resize, data_augment  # noqa: E402

# The repo's training defaults (options/base_options.py).
UNIVFD_OPT = SimpleNamespace(
    rz_interp=["bilinear"],
    loadSize=256,
    blur_prob=0.5,
    blur_sig=[0.0, 3.0],
    jpg_prob=0.5,
    jpg_method=["cv2", "pil"],
    jpg_qual=list(range(30, 101)),
)
CROP = 224

_resize = transforms.Lambda(lambda img: custom_resize(img, UNIVFD_OPT))
_augment = transforms.Lambda(lambda img: data_augment(img, UNIVFD_OPT))
_to_tensor = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=MEAN["clip"], std=STD["clip"])]
)

VIEWS = {
    # Repo's training pipeline: resize, blur/JPEG augmentation, random crop, flip.
    "aug": transforms.Compose(
        [_resize, _augment, transforms.RandomCrop(CROP), transforms.RandomHorizontalFlip(), _to_tensor]
    ),
    # Our main test protocol, as stated in the proposal.
    "resize": transforms.Compose([_resize, transforms.CenterCrop(CROP), _to_tensor]),
    # The repo's validate.py protocol: no resize, so each generator is seen at its native scale.
    "crop": transforms.Compose([transforms.CenterCrop(CROP), _to_tensor]),
}


def _damaged_then_resized(damage):
    """Damage the image at its original size, as happens online, then apply the main test protocol."""
    return transforms.Compose([transforms.Lambda(damage), _resize, transforms.CenterCrop(CROP), _to_tensor])


def _u2_training(platform_downsize):
    """U2 training: equal file format first, then real-world damage, then the repo's crop and flip."""
    return transforms.Compose(
        [
            transforms.Lambda(lambda img: degrade.jpeg(img, EQUAL_FORMAT_QUALITY)),
            transforms.Lambda(lambda img: degrade.real_world(img, platform_downsize)),
            _resize,
            transforms.RandomCrop(CROP),
            transforms.RandomHorizontalFlip(),
            _to_tensor,
        ]
    )


VIEWS["u2_aug"] = _u2_training(platform_downsize=False)
VIEWS["u2v2_aug"] = _u2_training(platform_downsize=True)
for _quality in JPEG_QUALITIES:
    VIEWS[f"jpeg{_quality}"] = _damaged_then_resized(lambda img, q=_quality: degrade.jpeg(img, q))
VIEWS["shared"] = _damaged_then_resized(
    lambda img: degrade.jpeg(degrade.shrink(img, SHARED_SCALE), SHARED_QUALITY)
)

# jpeg95 doubles as the equal-format test: every real and fake image saved the same way.
DEGRADED_TEST_VIEWS = [f"jpeg{q}" for q in JPEG_QUALITIES] + ["shared"]
