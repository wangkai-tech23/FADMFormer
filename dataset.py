import os
import random
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import Dataset

try:
    from albumentations import (
        CLAHE,
        Compose,
        Emboss,
        GaussNoise,
        GridDistortion,
        HueSaturationValue,
        OneOf,
        OpticalDistortion,
        PiecewiseAffine,
        RandomBrightnessContrast,
        RandomRotate90,
        Sharpen,
        ShiftScaleRotate,
        ToGray,
        Transpose,
    )
    try:
        from albumentations import Flip
    except Exception:
        from albumentations import HorizontalFlip as Flip
except Exception:
    Compose = None


IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _build_aug(p: float = 0.5):
    """Optional albumentations augmentation used during training."""
    if Compose is None or p <= 0:
        return None
    return Compose(
        [
            RandomRotate90(p=0.5),
            Flip(p=0.5),
            Transpose(p=0.5),
            ToGray(p=0.1),
            OneOf([GaussNoise()], p=0.2),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            OneOf(
                [
                    OpticalDistortion(p=0.3),
                    GridDistortion(p=0.1),
                    PiecewiseAffine(p=0.3),
                ],
                p=0.2,
            ),
            OneOf(
                [
                    CLAHE(clip_limit=2),
                    Sharpen(),
                    Emboss(),
                    RandomBrightnessContrast(),
                ],
                p=0.5,
            ),
            HueSaturationValue(p=0.3),
        ],
        p=p,
    )


def _clean_id(img_id: str, form: str = ".png") -> str:
    img_id = str(img_id).strip()
    if not img_id:
        return img_id
    base = os.path.basename(img_id)
    stem, ext = os.path.splitext(base)
    if ext.lower() in IMG_EXTENSIONS:
        return stem
    if form and base.endswith(form):
        return base[: -len(form)]
    return base


def _find_file(folder: str, img_id: str, form: str = ".png") -> str:
    """Resolve image/mask path robustly whether txt ids contain extensions or not."""
    img_id_raw = str(img_id).strip()
    if not img_id_raw:
        raise FileNotFoundError("Empty image id.")

    # absolute/relative path in txt
    if os.path.isfile(img_id_raw):
        return img_id_raw

    candidates: List[str] = []
    candidates.append(os.path.join(folder, img_id_raw))

    stem = _clean_id(img_id_raw, form=form)
    if form:
        candidates.append(os.path.join(folder, stem + form))
    for ext in IMG_EXTENSIONS:
        candidates.append(os.path.join(folder, stem + ext))

    for path in candidates:
        if os.path.isfile(path):
            return path

    raise FileNotFoundError(
        f"Cannot find file for id='{img_id_raw}' in '{folder}'. Tried examples: {candidates[:4]}"
    )


def _mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    arr = np.array(mask, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = (arr > 127).astype(np.float32)
    arr = np.expand_dims(arr, axis=0)
    return torch.from_numpy(arr)


class TrainSetLoader(Dataset):
    NUM_CLASS = 1

    def __init__(
        self,
        dataset_dir: str,
        img_id: Sequence[str],
        base_size: int = 512,
        crop_size: int = 512,
        transform=None,
        form: str = ".png",
        aug: float = 0.0,
        useprior: bool = True,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.transform = transform
        self._items = [_clean_id(x, form=form) for x in img_id if str(x).strip()]
        self.masks = os.path.join(dataset_dir, "masks")
        self.images = os.path.join(dataset_dir, "images")
        self.prior = self.masks
        self.base_size = int(base_size)
        self.crop_size = int(crop_size)
        self.form = form
        self.aug_p = float(aug)
        self.useprior = useprior
        self.aug = _build_aug(self.aug_p)

    def _sync_transform(self, img: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if random.random() < 0.5:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        crop_size = self.crop_size
        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh

        img = img.resize((ow, oh), Image.Resampling.BICUBIC)
        mask = mask.resize((ow, oh), Image.Resampling.NEAREST)

        if short_size < crop_size:
            padh = max(crop_size - oh, 0)
            padw = max(crop_size - ow, 0)
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)

        w, h = img.size
        x1 = random.randint(0, max(w - crop_size, 0))
        y1 = random.randint(0, max(h - crop_size, 0))
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))

        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))

        if self.aug is not None:
            img_np = np.array(img.convert("RGB"))
            mask_np = np.array(mask.convert("L"))
            augmented = self.aug(image=img_np, mask=mask_np)
            img = Image.fromarray(augmented["image"].astype(np.uint8)).convert("RGB")
            mask = Image.fromarray(augmented["mask"].astype(np.uint8)).convert("L")

        return img, mask

    def __getitem__(self, idx: int):
        cv2.setNumThreads(0)
        img_id = self._items[idx]
        img_path = _find_file(self.images, img_id, self.form)
        label_path = _find_file(self.masks, img_id, self.form)

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(label_path).convert("L")
        img, mask = self._sync_transform(img, mask)

        if self.transform is not None:
            img = self.transform(img)
        mask_tensor = _mask_to_tensor(mask)
        return img, mask_tensor

    def __len__(self) -> int:
        return len(self._items)


class TestSetLoader(Dataset):
    NUM_CLASS = 1

    def __init__(
        self,
        dataset_dir: str,
        img_id: Sequence[str],
        transform=None,
        base_size: int = 512,
        crop_size: int = 512,
        form: str = ".png",
        return_id: bool = False,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.transform = transform
        self._items = [_clean_id(x, form=form) for x in img_id if str(x).strip()]
        self.masks = os.path.join(dataset_dir, "masks")
        self.images = os.path.join(dataset_dir, "images")
        self.base_size = int(base_size)
        self.crop_size = int(crop_size)
        self.form = form
        self.return_id = bool(return_id)

    def _testval_sync_transform(self, img: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        size = self.base_size
        img = img.resize((size, size), Image.Resampling.BICUBIC)
        mask = mask.resize((size, size), Image.Resampling.NEAREST)
        return img, mask

    def __getitem__(self, idx: int):
        cv2.setNumThreads(0)
        img_id = self._items[idx]
        img_path = _find_file(self.images, img_id, self.form)
        label_path = _find_file(self.masks, img_id, self.form)

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(label_path).convert("L")
        img, mask = self._testval_sync_transform(img, mask)

        if self.transform is not None:
            img = self.transform(img)
        mask_tensor = _mask_to_tensor(mask)
        if self.return_id:
            return img, mask_tensor, img_id
        return img, mask_tensor

    def __len__(self) -> int:
        return len(self._items)
