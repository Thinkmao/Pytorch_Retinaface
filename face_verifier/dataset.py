import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class Sample:
    image_path: str
    label: int


class FaceBinaryDataset(Dataset):
    """Binary classification dataset.

    List file format (utf-8):
        relative/or/abs/path/to/image.jpg <label>

    where label is 0 (non-face) or 1 (face).
    """

    def __init__(
        self,
        root_dir: str,
        list_file: str,
        image_size: int = 112,
        margin: float = 0.0,
        augment: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.list_file = Path(list_file)
        self.image_size = image_size
        self.margin = max(0.0, float(margin))
        self.augment = augment
        self.samples = self._load_samples(self.list_file)

    def _load_samples(self, list_file: Path) -> List[Sample]:
        samples: List[Sample] = []
        with list_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) != 2:
                    raise ValueError(f"Invalid list line: {line}")
                image_path, label = parts
                label_int = int(label)
                if label_int not in (0, 1):
                    raise ValueError(f"Label must be 0/1, got {label_int}")
                samples.append(Sample(image_path=image_path, label=label_int))
        if not samples:
            raise RuntimeError(f"No samples found in {list_file}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_path(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else (self.root_dir / p)

    def _safe_margin_crop(self, img: np.ndarray) -> np.ndarray:
        if self.margin <= 0:
            return img
        h, w = img.shape[:2]
        mx, my = int(w * self.margin), int(h * self.margin)
        canvas = cv2.copyMakeBorder(img, my, my, mx, mx, cv2.BORDER_REFLECT_101)
        return canvas

    def _apply_augment(self, img: np.ndarray) -> np.ndarray:
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
        if random.random() < 0.35:
            k = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)
        if random.random() < 0.35:
            alpha = random.uniform(0.8, 1.2)
            beta = random.uniform(-12, 12)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        if random.random() < 0.2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return img

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        img_path = self._resolve_path(sample.image_path)
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")

        img = self._safe_margin_crop(img)
        if self.augment:
            img = self._apply_augment(img)

        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        img = img.transpose(2, 0, 1)

        x = torch.from_numpy(img)
        y = torch.tensor(sample.label, dtype=torch.float32)
        return x, y
