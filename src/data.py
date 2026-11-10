"""Подготовка «кадров» для эксперимента.

Реальных камер видеофиксации у нас нет, поэтому поток кадров имитируется
открытым набором изображений CIFAR-10 (тестовая часть):
  * положительный класс «транспорт»   — automobile, truck;
  * отрицательный класс «не транспорт» — животные (bird, cat, deer, dog, frog, horse).
Самолёты и корабли исключены, т.к. это не дорожный транспорт.

Задача модели — отфильтровать кадры, на которых есть автотранспорт
(первичный этап видеоаналитики перед распознаванием номеров и т.п.).
"""
import numpy as np
import torch
from torchvision import datasets

POSITIVE_CLASSES = [1, 9]               # automobile, truck
NEGATIVE_CLASSES = [2, 3, 4, 5, 6, 7]   # животные


def load_frames(n_images: int, root: str = "data", seed: int = 42):
    """Возвращает (frames, labels).

    frames — uint8-тензор формы [N, 3, 32, 32] (как «сырые» кадры с камеры);
    labels — int64-тензор [N]: 1 = на кадре есть транспорт, 0 = нет.
    Выборка сбалансирована: половина кадров с транспортом.
    """
    ds = datasets.CIFAR10(root=root, train=False, download=True)
    images = ds.data                      # numpy [10000, 32, 32, 3] uint8
    targets = np.array(ds.targets)

    rng = np.random.default_rng(seed)
    pos_idx = np.where(np.isin(targets, POSITIVE_CLASSES))[0]
    neg_idx = np.where(np.isin(targets, NEGATIVE_CLASSES))[0]
    n_pos = n_images // 2
    n_neg = n_images - n_pos
    if n_pos > len(pos_idx):
        raise ValueError(f"Слишком много кадров: максимум {2 * len(pos_idx)}")

    selected = np.concatenate([
        rng.choice(pos_idx, n_pos, replace=False),
        rng.choice(neg_idx, n_neg, replace=False),
    ])
    rng.shuffle(selected)

    frames = torch.from_numpy(images[selected]).permute(0, 3, 1, 2).contiguous()
    labels = torch.from_numpy(np.isin(targets[selected], POSITIVE_CLASSES).astype(np.int64))
    return frames, labels
