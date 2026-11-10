"""Модель распознавания: предобученная ResNet-50 (ImageNet) из torchvision.

ImageNet содержит много классов транспорта (такси, пикап, фургон, автобус ...).
Вероятность «на кадре есть транспорт» = сумма вероятностей этих классов.
"""
import torch
from torchvision.models import ResNet50_Weights, resnet50

# Названия классов ImageNet, относящихся к дорожному транспорту
VEHICLE_CLASS_NAMES = {
    "ambulance", "beach wagon", "cab", "convertible", "jeep", "limousine",
    "minivan", "Model T", "racer", "sports car", "police van", "moving van",
    "pickup", "tow truck", "trailer truck", "fire engine", "garbage truck",
    "school bus", "minibus", "trolleybus", "recreational vehicle", "car wheel",
    "grille", "tractor", "forklift", "snowplow", "go-kart", "golf cart",
    "half track", "moped", "motor scooter", "streetcar", "amphibian",
}

WEIGHTS = ResNet50_Weights.IMAGENET1K_V2


def vehicle_class_ids():
    categories = WEIGHTS.meta["categories"]
    ids = [i for i, name in enumerate(categories) if name in VEHICLE_CLASS_NAMES]
    if not ids:
        raise RuntimeError("Не найдено ни одного класса транспорта в ImageNet")
    return ids


def load_model(device: torch.device):
    """Каждый вариант эксперимента получает «свежую» копию модели."""
    model = resnet50(weights=WEIGHTS)
    model.eval()
    return model.to(device)
