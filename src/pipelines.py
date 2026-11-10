"""Варианты конвейера обработки кадров: от исходного (неоптимизированного)
к полностью оптимизированному. Каждый следующий вариант добавляет одну
оптимизацию к предыдущему — так видно вклад каждой из них.
"""
import contextlib
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
import torchvision.transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_SIZE = 224


@dataclass(frozen=True)
class Variant:
    key: str                     # короткий идентификатор
    title: str                   # подпись для отчёта
    batch_size: int = 1
    inference_mode: bool = False  # отключение автоградиента
    gpu_preprocess: bool = False  # предобработка на GPU вместо CPU/PIL
    channels_last: bool = False   # формат памяти NHWC + cudnn.benchmark
    amp_fp16: bool = False        # смешанная точность FP16
    compile: bool = False         # torch.compile
    sync_per_batch: bool = True   # перенос результата на CPU после каждого батча


BASELINE = Variant("v0_baseline", "V0 Исходная версия (batch=1, FP32, CPU-предобработка)")
V1 = replace(BASELINE, key="v1_inference_mode", title="V1 + inference_mode", inference_mode=True)
V2 = replace(V1, key="v2_batching", title="V2 + пакетная обработка (batch=64)",
             batch_size=64, sync_per_batch=False)
V3 = replace(V2, key="v3_gpu_preprocess", title="V3 + предобработка на GPU", gpu_preprocess=True)
V4 = replace(V3, key="v4_channels_last", title="V4 + channels_last, cudnn.benchmark",
             channels_last=True)
V5 = replace(V4, key="v5_fp16", title="V5 + смешанная точность FP16", amp_fp16=True)
V6 = replace(V5, key="v6_compile", title="V6 + torch.compile", compile=True)

ALL_VARIANTS = [BASELINE, V1, V2, V3, V4, V5, V6]

# Исходная предобработка: каждый кадр отдельно через PIL на CPU
_cpu_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def prepare_model(model: torch.nn.Module, variant: Variant, device: torch.device):
    """Применяет к модели оптимизации варианта. Возвращает (model, compile_error)."""
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = variant.channels_last
    if variant.channels_last:
        model = model.to(memory_format=torch.channels_last)
    compile_error = None
    if variant.compile:
        try:
            model = torch.compile(model)
        except Exception as exc:  # компиляция может быть недоступна в среде
            compile_error = repr(exc)
    return model, compile_error


class GpuPreprocessor:
    def __init__(self, device: torch.device, channels_last: bool):
        self.device = device
        self.channels_last = channels_last
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def __call__(self, batch_u8: torch.Tensor) -> torch.Tensor:
        x = batch_u8.to(self.device, non_blocking=True).float().div_(255.0)
        x = F.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        return x


def run_pipeline(model, variant: Variant, frames: torch.Tensor, vehicle_ids: torch.Tensor,
                 device: torch.device):
    """Прогоняет все кадры через модель.

    Возвращает (top1, vehicle_prob) — CPU-тензоры [N]:
    top1 — индекс наиболее вероятного класса ImageNet,
    vehicle_prob — суммарная вероятность классов транспорта.
    """
    n = frames.shape[0]
    bs = min(variant.batch_size, n)
    grad_ctx = torch.inference_mode() if variant.inference_mode else contextlib.nullcontext()
    use_amp = variant.amp_fp16 and device.type == "cuda"
    gpu_prep = GpuPreprocessor(device, variant.channels_last) if variant.gpu_preprocess else None

    top1_parts, prob_parts = [], []
    with grad_ctx:
        for start in range(0, n, bs):
            chunk = frames[start:start + bs]
            if gpu_prep is not None:
                x = gpu_prep(chunk)
            else:
                x = torch.stack([_cpu_transform(img) for img in chunk]).to(device)
                if variant.channels_last:
                    x = x.contiguous(memory_format=torch.channels_last)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)

            probs = logits.float().softmax(dim=1)
            top1 = probs.argmax(dim=1)
            vprob = probs.index_select(1, vehicle_ids).sum(dim=1)

            if variant.sync_per_batch:
                # исходная версия: сразу забираем результат на CPU (синхронизация с GPU)
                top1_parts.append(top1.detach().cpu())
                prob_parts.append(vprob.detach().cpu())
            else:
                top1_parts.append(top1.detach())
                prob_parts.append(vprob.detach())

    top1_all = torch.cat(top1_parts).cpu()
    prob_all = torch.cat(prob_parts).cpu()
    return top1_all, prob_all
