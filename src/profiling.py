"""Профилирование узких мест (torch.profiler): исходная vs оптимизированная версия."""
import torch
from torch.profiler import ProfilerActivity, profile

from .model import load_model
from .pipelines import prepare_model, run_pipeline


def profile_variant(variant, frames, vehicle_ids, device, out_path, n_images=128):
    model = load_model(device)
    model, _ = prepare_model(model, variant, device)
    sample = frames[:n_images]
    if variant.gpu_preprocess and device.type == "cuda":
        sample = sample.pin_memory()
    run_pipeline(model, variant, sample, vehicle_ids, device)  # прогрев

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities) as prof:
        run_pipeline(model, variant, sample, vehicle_ids, device)
        if device.type == "cuda":
            torch.cuda.synchronize()

    table = None
    sort_keys = ["self_cuda_time_total", "self_device_time_total", "self_cpu_time_total"] \
        if device.type == "cuda" else ["self_cpu_time_total"]
    for key in sort_keys:
        try:
            table = prof.key_averages().table(sort_by=key, row_limit=15)
            break
        except Exception:
            continue
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Профиль: {variant.title}, кадров: {len(sample)}\n\n")
        f.write(table or "профиль недоступен")
    del model
    return out_path
