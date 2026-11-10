"""Замер производительности, ресурсов, качества и стабильности одного варианта."""
import shutil
import statistics
import subprocess
import threading
import time

import torch

from .model import load_model
from .pipelines import Variant, prepare_model, run_pipeline


def sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


class GpuSampler:
    """Фоновый опрос nvidia-smi: потребляемая мощность (Вт) и загрузка GPU (%)."""

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.available = shutil.which("nvidia-smi") is not None
        self.power, self.util = [], []
        self._stop = threading.Event()
        self._thread = None

    def _loop(self):
        cmd = ["nvidia-smi", "--query-gpu=power.draw,utilization.gpu",
               "--format=csv,noheader,nounits", "-i", "0"]
        while not self._stop.is_set():
            try:
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=2).stdout
                p, u = [s.strip() for s in out.strip().split(",")[:2]]
                self.power.append(float(p))
                self.util.append(float(u))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        if self.available:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    @property
    def mean_power(self):
        return statistics.mean(self.power) if self.power else None

    @property
    def mean_util(self):
        return statistics.mean(self.util) if self.util else None


def measure_variant(variant: Variant, frames, labels, vehicle_ids, device, repeats: int,
                    reference=None):
    """Возвращает словарь метрик и предсказания (top1, vehicle_prob)."""
    model = load_model(device)
    model, compile_error = prepare_model(model, variant, device)

    pinned = frames
    if variant.gpu_preprocess and device.type == "cuda":
        pinned = frames.pin_memory()   # быстрая асинхронная передача на GPU

    # Прогрев (для torch.compile здесь же происходит компиляция)
    warm_n = min(len(frames), max(variant.batch_size * 2, 8))
    sync(device)
    t0 = time.perf_counter()
    try:
        run_pipeline(model, variant, pinned[:warm_n], vehicle_ids, device)
    except Exception as exc:
        if variant.compile:
            compile_error = repr(exc)
            return {"variant": variant.key, "title": variant.title, "error": compile_error}, None
        raise
    sync(device)
    warmup_s = time.perf_counter() - t0

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    times, outputs = [], []
    with GpuSampler() as sampler:
        for _ in range(repeats):
            sync(device)
            t = time.perf_counter()
            top1, vprob = run_pipeline(model, variant, pinned, vehicle_ids, device)
            sync(device)
            times.append(time.perf_counter() - t)
            outputs.append((top1, vprob))

    n = len(frames)
    mean_t = statistics.mean(times)
    std_t = statistics.pstdev(times) if len(times) > 1 else 0.0
    top1, vprob = outputs[-1]
    is_vehicle = (vprob >= 0.5).long()

    # Стабильность: совпадают ли результаты между повторами, нет ли NaN
    repeat_consistent = all(torch.equal(o[0], outputs[0][0]) for o in outputs)
    has_nan = bool(torch.isnan(vprob).any())

    res = {
        "variant": variant.key,
        "title": variant.title,
        "batch_size": min(variant.batch_size, n),
        "n_images": n,
        "repeats": repeats,
        "time_s": round(mean_t, 4),
        "time_std_pct": round(100 * std_t / mean_t, 2),
        "throughput_img_s": round(n / mean_t, 1),
        "latency_ms_per_img": round(1000 * mean_t / n, 3),
        "warmup_s": round(warmup_s, 2),
        "peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1)
        if device.type == "cuda" else None,
        "avg_power_w": round(sampler.mean_power, 1) if sampler.mean_power else None,
        "avg_gpu_util_pct": round(sampler.mean_util, 1) if sampler.mean_util is not None else None,
        "energy_mj_per_img": round(1000 * sampler.mean_power * mean_t / n, 2)
        if sampler.mean_power else None,
        "accuracy_pct": round(100 * float((is_vehicle == labels).float().mean()), 2),
        "repeat_consistent": repeat_consistent,
        "has_nan": has_nan,
        "compile_error": compile_error,
    }
    if reference is not None:
        ref_top1, ref_vprob = reference
        ref_vehicle = (ref_vprob >= 0.5).long()
        res["top1_agreement_pct"] = round(100 * float((top1 == ref_top1).float().mean()), 2)
        res["decision_agreement_pct"] = round(100 * float((is_vehicle == ref_vehicle).float().mean()), 2)
        res["max_prob_diff"] = round(float((vprob - ref_vprob).abs().max()), 5)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return res, (top1, vprob)
