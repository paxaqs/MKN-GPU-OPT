"""Главный скрипт: прогоняет все варианты, сохраняет таблицы, графики и сводку.

Запуск (Google Colab с GPU):   python -m src.run_all
Быстрая проверка на CPU (CI):   python -m src.run_all --quick --device cpu
"""
import argparse
import csv
import json
import os
import platform
from dataclasses import replace

import torch

from .benchmark import measure_variant
from .data import load_frames
from .model import vehicle_class_ids
from .pipelines import ALL_VARIANTS, BASELINE, V5
from .plots import make_plots
from .profiling import profile_variant

TABLE_COLUMNS = [
    ("title", "Вариант"),
    ("throughput_img_s", "Кадров/с"),
    ("speedup", "Ускорение, раз"),
    ("latency_ms_per_img", "мс/кадр"),
    ("peak_gpu_mem_mb", "Видеопамять, МБ"),
    ("avg_gpu_util_pct", "Загрузка GPU, %"),
    ("energy_mj_per_img", "мДж/кадр"),
    ("accuracy_pct", "Точность, %"),
    ("decision_agreement_pct", "Совпадение с V0, %"),
    ("time_std_pct", "Разброс времени, %"),
]


def parse_args():
    p = argparse.ArgumentParser(description="Оптимизация ИИ-конвейера на GPU (ООО «МКН»)")
    p.add_argument("--n-images", type=int, default=2048)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--skip-compile", action="store_true", help="не запускать вариант V6")
    p.add_argument("--quick", action="store_true", help="короткий прогон для проверки кода")
    p.add_argument("--out", default="results")
    return p.parse_args()


def env_info(device):
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": device.type,
        "cpu": platform.processor() or platform.machine(),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        info.update({
            "gpu": props.name,
            "gpu_memory_gb": round(props.total_memory / 2**30, 1),
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        })
    return info


def md_table(rows, columns):
    lines = ["| " + " | ".join(h for _, h in columns) + " |",
             "|" + "---|" * len(columns)]
    for r in rows:
        cells = []
        for key, _ in columns:
            v = r.get(key)
            cells.append("—" if v is None else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_csv(rows, path):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    args = parse_args()
    if args.quick:
        args.n_images, args.repeats = 64, 1
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else ("cpu" if args.device == "auto" else args.device))
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    env = env_info(device)
    print("Среда:", json.dumps(env, ensure_ascii=False))
    if device.type != "cuda":
        print("ВНИМАНИЕ: GPU не найден — запуск на CPU (только для проверки кода).")

    frames, labels = load_frames(args.n_images)
    vehicle_ids = torch.tensor(vehicle_class_ids(), device=device)

    variants = list(ALL_VARIANTS)
    if args.skip_compile or device.type != "cuda":
        variants = [v for v in variants if not v.compile]

    rows, reference = [], None
    for v in variants:
        print(f"\n>>> {v.title}")
        res, out = measure_variant(v, frames, labels, vehicle_ids, device, args.repeats, reference)
        if "error" in res:
            print("    ошибка:", res["error"])
            rows.append(res)
            continue
        if reference is None:
            reference = out
            res.update(top1_agreement_pct=100.0, decision_agreement_pct=100.0, max_prob_diff=0.0)
        res["speedup"] = round(res["throughput_img_s"] / rows[0]["throughput_img_s"], 2) if rows else 1.0
        rows.append(res)
        print("   ", {k: res.get(k) for k, _ in TABLE_COLUMNS if k != "title"})

    # Влияние размера пакета (конфигурация V5, без компиляции — чтобы не перекомпилировать)
    sweep_sizes = [8, 16, 32] if args.quick else [16, 32, 64, 128, 256]
    sweep_rows = []
    for bs in sweep_sizes:
        v = replace(V5, key=f"sweep_bs{bs}", title=f"V5, batch={bs}", batch_size=bs)
        print(f"\n>>> {v.title}")
        res, _ = measure_variant(v, frames, labels, vehicle_ids, device,
                                 max(1, args.repeats - 1), reference)
        sweep_rows.append(res)
        print("    кадров/с:", res["throughput_img_s"], "память, МБ:", res["peak_gpu_mem_mb"])

    # Профилирование узких мест
    n_prof = 16 if args.quick else 128
    print("\n>>> Профилирование")
    profile_variant(BASELINE, frames, vehicle_ids, device, f"{args.out}/profile_v0_baseline.txt", n_prof)
    profile_variant(V5, frames, vehicle_ids, device, f"{args.out}/profile_v5_optimized.txt", n_prof)

    # Сохранение
    write_csv(rows, f"{args.out}/results.csv")
    write_csv(sweep_rows, f"{args.out}/batch_sweep.csv")
    with open(f"{args.out}/env.json", "w", encoding="utf-8") as f:
        json.dump(env, f, ensure_ascii=False, indent=2)
    make_plots([r for r in rows if "error" not in r], sweep_rows, args.out)

    ok = [r for r in rows if "error" not in r]
    base, best = ok[0], max(ok, key=lambda r: r["throughput_img_s"])
    summary = [
        "# Результаты оптимизации ИИ-конвейера на GPU",
        "",
        "Среда: " + ", ".join(f"{k}={v}" for k, v in env.items()),
        f"Кадров: {args.n_images}, повторов замера: {args.repeats}",
        "",
        "## Этапы оптимизации",
        "",
        md_table(rows, TABLE_COLUMNS),
        "",
        "## Размер пакета (конфигурация V5)",
        "",
        md_table(sweep_rows, [("batch_size", "batch"), ("throughput_img_s", "Кадров/с"),
                              ("peak_gpu_mem_mb", "Видеопамять, МБ"),
                              ("energy_mj_per_img", "мДж/кадр")]),
        "",
        "## Итог",
        "",
        f"* Лучший вариант: {best['title']} — {best['throughput_img_s']} кадров/с "
        f"против {base['throughput_img_s']} у исходной версии (ускорение в {best['speedup']} раза).",
        f"* Совпадение решений «транспорт / нет» с исходной версией: {best.get('decision_agreement_pct')} %, "
        f"точность: {best['accuracy_pct']} % (исходная: {base['accuracy_pct']} %).",
        f"* Время прогрева/компиляции лучшего варианта: {best['warmup_s']} с.",
    ]
    if base.get("energy_mj_per_img") and best.get("energy_mj_per_img"):
        summary.append(f"* Энергия на кадр: {base['energy_mj_per_img']} → {best['energy_mj_per_img']} мДж.")
    errors = [r for r in rows if "error" in r]
    for r in errors:
        summary.append(f"* {r['title']}: не выполнен ({r['error'][:200]})")
    text = "\n".join(summary)
    with open(f"{args.out}/summary.md", "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print("\n" + "=" * 70 + "\n" + text)


if __name__ == "__main__":
    main()
