"""Графики по результатам замеров."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _bar(rows, field, ylabel, title, path, color):
    rows = [r for r in rows if r.get(field) is not None]
    if not rows:
        return
    labels = [r["variant"].split("_")[0].upper() for r in rows]
    values = [r[field] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, values, color=color)
    ax.bar_label(bars, fmt="%.1f", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def make_plots(rows, sweep_rows, out_dir):
    _bar(rows, "throughput_img_s", "кадров в секунду", "Пропускная способность по этапам оптимизации",
         f"{out_dir}/throughput.png", "#2a6fdb")
    _bar(rows, "peak_gpu_mem_mb", "МБ", "Пиковое потребление видеопамяти",
         f"{out_dir}/gpu_memory.png", "#e07b39")
    _bar(rows, "energy_mj_per_img", "мДж на кадр", "Энергия на обработку одного кадра",
         f"{out_dir}/energy.png", "#3a9d5d")

    if sweep_rows:
        bs = [r["batch_size"] for r in sweep_rows]
        fig, ax1 = plt.subplots(figsize=(8, 4.5))
        ax1.plot(bs, [r["throughput_img_s"] for r in sweep_rows], "o-", color="#2a6fdb",
                 label="кадров/с")
        ax1.set_xscale("log", base=2)
        ax1.set_xlabel("размер пакета (batch size)")
        ax1.set_ylabel("кадров в секунду", color="#2a6fdb")
        mem = [r["peak_gpu_mem_mb"] for r in sweep_rows]
        if all(m is not None for m in mem):
            ax2 = ax1.twinx()
            ax2.plot(bs, mem, "s--", color="#e07b39", label="видеопамять, МБ")
            ax2.set_ylabel("видеопамять, МБ", color="#e07b39")
        ax1.set_title("Влияние размера пакета на скорость и память")
        ax1.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{out_dir}/batch_sweep.png", dpi=130)
        plt.close(fig)
