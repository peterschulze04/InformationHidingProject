"""Plot the FLD ensemble baseline benchmark results."""

import pathlib
import matplotlib.pyplot as plt

OUT_DIR = pathlib.Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

# (label, time_s, mem_mb, data_mb)  -- replace with your own measurements
RESULTS = [
    ("D256 L50\nN=300", 0.05, 0.1, 1.2),
    ("D256 L50\nN=600", 0.09, 2.5, 2.5),
    ("D1024 L50\nN=300", 0.20, 4.9, 4.9),
    ("D1024 L50\nN=600", 0.34, 12.4, 9.8),
    ("D1024 L100\nN=600", 0.63, 13.0, 9.8),
    ("D1024 autoL\nN=600", 1.26, 13.8, 9.8),
    ("D1024 auto\nd_sub+L", 33.14, 53.4, 9.8),
]

# cProfile self-time shares (N=600 D=1024 L=100)
HOTSPOTS = [
    ("np.linalg.solve", 0.224),
    ("_fast_fancy_indexing", 0.106),
    ("fld.fit (rest)", 0.082),
    ("_find_threshold", 0.071),
    ("other", 0.086),
]


def plot_time():
    labels = [r[0] for r in RESULTS]
    times = [r[1] for r in RESULTS]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(labels, times, color="#3b6ea5")
    ax.set_yscale("log")
    ax.set_ylabel("training time (s, log scale)")
    ax.set_title("Legacy FldEnsembleTrainer — training time per config")
    for b, t in zip(bars, times):
        ax.text(b.get_x() + b.get_width() / 2, t, f"{t:g}",
                ha="center", va="bottom", fontsize=8)
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "time_per_config.png", dpi=150)


def plot_memory():
    labels = [r[0] for r in RESULTS]
    mem = [r[2] for r in RESULTS]
    data = [r[3] for r in RESULTS]
    x = range(len(labels))
    width = 0.4
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar([i - width / 2 for i in x], data, width, label="input data", color="#9bbf85")
    ax.bar([i + width / 2 for i in x], mem, width, label="peak overhead", color="#c0504d")
    ax.set_ylabel("MB")
    ax.set_title("Peak memory overhead vs. input data size")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "memory_overhead.png", dpi=150)


def plot_hotspots():
    labels = [h[0] for h in HOTSPOTS]
    vals = [h[1] for h in HOTSPOTS]
    total = sum(vals)
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(labels[::-1], vals[::-1], color="#3b6ea5")
    ax.set_xlabel("self time (s)")
    ax.set_title("Where the time goes (cProfile, N=600 D=1024 L=100)")
    for b, v in zip(bars, vals[::-1]):
        ax.text(v, b.get_y() + b.get_height() / 2,
                f"  {v:.3f}s ({100 * v / total:.0f}%)", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "hotspots.png", dpi=150)


if __name__ == "__main__":
    plot_time()
    plot_memory()
    plot_hotspots()
    print(f"Figures written to {OUT_DIR}")
    plt.show()
