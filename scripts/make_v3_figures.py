"""Route B figures: audit arc + v3 per-seed. Run where the v3 logs live."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = os.environ.get("V3_BASE", "artifacts/v3/cts")
os.makedirs("paper/figures", exist_ok=True)


def load_aupc(path):
    with open(path) as f:
        for line in f:
            if "AUPC=" in line:
                return float(line.split("AUPC=")[-1].split()[0])
    return None


def fig_audit_arc():
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    stages = ["F1\nstatic serving", "F2\nleakage", "F3\nisolation", "F3+\ngate"]
    deltas = [0.0, 0.070, -0.1875, 0.0]
    colors = ["#757575", "#b26a00", "#b71c1c", "#1b5e20"]
    bars = ax.bar(stages, deltas, color=colors, alpha=0.85, width=0.55)
    for b, d in zip(bars, deltas):
        label = f"+{d:.3f}" if d > 0 else f"{d:.3f}"
        ax.text(b.get_x() + b.get_width() / 2, d + (0.006 if d >= 0 else -0.014),
                label, ha="center", fontsize=9,
                color="#1b5e20" if d > 0 else ("#b71c1c" if d < 0 else "#333"))
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("naive $-$ frozen AUPC")
    ax.set_ylim(-0.25, 0.12)
    ax.set_title("Each protocol fix changes the conclusion (16-task, 8 seeds, Mistral-7B)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig("paper/figures/fig2_audit_arc.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("fig2_audit_arc done")


def fig_v3_seeds():
    arms = {"frozen": [], "naive": [], "egc": []}
    for s in range(8):
        for v in arms:
            a = load_aupc(f"{BASE}/{v}_s{s}_16.log")
            if a is not None:
                arms[v].append(a)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    colors = {"frozen": "#1f4e79", "naive": "#b71c1c", "egc": "#b26a00"}
    styles = {"frozen": "-", "naive": "--", "egc": ":"}
    for v, ys in arms.items():
        if not ys:
            continue
        x = np.arange(len(ys)) + {"frozen": 0.0, "naive": -0.07, "egc": 0.07}[v]
        ax.plot(x, ys, styles[v], marker="o", label=v, color=colors[v], ms=5)
    ax.set_xlabel("seed")
    ax.set_ylabel("AUPC (prequential)")
    ax.set_ylim(0.1, 0.65)
    ax.set_xticks(range(8))
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title("Full protocol isolation: updates harm (frozen deterministic)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig("paper/figures/fig3_v3_seeds.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("fig3_v3_seeds done")


if __name__ == "__main__":
    fig_audit_arc()
    fig_v3_seeds()
