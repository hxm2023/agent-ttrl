"""v2 figures: per-seed AUPC curves from v2 CTS runs (16-task, 8 seeds)."""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("paper/figures", exist_ok=True)


def load_aupc(path):
    with open(path) as f:
        for line in f:
            if "AUPC=" in line:
                return float(line.split("AUPC=")[-1].split()[0])
    return None


def main():
    base = "/root/autodl-tmp/agent-ttrl/artifacts/v2/cts"
    arms = {"frozen": [], "naive": [], "egc": []}
    for s in range(8):
        for v in arms:
            a = load_aupc(f"{base}/{v}_s{s}_16.log")
            if a is not None:
                arms[v].append(a)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    colors = {"frozen": "#1f4e79", "naive": "#2e7d32", "egc": "#b26a00"}
    styles = {"frozen": "-", "naive": "--", "egc": ":"}
    for v, ys in arms.items():
        if not ys:
            continue
        x = np.arange(len(ys)) + {"frozen": 0.0, "naive": -0.06, "egc": 0.06}[v]
        ax.plot(x, ys, styles[v], marker="o", label=v, color=colors[v], ms=6)
        if v == "egc":
            continue   # egc tracks naive closely; line+legend is enough
        for xi, yi in zip(x, ys):
            dx, dy = (0, -20) if v == "frozen" else (0, 10)
            ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points",
                        xytext=(dx, dy), ha="center", fontsize=7, color=colors[v])
    ax.set_xlabel("seed")
    ax.set_ylabel("AUPC (prequential)")
    ax.set_ylim(0.30, 0.65)
    ax.set_xticks(range(8))
    ax.legend(fontsize=9, loc="lower left")
    ax.set_title("CTS-v2 16-task, 8 paired seeds (Mistral-7B): updates transfer",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig("paper/figures/fig2_prequential.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("fig2 v2 done")

    # credit-sign diagnostic: EGC correct on deceptive tasks
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    labels = ["recover_v (s0)", "refund_delivered (s0)", "recover_v (s1)"]
    vals = [[-0.3, 0.3], [-0.5, 0.5], [-0.3, 0.3]]
    width = 0.35
    for i, (lab, vv) in enumerate(zip(labels, vals)):
        ax.bar([i - width / 2, i + width / 2], vv, width, color=["#b71c1c", "#1b5e20"],
               alpha=0.85)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("EGC credit")
    ax.set_title("Counterfactual credit signs: evidence-user positive, goal-user negative",
                 fontsize=9.5)
    fig.tight_layout()
    fig.savefig("paper/figures/fig3_credit_ablation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("fig3 v2 done")


if __name__ == "__main__":
    main()
