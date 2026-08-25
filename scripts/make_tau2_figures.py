"""tau2 positive-control figures: (a) seed success table heatmap,
(b) successful task-2 trajectory as a tool-call chain."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import os
os.makedirs("paper/figures", exist_ok=True)


def fig_tau2_seeds():
    """Per-seed reward heatmap for tasks 0 and 2."""
    data = np.array([[1.0, 0.0, 0.0, 1.0, 0.0],
                     [1.0, 1.0, 0.0, 0.0, 1.0]])
    fig, ax = plt.subplots(figsize=(5.2, 1.9))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"seed {i}" for i in range(5)], fontsize=8)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["task 0\nexchange", "task 2\ncount+return"], fontsize=8)
    for i in range(2):
        for j in range(5):
            ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center",
                    fontsize=9, color="black")
    ax.set_title("official tau2 retail: full reward per seed "
                 "(DB 1.0 and NL-assertion 1.0)", fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig("paper/figures/fig7_tau2_seeds.png", dpi=300)
    plt.close(fig)


def fig_tau2_trajectory():
    """Successful task-2 trajectory as an annotated tool-call chain."""
    steps = [
        ("user", "asks: how many t-shirt options? return cleaner,\nheadphones, smartwatch", None),
        ("tool", "find_user_id_by_name_zip\n(Yusuf Rossi, 19122)", "yusuf_rossi_9620"),
        ("tool", "list_all_product_types()", "T-Shirt: 9523456873"),
        ("tool", "get_product_details\n(9523456873)", "12 variants, 10 available"),
        ("msg", "there are 10 t-shirt options available", None),
        ("tool", "get_user_details\n(yusuf_rossi_9620)", "orders: 5 ids"),
        ("tool", "get_order_details\n(#W2378156)", "headphones, vacuum cleaner,\nkeyboard, thermostat, watch"),
        ("tool", "return_delivered_order_items\n(3 item ids, credit_card_9513926)", "return requested"),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.axis("off")
    y = 1.0
    for i, (kind, label, result) in enumerate(steps):
        color = {"user": "#1565c0", "tool": "#2e7d32", "msg": "#6a1b9a"}[kind]
        ax.text(0.02, y, label, fontsize=8.5, color=color,
                va="center", family="monospace",
                bbox=dict(boxstyle="round,pad=0.35", fc="white",
                          ec=color, lw=1.0))
        if result:
            ax.text(0.56, y, f"-> {result}", fontsize=8, va="center",
                    color="#37474f", family="monospace")
        y -= 0.13
    ax.set_title("successful task-2 trajectory (official tau2, local 14B): "
                 "reward 1.0", fontsize=9.5)
    fig.tight_layout()
    fig.savefig("paper/figures/fig8_tau2_trajectory.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    fig_tau2_seeds()
    fig_tau2_trajectory()
    print("figures written")
