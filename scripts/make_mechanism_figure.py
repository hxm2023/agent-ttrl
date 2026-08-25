"""F3 mechanism figure: per-template first-attempt AUPC, frozen vs naive,
both backbones (from artifacts/mechanism/mechanism.json)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TPL_ORDER = ["F1_cancel", "F1_exchange", "F1_refund", "F1_refund_delivered",
             "F1_refund_v", "F3_recover", "F3_recover_v"]

d = json.load(open("artifacts/mechanism/mechanism.json"))
fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.9), sharey=True)

for ax, (name, b) in zip(axes, d["backbones"].items()):
    tpls = [t for t in TPL_ORDER if t in b["frozen_tpl"] or t in b["naive_tpl"]]
    f = [b["frozen_tpl"].get(t, 0.0) for t in tpls]
    n = [b["naive_tpl"].get(t, 0.0) for t in tpls]
    x = np.arange(len(tpls))
    w = 0.36
    ax.bar(x - w/2, f, w, label="frozen", color="#1b5e20", alpha=0.85)
    ax.bar(x + w/2, n, w, label="naive", color="#b71c1c", alpha=0.85)
    for xi, fi, ni in zip(x, f, n):
        ax.text(xi - w/2, fi + 0.02, f"{fi:.0%}", ha="center", fontsize=7)
        ax.text(xi + w/2, ni + 0.02, f"{ni:.0%}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("F1_", "").replace("F3_", "").replace("_", "\n")
                        for t in tpls], fontsize=7)
    ax.set_ylim(0, 1.18)
    ax.set_title(f"{'Mistral-7B' if name == 'mistral' else 'Qwen2.5-7B'}: "
                 f"frozen {b['frozen_aupc']:.3f} vs naive "
                 f"{np.mean(b['naive_aupc']):.3f}", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel("first-attempt AUPC per template", fontsize=8)

fig.suptitle("F3 mechanism: updates degrade the templates the frozen policy solved",
             fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("paper/figures/fig9_mechanism.png", dpi=300)
print("written paper/figures/fig9_mechanism.png")
