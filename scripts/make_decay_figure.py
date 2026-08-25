"""Cumulative-commit degradation figure: first-attempt success rate by
task window for the loose-gate streams (Mistral 64-task, Qwen 32-task)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Mistral 64-task (protocols/runs/v3/cts/naive_s0x_64.log)
# Qwen 32-task (protocols/runs/v3q/loosegate_32task_qwen.txt)
import re


def window_rates(path, n, wsize):
    recs = []
    for line in open(path, encoding="utf-8", errors="ignore"):
        m = re.search(r"t(\d+) \S+: y_pre=(\d+\.?\d*)", line)
        if m:
            recs.append((int(m.group(1)), float(m.group(2))))
    rates = []
    for lo in range(0, n, wsize):
        win = [r for r in recs if lo <= r[0] < lo + wsize]
        rates.append(sum(1 for _, y in win if y == 1.0) / max(1, len(win)))
    return rates


mistral = window_rates("protocols/runs/v3/cts/naive_s0x_64.log", 64, 8)
qwen = window_rates("protocols/runs/v3q/loosegate_32task_qwen.txt", 32, 8)

fig, ax = plt.subplots(figsize=(4.6, 2.6))
xs_m = np.arange(len(mistral))
xs_q = np.arange(len(qwen))
ax.plot(xs_m, mistral, "o-", color="#b71c1c", label="Mistral-7B (64 tasks)")
ax.plot(xs_q, qwen, "s--", color="#1b5e20", label="Qwen2.5-7B (32 tasks)")
ax.axhline(0.875, color="#555", ls=":", lw=1)
ax.text(0.1, 0.90, "frozen (deterministic)", fontsize=7, color="#555")
ax.set_xlabel("task window (8 tasks each)")
ax.set_ylabel("first-attempt success rate")
ax.set_ylim(-0.05, 1.05)
ax.set_xticks(range(max(len(mistral), len(qwen))))
ax.set_xticklabels([str(i * 8) for i in range(max(len(mistral), len(qwen)))],
                   fontsize=7)
ax.legend(fontsize=7, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("cumulative loose-gate updates degrade the policy on both backbones",
             fontsize=8.5)
fig.tight_layout()
fig.savefig("paper/figures/fig10_decay.png", dpi=300)
print("written paper/figures/fig10_decay.png")
print("mistral rates:", mistral)
print("qwen rates:", qwen)
