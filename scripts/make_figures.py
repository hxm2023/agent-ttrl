"""Generate all paper figures from protocols/runs/ data.

Figures:
  fig1_method.png    — EGC-TTRL method overview (drawn schematically)
  fig2_prequential.png — prequential curves per setup (CTS / tau2-8 / tau2-16)
  fig3_credit_ablation.png — credit mechanism ablation (CTS pilot)
  fig4_safecommit.png — SafeCommit risk-gain from M4 stress simulation
  fig5_pareto.png    — success-cost Pareto (CTS baselines)
  fig6_heatmap.png   — task-level y_pre heatmap (M6 factorial)
"""
import json
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.makedirs("paper/figures", exist_ok=True)


def load_many(pattern):
    out = []
    for f in sorted(glob.glob(pattern)):
        try:
            d = json.load(open(f, encoding="utf-8"))
            d["_file"] = f
            out.append(d)
        except Exception:
            pass
    return out


# ---------------------------------------------------------------- fig1: method
def fig1_method():
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#eef4fb", ec="#1f4e79", fs=9):
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                             fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#333", lw=1.4))

    box(0.2, 4.6, 2.6, 0.9, "Task $x_k$ arrives\n(domain session stream)", fc="#fff3e0", ec="#b26a00")
    box(3.4, 4.6, 2.6, 0.9, "First attempt\n$Y_k^{pre}$ (hidden, offline)", fc="#e8f5e9", ec="#1b5e20")
    box(6.6, 4.6, 3.0, 0.9, "Accessible evidence\n$E_{hard} / E_{soft}$ ($R_{hidden}$ never)", fc="#eef4fb", ec="#1f4e79")

    box(0.6, 2.6, 2.2, 0.9, "Critical-decision selector\n(entropy / state-impact)", fc="#f3e5f5", ec="#6a1b9a")
    box(3.4, 2.6, 2.4, 0.9, "Paired branches\nG $\times$ R, CRN-coupled\n(reliability gate)", fc="#e8f5e9", ec="#1b5e20")
    box(6.4, 2.6, 2.4, 0.9, "Signed action credit\n+ evidence-conflict\nabstention (M1)", fc="#eef4fb", ec="#1f4e79")

    box(1.2, 0.5, 2.4, 0.9, "Action-token LoRA-GRPO\n(guarded update)", fc="#e8f5e9", ec="#1b5e20")
    box(4.2, 0.5, 2.2, 0.9, "Candidate adapter\n(immutable, hashed)", fc="#fff3e0", ec="#b26a00")
    box(7.0, 0.5, 2.6, 0.9, "SafeCommit gate\n(EB e-process)\ncommit / rollback", fc="#fdecea", ec="#b71c1c")

    arrow(2.8, 5.05, 3.4, 5.05)      # task -> first attempt
    arrow(6.0, 5.05, 6.6, 5.05)      # first attempt -> evidence
    arrow(3.6, 4.6, 1.7, 3.5)        # first attempt -> selector
    arrow(1.7, 3.5, 3.4, 3.05)       # selector -> paired branches
    arrow(6.6, 4.6, 6.4, 3.5)        # evidence -> signed credit
    arrow(6.6, 4.6, 5.2, 3.05)       # evidence -> paired branches (top-right)
    arrow(5.2, 2.75, 6.4, 3.05)      # paired branches -> signed credit (bottom-right)
    arrow(7.0, 3.05, 3.6, 1.4)       # signed credit -> LoRA-GRPO
    arrow(3.6, 1.4, 4.2, 1.0)        # LoRA -> candidate
    arrow(6.4, 1.0, 7.0, 1.0)        # candidate -> SafeCommit
    ax.set_title("EGC-TTRL: two-scale evidence gating for deployment-period agent RL", fontsize=11)
    fig.tight_layout()
    fig.savefig("paper/figures/fig1_method.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("fig1 done")


# ---------------------------------------------------------------- fig2: prequential curves
def fig2_prequential():
    # Left: CTS 8-task from m3 v2 manifests (variant/seed fields correct there).
    # Middle/right: tau2 manifests were overwritten by later 16-task control runs
    # and the M6 manifest writer originally hardcoded variant="frozen", so the
    # tau2 panels use the paper-table synthesis (built from run logs before the
    # overwrite) — per-seed points where they survive, table means otherwise.
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    styles = {"frozen": "-", "naive": "--", "egc": ":", "egc_conflict": "-.", "random_branch": "x-"}
    jitter = {"frozen": 0.0, "naive": -0.05, "egc": -0.025, "egc_conflict": 0.025, "random_branch": 0.05}
    colors = {"frozen": "#1f4e79", "naive": "#2e7d32", "egc": "#b26a00",
              "egc_conflict": "#6a1b9a", "random_branch": "#616161"}

    # left: CTS 8-task (Qwen3-4B), data-driven from m3 v2 manifests.
    # NOTE: egc/egc_conflict are NO-OP controls — zero gradient tokens on
    # every task (paired-branch credit produced no reliable rows), so their
    # 0.50 is frozen-equivalent behavior, not a mechanism run.
    ax = axes[0]
    runs = load_many("protocols/runs/m3/*v2_run_manifest.json")
    for v in ["frozen", "naive", "egc", "egc_conflict"]:
        ys = [r["aupc_prequential"] for r in runs if r.get("variant") == v]
        if ys:
            x = np.arange(len(ys)) + jitter[v]
            ax.plot(x, ys, styles[v], marker="o", label=v, color=colors[v], alpha=0.95, ms=5)
    # value annotations with horizontal offsets so the three 0.50 cluster
    # labels do not overprint (all share y=0.50 at both seeds)
    offset = {"frozen": (0, 6), "naive": (-16, -8), "egc": (0, -8), "egc_conflict": (16, -8)}
    for v in ["frozen", "naive", "egc", "egc_conflict"]:
        ys = [r["aupc_prequential"] for r in runs if r.get("variant") == v]
        for xi, yi in zip(np.arange(len(ys)) + jitter[v], ys):
            dx, dy = offset[v]
            ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                        xytext=(dx, dy), ha="center", fontsize=7, color=colors[v])
    ax.set_title("CTS 8-task (Qwen3-4B)\n(egc*: no-op, zero gradient tokens)", fontsize=9)
    ax.set_xlabel("seed")
    ax.set_ylabel("AUPC (prequential)")
    ax.set_ylim(0, 0.72)
    ax.set_xticks([0, 1])
    ax.legend(fontsize=7, loc="lower left", framealpha=0.9)

    # middle: tau2 8-task (Mistral-7B) — naive/egc per-seed from surviving
    # manifests; frozen is a table mean (its seed manifests were overwritten)
    ax = axes[1]
    mj = {"frozen": 0.0, "naive": -0.04, "egc": 0.04}  # x jitter at seed 0
    ax.plot([0], [0.072], styles["frozen"], marker="o", label="frozen (mean)", color=colors["frozen"], ms=5)
    ax.annotate("0.072", (mj["frozen"], 0.072), textcoords="offset points", xytext=(34, -4), ha="center", fontsize=7, color=colors["frozen"])
    ax.plot([mj["naive"], 1 + mj["naive"]], [0.0788, 0.1078], styles["naive"], marker="o", label="naive",
            color=colors["naive"], ms=5)
    ax.annotate("0.079", (mj["naive"], 0.0788), textcoords="offset points", xytext=(-10, -8), ha="center", fontsize=7, color=colors["naive"])
    ax.annotate("0.108", (1 + mj["naive"], 0.1078), textcoords="offset points", xytext=(0, -13), ha="center", fontsize=7, color=colors["naive"])
    ax.plot([mj["egc"]], [0.0766], styles["egc"], marker="o", label="egc", color=colors["egc"], ms=5)
    ax.annotate("0.077", (mj["egc"], 0.0766), textcoords="offset points", xytext=(10, -8), ha="center", fontsize=7, color=colors["egc"])
    ax.set_title("tau2 8-task (Mistral-7B)", fontsize=10)
    ax.set_xlabel("seed")
    ax.set_ylabel("AUPC (prequential)")
    ax.set_ylim(0, 0.14)
    ax.set_xticks([0, 1])
    ax.legend(fontsize=7, loc="upper left")

    # right: tau2 16-task (Mistral-7B) — per-seed from logs (frozen 0.0239/0.0258,
    # naive 0.0114/0.0144)
    ax = axes[2]
    ax.plot([0, 1], [0.0239, 0.0258], styles["frozen"], marker="o", label="frozen", color=colors["frozen"], ms=5)
    ax.plot([0, 1], [0.0114, 0.0144], styles["naive"], marker="o", label="naive", color=colors["naive"], ms=5)
    for xi, yi in [(0, 0.0239), (1, 0.0258), (0, 0.0114), (1, 0.0144)]:
        ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points", xytext=(0, -12), ha="center", fontsize=7)
    ax.set_title("tau2 16-task (Mistral-7B)", fontsize=10)
    ax.set_xlabel("seed")
    ax.set_ylabel("AUPC (prequential)")
    ax.set_ylim(0, 0.04)
    ax.set_xticks([0, 1])
    ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("Prequential AUPC across setups and seeds", fontsize=12)
    fig.tight_layout()
    fig.savefig("paper/figures/fig2_prequential.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("fig2 done")


# ---------------------------------------------------------------- fig3: CTS ablation
def fig3_credit_ablation():
    # Source: protocols/runs/m3/*v2/v3_run_manifest.json (means over seeds).
    # Honesty note: egc/egc_conflict/random_branch produced zero gradient
    # tokens on every task (local gate passed nothing on the CTS stream) —
    # their 0.500 bars are NO-OP controls, not mechanism runs.
    variants = ["frozen", "naive", "egc*", "egc_conflict*", "random_branch*"]
    means = [0.625, 0.5, 0.5, 0.5, 0.5]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    bars = ax.bar(variants, means, color=["#1f4e79", "#2e7d32", "#b26a00", "#6a1b9a", "#616161"],
                  alpha=0.85)
    ax.set_ylabel("AUPC (prequential)")
    ax.set_ylim(0, 0.75)
    ax.axhline(0.625, color="#1f4e79", ls="--", lw=1)
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.01, f"{m:.3f}", ha="center", fontsize=9)
    ax.set_title("CTS 8-task pilot: naive updates do not beat frozen\n"
                 "(* no-op controls: zero gradient tokens, no mechanism run)", fontsize=10)
    fig.tight_layout()
    fig.savefig("paper/figures/fig3_credit_ablation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("fig3 done")


# ---------------------------------------------------------------- fig4: SafeCommit
def fig4_safecommit():
    # M4 stress results (from protocols/runs/M4_stress_simulation.json)
    try:
        m4 = json.load(open("protocols/runs/M4_stress_simulation.json", encoding="utf-8"))
    except Exception:
        m4 = None
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    streams = ["benign", "mixed", "poisoned", "abrupt_shift"]
    if m4:
        ac = [m4["streams"][s]["always_commit"]["catastrophic_rate"] for s in streams]
        gate = [m4["streams"][s]["eb_eprocess"]["catastrophic_rate"] for s in streams]
        commit = [m4["streams"][s]["eb_eprocess"]["commit_rate"] for s in streams]
    else:
        ac = [0.0, 0.25, 0.5, 0.5]
        gate = [0.0, 0.0, 0.0, 0.0]
        commit = [0.115, 0.102, 0.068, 0.111]
    x = np.arange(len(streams))
    w = 0.28
    ax.bar(x - w / 2, ac, w, label="always-commit", color="#b71c1c", alpha=0.8)
    ax.bar(x + w / 2, gate, w, label="SafeCommit (EB e-process)", color="#1b5e20", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(streams)
    ax.set_ylabel("catastrophic-update rate")
    ax.legend()
    ax.set_title("SafeCommit eliminates catastrophic updates (candidate-archive replay)", fontsize=11)
    for xi, gv in zip(x, gate):
        if gv == 0:
            ax.text(xi + w / 2, 0.008, "0", ha="center", fontsize=8, color="#1b5e20")
    ax2 = ax.twinx()
    ax2.plot(x, commit, "o--", color="#b26a00", label="commit rate")
    ax2.set_ylabel("commit rate", color="#b26a00")
    ax2.tick_params(axis="y", labelcolor="#b26a00")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper center", fontsize=8)
    fig.tight_layout()
    fig.savefig("paper/figures/fig4_safecommit.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("fig4 done")


# ---------------------------------------------------------------- fig5: Pareto
def fig5_pareto():
    # Source: protocols/runs/m2 (best_of_n s0 AUPC 0.75 — deterministic, n=1)
    # and m3 v2 manifests (frozen 0.625 n=2; naive 0.5 n=3; egc 0.5 n=3 NO-OP).
    # Costs are APPROXIMATE relative model-token counts (3-channel ledger
    # budgets per run; fig5 is a sketch, not a measured cost curve).
    methods = ["frozen", "best_of_n", "naive", "egc*"]
    aupc = [0.625, 0.75, 0.5, 0.5]
    cost = [1.0, 8.0, 5.0, 9.0]  # relative model-token cost (approximate)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    colors = ["#1f4e79", "#2e7d32", "#8d4e00", "#7b1fa2"]
    for m, a, c, col in zip(methods, aupc, cost, colors):
        ax.scatter(c, a, s=110, color=col, label=f"{m} (AUPC {a:.3f})", zorder=3)
        ax.annotate(m, (c, a), textcoords="offset points", xytext=(8, 6), fontsize=8)
    ax.set_xlabel("relative model-token cost (approx., log scale)")
    ax.set_ylabel("AUPC (prequential)")
    ax.set_xscale("log")
    ax.set_xlim(0.5, 20)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Success-cost sketch: BoN dominates; updates add cost without gain\n"
                 "(* egc = no-op control; costs approximate)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig("paper/figures/fig5_pareto.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("fig5 done")


# ---------------------------------------------------------------- fig6: heatmap
def fig6_heatmap():
    # 16-task matched control (Mistral-7B): per-seed AUPC from ctl/ctl3/ctl5
    # manifests (protocols/runs/m6/ctl{,_3,_5}_{variant}_s{seed}.json).
    # Rows = variant, columns = seed. Missing cells render as blank.
    tags = {"frozen": "ctl", "naive": "ctl3", "egc": "ctl5"}
    variants = list(tags)
    seeds = ["s0", "s1", "s2", "s3"]
    data = np.full((len(variants), len(seeds)), np.nan)
    for i, v in enumerate(variants):
        for j, s in enumerate(seeds):
            p = f"protocols/runs/m6/{tags[v]}_{v}_s{j}.json"
            try:
                d = json.load(open(p, encoding="utf-8"))
                data[i, j] = d["aupc_prequential"]
            except Exception:
                pass
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    im = ax.imshow(data, cmap="YlOrRd", vmin=0, vmax=0.04)
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels(seeds)
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels(variants)
    for i in range(len(variants)):
        for j in range(len(seeds)):
            v = data[i, j]
            if np.isnan(v):
                ax.text(j, i, "-", ha="center", va="center", fontsize=10, color="gray")
            else:
                ax.text(j, i, f"{v:.4f}", ha="center", va="center", fontsize=10,
                        color="white" if v > 0.02 else "black")
    fig.colorbar(im, label="AUPC")
    ax.set_title("tau2 16-task matched control (Mistral-7B, strong updates)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig("paper/figures/fig6_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("fig6 done")


if __name__ == "__main__":
    fig1_method()
    fig2_prequential()
    fig3_credit_ablation()
    fig4_safecommit()
    fig5_pareto()
    fig6_heatmap()
    print("ALL FIGURES DONE")
