"""Protocol figure: the v3 audit-grade pipeline with evidence tiers.
Shows what enters the update loop (E_hard accessible evidence only) and
what is reporting-only (R_hidden hidden evaluator)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

fig, ax = plt.subplots(figsize=(7.5, 3.0))
ax.axis("off")

BOX = dict(boxstyle="round,pad=0.35", fc="#f5f5f5", ec="#37474f", lw=1.1)
EBOX = dict(boxstyle="round,pad=0.35", fc="#e8f5e9", ec="#1b5e20", lw=1.3)
RBOX = dict(boxstyle="round,pad=0.35", fc="#ffebee", ec="#b71c1c", lw=1.3)

def box(x, y, text, style=BOX, fs=7.5):
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            bbox=style, family="sans-serif")

# serving / production row
box(0.06, 0.82, "production request\n(access RNG seed)", EBOX)
box(0.22, 0.82, "served policy\n(committed adapter)", BOX)
box(0.38, 0.82, "agent tools\nconversation", BOX)
box(0.54, 0.82, "first-attempt\noutcome", BOX)
box(0.72, 0.82, "official hidden\nevaluator", RBOX)
box(0.88, 0.82, "reward\n(reporting only)", RBOX)

# training row
box(0.22, 0.42, "rollouts from\ncandidate", BOX)
box(0.38, 0.42, "accessible evidence\n(E_hard: tool results,\nconversation)", EBOX)
box(0.54, 0.42, "signed replay\n(+/- rows)", BOX)
box(0.70, 0.42, "shadow candidate\n(LoRA)", BOX)
box(0.86, 0.42, "pre-commit gate\n(accessible instances)", BOX)

# commit loop
box(0.60, 0.10, "canary + atomic commit", BOX)
box(0.20, 0.10, "policy_version++\n(CRN preserved)", BOX)

def arrow(x1, y1, x2, y2, color="#37474f"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                 arrowstyle="-|>", mutation_scale=11, lw=1.1,
                 color=color, shrinkA=2, shrinkB=2))

arrow(0.105, 0.82, 0.175, 0.82)
arrow(0.265, 0.82, 0.335, 0.82)
arrow(0.425, 0.82, 0.495, 0.82)
arrow(0.585, 0.82, 0.675, 0.82, color="#b71c1c")
arrow(0.765, 0.82, 0.835, 0.82, color="#b71c1c")
# production -> training
arrow(0.54, 0.74, 0.54, 0.50, color="#1b5e20")
arrow(0.22, 0.50, 0.22, 0.44)
arrow(0.30, 0.42, 0.36, 0.42)
arrow(0.475, 0.42, 0.50, 0.42)
arrow(0.62, 0.42, 0.68, 0.42)
arrow(0.80, 0.42, 0.84, 0.42)
# gate -> commit -> served
arrow(0.86, 0.34, 0.78, 0.16, color="#1b5e20")
arrow(0.70, 0.14, 0.27, 0.14)
arrow(0.27, 0.18, 0.25, 0.74)

ax.text(0.50, 0.965, "v3 audit-grade pipeline: E_hard enters the loop, R_hidden never does",
        ha="center", fontsize=9.5, fontweight="bold")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig("paper/figures/fig1_protocol.png", dpi=300)
print("written paper/figures/fig1_protocol.png")
