"""F3 mechanism analysis: per-template AUPC + entity-id hallucination rate +
tool-call rate, frozen vs naive, both backbones.
- Qwen: v3q manifests (full exec logs with calls+obs).
- Mistral: v3 _16.log lines (y_pre per task; rollout texts for id analysis).
Writes JSON consumed by make_mechanism_figure.py."""
import json
import os
import re
import sys
from pathlib import Path

ID_RE = re.compile(r"(order-\d+|user-\d+|sku-\d+|item-\d+)")
OUT = Path("/root/autodl-tmp/agent-ttrl/artifacts/mechanism")


# ---------------------------------------------------------------- Qwen (manifests)
def load_manifests(root: str, arm: str, seeds: list[int]):
    ms = []
    for s in seeds:
        p = Path(root) / f"{arm}_s{s}" / "run_manifest.json"
        if p.exists():
            ms.append(json.load(open(p)))
    return ms


def template_aupc(ms: list[dict]) -> dict[str, float]:
    agg: dict[str, list[float]] = {}
    for m in ms:
        for t in m["tasks"]:
            agg.setdefault(t["template"], []).append(1.0 if t["y_pre"] else 0.0)
    return {k: sum(v) / len(v) for k, v in agg.items()}


def hallucination_rate(ms: list[dict]) -> float:
    hall = 0
    total = 0
    for m in ms:
        for t in m["tasks"]:
            obs_ids = set()
            for e in t["exec"]:
                if "obs" in e:
                    obs_ids |= set(ID_RE.findall(e["obs"]))
            for e in t["exec"]:
                if "call" not in e:
                    continue
                total += 1
                call_ids = set(ID_RE.findall(e["call"]))
                if call_ids and not call_ids & obs_ids:
                    hall += 1
    return hall / max(1, total)


def tool_call_rate(ms: list[dict]) -> float:
    n = 0
    with_call = 0
    for m in ms:
        for t in m["tasks"]:
            first = t["exec"][0] if t["exec"] else {}
            n += 1
            if first.get("n_calls", 0) > 0:
                with_call += 1
    return with_call / max(1, n)


# ---------------------------------------------------------------- Mistral (logs)
def parse_v3_logs(logs: list[Path]) -> list[dict]:
    """Parse v3 _16.log lines into per-task records with y_pre and rollouts."""
    out = []
    for p in logs:
        for line in open(p, encoding="utf-8", errors="ignore"):
            m = re.search(r"t(\d+) (\S+): y_pre=(\d+\.?\d*) hidden=(\w+)", line)
            if not m:
                continue
            rec = {"task": int(m.group(1)), "template": m.group(2),
                   "y_pre": float(m.group(3)), "rollout_texts": []}
            for rm in re.finditer(r"'text': '([^']*)'", line):
                rec["rollout_texts"].append(rm.group(1))
            out.append(rec)
    return out


def mistral_template_aupc(recs: list[dict]) -> dict[str, float]:
    agg: dict[str, list[float]] = {}
    for r in recs:
        agg.setdefault(r["template"], []).append(r["y_pre"])
    return {k: sum(v) / len(v) for k, v in agg.items()}


def mistral_example_id_rate(recs: list[dict]) -> float:
    """Fraction of rollout generations containing a NON-task example id.
    The v3 diagnosis: few-shot example ids ('order-EXAMPLE' style) leaked
    into actions. Count generations whose ids never appear in any other
    generation of the same task (proxy for hallucinated ids)."""
    per_task: dict[int, list[str]] = {}
    for r in recs:
        per_task.setdefault(r["task"], []).extend(r["rollout_texts"])
    hall = 0
    total = 0
    for task, texts in per_task.items():
        all_ids = set()
        for t in texts:
            all_ids |= set(ID_RE.findall(t))
        for t in texts:
            ids = set(ID_RE.findall(t))
            total += 1
            # ids that appear in exactly this one generation = likely invented
            if ids and len(ids & all_ids) < len(ids):
                hall += 1
    return hall / max(1, total)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    v3 = "/root/autodl-tmp/agent-ttrl/artifacts/v3/cts"
    v3q = "/root/autodl-tmp/agent-ttrl/artifacts/v3q/cts"
    seeds = list(range(8))

    result = {"backbones": {}}

    # ---- Qwen
    frozen = load_manifests(v3q, "frozen", [0])
    naive = load_manifests(v3q, "naive", seeds)
    result["backbones"]["qwen"] = {
        "frozen_tpl": template_aupc(frozen),
        "naive_tpl": template_aupc(naive),
        "frozen_aupc": frozen[0]["aupc_prequential"],
        "naive_aupc": [m["aupc_prequential"] for m in naive],
        "frozen_hall": hallucination_rate(frozen),
        "naive_hall": hallucination_rate(naive),
        "frozen_tool": tool_call_rate(frozen),
        "naive_tool": tool_call_rate(naive),
    }
    print("qwen frozen", round(result["backbones"]["qwen"]["frozen_aupc"], 4),
          "naive", round(sum(result["backbones"]["qwen"]["naive_aupc"]) / 8, 4),
          "| hall", round(result["backbones"]["qwen"]["frozen_hall"], 3),
          round(result["backbones"]["qwen"]["naive_hall"], 3),
          "| tool", round(result["backbones"]["qwen"]["frozen_tool"], 3),
          round(result["backbones"]["qwen"]["naive_tool"], 3))

    # ---- Mistral (logs; manifests partially overwritten)
    v3_logs = Path(v3)
    frozen_logs = [v3_logs / f"frozen_s{s}_16.log" for s in [0]]
    naive_logs = [v3_logs / f"naive_s{s}_16.log" for s in seeds]
    f_recs = parse_v3_logs(frozen_logs)
    n_recs = parse_v3_logs(naive_logs)
    result["backbones"]["mistral"] = {
        "frozen_tpl": mistral_template_aupc(f_recs),
        "naive_tpl": mistral_template_aupc(n_recs),
        "frozen_aupc": sum(r["y_pre"] for r in f_recs) / max(1, len(f_recs)),
        "naive_aupc": [
            sum(r["y_pre"] for r in parse_v3_logs([v3_logs / f"naive_s{s}_16.log"]))
            / max(1, len(f_recs)) for s in seeds],
        "frozen_hall": mistral_example_id_rate(f_recs),
        "naive_hall": mistral_example_id_rate(n_recs),
        "frozen_tool": None,
        "naive_tool": None,
    }
    print("mistral frozen", round(result["backbones"]["mistral"]["frozen_aupc"], 4),
          "naive", round(sum(result["backbones"]["mistral"]["naive_aupc"]) / 8, 4),
          "| hall", round(result["backbones"]["mistral"]["frozen_hall"], 3),
          round(result["backbones"]["mistral"]["naive_hall"], 3))

    with open(OUT / "mechanism.json", "w") as f:
        json.dump(result, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
