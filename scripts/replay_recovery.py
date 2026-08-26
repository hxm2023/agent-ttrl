"""Replay trajectories from run manifests and measure END-OF-EPISODE
accessible success (within-episode recovery), vs first-attempt y_pre.

Protocol-clean: the replay executes the recorded calls on a fresh
instance of the same template/seed and reads the environment's
accessible_success() — accessible evidence only, no hidden evaluator.
Usage: python scripts/replay_recovery.py <root> <arm> <seeds...>
"""
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/agent-ttrl/src")
from agent_ttrl.environments.cts_v2 import TEMPLATES

CALL_RE = re.compile(r'([a-z_]+)\(([^)]*)\)')


def parse_call(call: str) -> tuple[str, dict]:
    m = CALL_RE.search(call)
    name = m.group(1)
    kwargs = {}
    for am in re.finditer(r"'([a-z_]+)': '([^']*)'", m.group(2)):
        kwargs[am.group(1)] = am.group(2)
    if not kwargs:  # fallback: key="value"
        for am in re.finditer(r"([a-z_]+)=\"([^\"]*)\"", m.group(2)):
            kwargs[am.group(1)] = am.group(2)
    return name, kwargs


def replay_manifest(path: Path) -> dict:
    m = json.load(open(path))
    per_tpl = defaultdict(lambda: {"first": [0, 0], "end": [0, 0]})
    for t in m["tasks"]:
        tpl = TEMPLATES[t["template"]]
        inst = tpl.instantiate(random.Random(1000 + m["seed"] * 100 + t["task"]))
        for e in t["exec"]:
            if "call" not in e:
                continue
            name, kwargs = parse_call(e["call"])
            inst.exec_call(name, kwargs)
        end_ok = 1.0 if inst.accessible_success() else 0.0
        per_tpl[t["template"]]["first"][0] += t["y_pre"]
        per_tpl[t["template"]]["first"][1] += 1
        per_tpl[t["template"]]["end"][0] += end_ok
        per_tpl[t["template"]]["end"][1] += 1
    return per_tpl


def main() -> int:
    root = Path(sys.argv[1])
    arm = sys.argv[2]
    seeds = [int(s) for s in sys.argv[3:]]
    agg = defaultdict(lambda: {"first": [0, 0], "end": [0, 0]})
    for s in seeds:
        p = root / f"{arm}_s{s}" / "run_manifest.json"
        if not p.exists():
            print(f"[skip] {p}")
            continue
        per_tpl = replay_manifest(p)
        for tpl, v in per_tpl.items():
            for key in ("first", "end"):
                agg[tpl][key][0] += v[key][0]
                agg[tpl][key][1] += v[key][1]
    print(f"== {arm} seeds={seeds}")
    tot = defaultdict(lambda: [0, 0])
    for tpl in sorted(agg):
        f = agg[tpl]["first"][0] / max(1, agg[tpl]["first"][1])
        e = agg[tpl]["end"][0] / max(1, agg[tpl]["end"][1])
        print(f"  {tpl:22s} first {f:.3f}  end-of-episode {e:.3f}  recovery +{e-f:+.3f}")
        for key in ("first", "end"):
            tot[key][0] += agg[tpl][key][0]
            tot[key][1] += agg[tpl][key][1]
    tf = tot["first"][0] / max(1, tot["first"][1])
    te = tot["end"][0] / max(1, tot["end"][1])
    print(f"  OVERALL            first {tf:.3f}  end-of-episode {te:.3f}  recovery +{te-tf:+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
