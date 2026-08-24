import json

# per-template success counts, 16-task runs (protocol-compliant, seeds 0-7)
agg = {"frozen": {}, "naive": {}}
for s in range(8):
    for v in ["frozen", "naive"]:
        d = json.load(open(f"artifacts/v2/cts/{v}_s{s}/run_manifest.json"))
        # only use runs with 16 tasks (protocol-compliant version overwrote dirs)
        if len(d["tasks"]) != 16:
            continue
        for t in d["tasks"]:
            key = t["template"]
            agg[v].setdefault(key, [0, 0])
            agg[v][key][1] += 1
            agg[v][key][0] += t["y_pre"]
print(f"{'template':22s} {'frozen':>10s} {'naive':>10s}")
for key in sorted(set(agg["frozen"]) | set(agg["naive"])):
    f = agg["frozen"].get(key, [0, 0])
    n = agg["naive"].get(key, [0, 0])
    print(f"{key:22s} {f[0]}/{f[1]:<6d} {n[0]}/{n[1]:<6d}")
