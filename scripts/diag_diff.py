import json
for s in [0, 1, 2, 3, 4]:
    fr = json.load(open(f"artifacts/v2/cts/frozen_s{s}/run_manifest.json"))
    nv = json.load(open(f"artifacts/v2/cts/naive_s{s}/run_manifest.json"))
    diffs = []
    for i in range(min(len(fr["tasks"]), len(nv["tasks"]))):
        ft, nt = fr["tasks"][i], nv["tasks"][i]
        if ft["y_pre"] != nt["y_pre"]:
            diffs.append((i, ft["template"], ft["y_pre"], nt["y_pre"]))
    print(f"s{s}: {diffs}")
