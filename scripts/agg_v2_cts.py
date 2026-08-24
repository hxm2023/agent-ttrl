import json
rows = []
for s in [0, 1, 2, 3, 4]:
    fr = json.load(open(f"artifacts/v2/cts/frozen_s{s}/run_manifest.json"))
    nv = json.load(open(f"artifacts/v2/cts/naive_s{s}/run_manifest.json"))
    print(f"s{s}: frozen {fr['aupc_prequential']:.3f}/{fr['aupc_heldout_template']:.2f} held={fr['held_out_template']:22s} | naive {nv['aupc_prequential']:.3f}/{nv['aupc_heldout_template']:.2f}")
    rows.append(nv["aupc_prequential"] - fr["aupc_prequential"])
print("deltas:", [round(d, 3) for d in rows], "mean", round(sum(rows) / len(rows), 3))
