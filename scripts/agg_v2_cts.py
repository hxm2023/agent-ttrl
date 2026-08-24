import json

print(f"{'seed':>4} | {'frozen':>8} | {'naive':>8} | {'egc':>8} | held")
allrows = []
for s in [0, 1, 2, 3, 4]:
    fr = json.load(open(f"artifacts/v2/cts/frozen_s{s}/run_manifest.json"))
    nv = json.load(open(f"artifacts/v2/cts/naive_s{s}/run_manifest.json"))
    eg = json.load(open(f"artifacts/v2/cts/egc_s{s}/run_manifest.json"))
    print(f"{s:>4} | {fr['aupc_prequential']:>8.3f} | {nv['aupc_prequential']:>8.3f} | {eg['aupc_prequential']:>8.3f} | {fr['held_out_template']}")
    allrows.append((fr["aupc_prequential"], nv["aupc_prequential"], eg["aupc_prequential"],
                    fr["aupc_heldout_template"], nv["aupc_heldout_template"], eg["aupc_heldout_template"]))
f = [r[0] for r in allrows]; n = [r[1] for r in allrows]; e = [r[2] for r in allrows]
print(f"mean: frozen {sum(f)/5:.3f} naive {sum(n)/5:.3f} egc {sum(e)/5:.3f}")
print(f"deltas naive-frozen: {[round(n[i]-f[i],3) for i in range(5)]}")
print(f"deltas egc-frozen:   {[round(e[i]-f[i],3) for i in range(5)]}")
print(f"heldout: frozen {sum(r[3] for r in allrows)/10:.2f} naive {sum(r[4] for r in allrows)/10:.2f} egc {sum(r[5] for r in allrows)/10:.2f}")
