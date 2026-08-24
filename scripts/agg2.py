import json
for s in [0, 1, 2]:
    fr = json.load(open(f"artifacts/v2/cts/frozen_s{s}/run_manifest.json"))
    nv = json.load(open(f"artifacts/v2/cts/naive_s{s}/run_manifest.json"))
    eg = json.load(open(f"artifacts/v2/cts/egc_s{s}/run_manifest.json"))
    print(f"s{s}: frozen {fr['aupc_prequential']:.3f} naive {nv['aupc_prequential']:.3f} egc {eg['aupc_prequential']:.3f} (held {fr['held_out_template']})")
    print(f"     heldout: {fr['aupc_heldout_template']:.2f} / {nv['aupc_heldout_template']:.2f} / {eg['aupc_heldout_template']:.2f}")
