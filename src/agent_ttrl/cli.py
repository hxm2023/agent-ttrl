"""agent-ttrl CLI skeleton (design doc §16.4).

Capability separation: online/run commands never touch sealed roles; the audit
evaluator lives in a separate entry point (agent-ttrl-audit) with its own
permissions.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent_ttrl.credit.paired_credit import paired_credit
from agent_ttrl.environments.cts_runner import run_fixture
from agent_ttrl.environments.cts_world import ShiftConfig, ShiftFamily
from benchmarks.controlled_tool_shift.fixtures import FIXTURES, HAND_EXPECTED_ALPHA, HAND_INTERVALS, HAND_U


def cmd_smoke(args) -> int:
    """R001-style correctness smoke: CTS golden pack + hand fixture."""
    results = {}
    for fx in FIXTURES:
        run = run_fixture(fx)
        if fx.fid.startswith("CTS-F1"):  # fail-closed fixtures
            ok = fx.fail_closed_reason in f"{run.verdict.status}/{run.verdict.reason_code}"
        else:
            ok = run.verdict.status in ("OK", "NO_RELIABLE_CREDIT")
        results[fx.fid] = {"ok": ok, "status": run.verdict.status,
                           "reason": run.verdict.reason_code}
    # hand fixture: interval gating at eta=0.05
    eta = 0.05
    alpha = [1 if (lo > eta or hi < -eta) else 0 for lo, hi in HAND_INTERVALS]
    results["HAND"] = {"ok": alpha == HAND_EXPECTED_ALPHA, "alpha": alpha}
    failed = {k: v for k, v in results.items() if not v["ok"]}
    print(json.dumps(results, indent=2))
    if failed:
        print(f"SMOKE FAIL: {list(failed)}", file=sys.stderr)
        return 1
    print("SMOKE PASS")
    return 0


def cmd_run_stream(args) -> int:
    """R001: run a frozen CTS stream (environment + oracle only; no policy update)."""
    domain = args.domain
    fx = next((f for f in FIXTURES if f.fid == f"CTS-{domain}"), None)
    if fx is None:
        print(f"unknown fixture CTS-{domain}", file=sys.stderr)
        return 2
    run = run_fixture(fx)
    manifest = {
        "schema_version": "agent-ttrl.run-manifest.v1",
        "milestone": "M1",
        "variant": "frozen",
        "fixture": fx.fid,
        "U": run.U.tolist(),
        "verdict": {"status": run.verdict.status, "reason_code": run.verdict.reason_code},
        "oracle_canonical": run.oracle_canonical,
        "conflicts": run.conflicts,
        "errors": run.errors,
    }
    out = Path(args.output)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


def cmd_audit_run(args) -> int:
    """Audit a run manifest: schema-validate + ledger conservation (stub checks)."""
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    required = {"schema_version", "milestone", "fixture", "verdict"}
    missing = required - set(manifest)
    if missing:
        print(f"AUDIT FAIL: missing {sorted(missing)}", file=sys.stderr)
        return 1
    if manifest["verdict"]["status"] not in ("OK", "NO_RELIABLE_CREDIT", "DEGENERATE_GROUP",
                                             "NO_SUPPORT", "INVALID", "NO_RELIABLE_CREDIT"):
        print(f"AUDIT FAIL: bad verdict status {manifest['verdict']['status']}", file=sys.stderr)
        return 1
    print("AUDIT PASS")
    return 0


def cmd_build_report(args) -> int:
    """Aggregate fixture results from run manifests (raw -> table rebuild)."""
    import glob
    out = Path(args.output)
    rows = []
    for mf in sorted(glob.glob(str(Path(args.manifest_dir) / "*.json"))):
        m = json.loads(Path(mf).read_text(encoding="utf-8"))
        rows.append({"fixture": m.get("fixture"), "status": m.get("verdict", {}).get("status")})
    report = {"fixtures": rows, "n": len(rows),
              "ok": all(r["status"] in ("OK", "NO_RELIABLE_CREDIT") for r in rows)}
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="agent-ttrl")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_smoke = sub.add_parser("smoke")
    p_smoke.set_defaults(fn=cmd_smoke)
    p_run = sub.add_parser("run-stream")
    p_run.add_argument("--domain", default="F01")
    p_run.add_argument("--output", default="artifacts/run_manifest.json")
    p_run.set_defaults(fn=cmd_run_stream)
    p_audit = sub.add_parser("audit-run")
    p_audit.add_argument("--manifest", required=True)
    p_audit.set_defaults(fn=cmd_audit_run)
    p_report = sub.add_parser("build-report")
    p_report.add_argument("--manifest-dir", default="artifacts")
    p_report.add_argument("--output", default="artifacts/report.json")
    p_report.set_defaults(fn=cmd_build_report)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
