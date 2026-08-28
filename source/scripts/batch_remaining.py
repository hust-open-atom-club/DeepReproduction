"""批量跑剩余 7 个 CVE：build -> poc -> verify，结果写 JSON + 日志。"""
import json, subprocess, sys, time, traceback
from pathlib import Path

source_root = Path(__file__).resolve().parents[1]
py = sys.executable
env = dict(__import__("os").environ)
env["PYTHONPATH"] = str(source_root)

CVES = ["CVE-2021-36979", "CVE-2021-36980", "CVE-2021-38593",
        "CVE-2021-45930", "CVE-2021-45931", "CVE-2021-45932", "CVE-2021-45933"]

def run_script(name, args):
    cmd = [py, str(source_root / "scripts" / name)] + args
    print(f"[{time.strftime('%H:%M:%S')}] RUN {name} {' '.join(args)}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(source_root), env=env, capture_output=True, text=True, encoding="utf-8", errors="backslashreplace", timeout=3600)
    dt = time.time() - t0
    out = (r.stdout or "") + "\n" + (r.stderr or "")
    print(f"[{time.strftime('%H:%M:%S')}] DONE {name} exit={r.returncode} ({dt:.0f}s)", flush=True)
    print(out[-1500:], flush=True)
    return r.returncode, out

def summary(cve):
    ws = Path(source_root) / "workspaces" / cve
    b = ws / "artifacts/build/build_artifact.yaml"
    p = ws / "artifacts/poc/poc_artifact.yaml"
    v = ws / "artifacts/verify/verify_result.yaml"
    import yaml
    row = {"cve": cve, "build": None, "poc": None, "verify": None}
    if b.exists():
        d = yaml.safe_load(b.read_text(encoding="utf-8"))
        row["build"] = d.get("build_success")
    if p.exists():
        d = yaml.safe_load(p.read_text(encoding="utf-8"))
        row["poc"] = {"verified": d.get("reproducer_verified"), "exit": d.get("observed_exit_code"), "stderr": (d.get("observed_stderr") or "")[:200]}
    if v.exists():
        d = yaml.safe_load(v.read_text(encoding="utf-8"))
        row["verify"] = {"verdict": d.get("verdict"), "reason": d.get("reason")}
    return row

results = []
for cve in CVES:
    row = {"cve": cve}
    try:
        rc, out = run_script("run_build.py", [cve, "--dataset-root", "Dataset", "--workspace-root", "workspaces"])
        row["build_rc"] = rc
        s = summary(cve)
        if s["build"] is not True:
            row["stage_fail"] = "build"
            row["detail"] = out[-800:]
            results.append(row)
            continue
        rc, out = run_script("run_poc.py", [cve, "--dataset-root", "Dataset", "--workspace-root", "workspaces"])
        row["poc_rc"] = rc
        s = summary(cve)
        if not (s["poc"] or {}).get("verified"):
            row["stage_fail"] = "poc"
            row["detail"] = out[-800:]
            results.append(row)
            continue
        rc, out = run_script("run_verify.py", [cve, "--dataset-root", "Dataset", "--workspace-root", "workspaces"])
        row["verify_rc"] = rc
        s = summary(cve)
        row["final"] = s
        if (s["verify"] or {}).get("verdict") != "success":
            row["stage_fail"] = "verify"
            row["detail"] = out[-800:]
    except Exception as e:
        row["error"] = traceback.format_exc()[-500:]
    results.append(row)
    Path(source_root / "batch_remaining_progress.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

Path(source_root / "batch_remaining_results.json").write_text(
    json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print("BATCH DONE")
print(json.dumps(results, ensure_ascii=False, indent=2))