"""批量重跑 9 个 CVE 的 verify 阶段（不依赖 LLM），结果写 JSON。"""
import json, subprocess, sys, time, traceback
from pathlib import Path
import os, yaml

source_root = Path(__file__).resolve().parents[1]
py = sys.executable
env = dict(os.environ)
env["PYTHONPATH"] = str(source_root)

CVES = ["CVE-2021-36979", "CVE-2021-36980", "CVE-2021-38593",
        "CVE-2021-45926", "CVE-2021-45927", "CVE-2021-45930",
        "CVE-2021-45931", "CVE-2021-45932", "CVE-2021-45933"]

def run_verify(cve):
    cmd = [py, str(source_root / "scripts" / "run_verify.py"), cve,
           "--dataset-root", "Dataset", "--workspace-root", "workspaces"]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(source_root), env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="backslashreplace", timeout=7200)
    dt = time.time() - t0
    out = (r.stdout or "") + "\n" + (r.stderr or "")
    return r.returncode, dt, out

def summary(cve):
    v = Path(source_root) / "workspaces" / cve / "artifacts" / "verify" / "verify_result.yaml"
    if not v.exists():
        return {"verdict": "missing"}
    d = yaml.safe_load(v.read_text(encoding="utf-8")) or {}
    return {
        "verdict": d.get("verdict"),
        "reason": d.get("reason"),
        "pre_exit": d.get("pre_patch_exit_code"),
        "post_exit": d.get("post_patch_exit_code"),
        "confidence": d.get("confidence"),
    }

results = []
for cve in CVES:
    row = {"cve": cve}
    try:
        rc, dt, out = run_verify(cve)
        row["rc"] = rc
        row["seconds"] = round(dt)
        s = summary(cve)
        row["verify"] = s
        if s.get("verdict") != "success":
            row["detail"] = out[-1200:]
        print(f"[{time.strftime('%H:%M:%S')}] {cve}: rc={rc} {dt:.0f}s verdict={s.get('verdict')}", flush=True)
    except Exception as e:
        row["error"] = traceback.format_exc()[-600:]
        print(f"[{time.strftime('%H:%M:%S')}] {cve}: ERROR {e}", flush=True)
    results.append(row)
    Path(source_root / "batch_verify_progress.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

Path(source_root / "batch_verify_results.json").write_text(
    json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print("BATCH VERIFY DONE")
print(json.dumps(results, ensure_ascii=False, indent=2))