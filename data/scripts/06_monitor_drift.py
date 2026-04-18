#!/usr/bin/env python3
"""
06_monitor_drift.py
Checkpoint 3: Monitor live inference data for drift vs baseline.
Run via cron every hour in production.
"""
import json
import os
import sys
import subprocess
from datetime import datetime

sys.path.insert(0, '/app')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from quality.checks import check_inference_drift, save_report

DATA_ROOT = os.environ.get("DATA_ROOT", os.path.expanduser("~/recipe-scraper-mlops/data"))
PROCESSED_DIR = os.path.join(DATA_ROOT, "processed")
INFERENCE_LOG = os.path.join(DATA_ROOT, "inference_log.jsonl")
REPORT_DIR = os.path.join(DATA_ROOT, "reports/drift")

def upload_report_to_swift(local_path, object_name):
    subprocess.run([
        "swift", "upload", "ObjStore_proj22",
        local_path, "--object-name", object_name
    ], capture_output=True)
    print(f"  Report uploaded to object store: {object_name}")

def load_jsonl(path, key, limit=1000):
    records = []
    if not os.path.exists(path):
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                if key in r:
                    records.append(r[key])
            if len(records) >= limit:
                break
    return records

def main():
    print("\n=== Checkpoint 3: Inference Drift Monitoring ===")
    os.makedirs(REPORT_DIR, exist_ok=True)

    # Load baseline from training data
    print("Loading baseline data...")
    baseline = load_jsonl(f"{PROCESSED_DIR}/aligned_pairs.jsonl", "input", limit=1000)
    if not baseline:
        print("ERROR: No baseline data found!")
        sys.exit(1)
    print(f"  Baseline samples: {len(baseline)}")

    # Load live inference inputs
    print("Loading inference log...")
    current = load_jsonl(INFERENCE_LOG, "input", limit=500)

    if not current:
        print("  No inference data yet — skipping drift check.")
        sys.exit(0)
    print(f"  Current samples: {len(current)}")

    # Run drift check
    report = check_inference_drift(baseline, current)

    # Save report with timestamp
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_report(report, f"{REPORT_DIR}/drift_{ts}.json")

    print(f"\n  Baseline avg length:  {report['baseline_avg_length']} chars")
    print(f"  Current avg length:   {report['current_avg_length']} chars")
    print(f"  Length drift:         {report['length_drift_pct']}%")
    print(f"  Title missing rate:   {report['title_missing_rate_pct']}%")
    print(f"  Ingredients missing:  {report['ingredients_missing_rate_pct']}%")
    print(f"\n  RESULT: {'✅ PASSED' if report['pass'] else '❌ FAILED - Drift detected!'}")

    upload_report_to_swift(
        f"{REPORT_DIR}/drift_{ts}.json",
        f"quality_reports/drift/drift_{ts}.json"
    )

    if not report['pass']:
        print("  ALERT: Significant drift detected! Consider retraining.")
        sys.exit(1)

if __name__ == "__main__":
    main()
