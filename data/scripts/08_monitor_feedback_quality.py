#!/usr/bin/env python3
"""
08_monitor_feedback_quality.py
Monitor quality of user feedback pairs coming from Mealie.
Checks for duplicates, quality issues, and drift vs baseline.
Run hourly via cron.
"""
import json
import os
import sys
import subprocess
from datetime import datetime

sys.path.insert(0, '/app')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from quality.checks import check_inference_drift, check_training_set_quality, save_report

DATA_ROOT = os.environ.get("DATA_ROOT", os.path.expanduser("~/recipe-scraper-mlops/data"))
BASELINE_FILE = os.path.join(DATA_ROOT, "processed/aligned_pairs.jsonl")
FEEDBACK_FILE = "/tmp/feedback_pairs.jsonl"
REPORT_DIR = os.path.join(DATA_ROOT, "reports/feedback")
CONTAINER = "ObjStore_proj22"

def download_feedback():
    os.makedirs("/tmp", exist_ok=True)
    result = subprocess.run([
        "swift", "download", CONTAINER,
        "feedback/feedback_pairs.jsonl",
        "-o", FEEDBACK_FILE
    ], capture_output=True)
    return result.returncode == 0

def load_jsonl(path, limit=1000):
    records = []
    if not os.path.exists(path):
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
            if len(records) >= limit:
                break
    return records

def upload_report(local_path, object_name):
    subprocess.run([
        "swift", "upload", CONTAINER,
        local_path, "--object-name", object_name
    ], capture_output=True)

def main():
    print("\n=== Feedback Quality Monitor ===")
    os.makedirs(REPORT_DIR, exist_ok=True)

    print("Downloading feedback from object store...")
    if not download_feedback():
        print("No feedback data yet — skipping.")
        sys.exit(0)

    feedback = load_jsonl(FEEDBACK_FILE)
    print(f"  Feedback pairs: {len(feedback)}")

    if not feedback:
        print("No feedback pairs found.")
        sys.exit(0)

    # Quality check
    print("\n--- Feedback Quality Check ---")
    quality_report = check_training_set_quality(feedback)

    # Drift check
    print("\n--- Drift Check vs Baseline ---")
    baseline = load_jsonl(BASELINE_FILE, limit=1000)
    baseline_inputs = [r.get("input", "") for r in baseline]
    feedback_inputs = [r.get("input", "") for r in feedback]
    drift_report = check_inference_drift(baseline_inputs, feedback_inputs)

    # Duplicates
    unique = len(set(r.get("input", "") for r in feedback))
    duplicate_count = len(feedback) - unique

    # Fairness stats
    model_output = sum(1 for p in feedback if p.get("source") == "model_output")
    user_correction = sum(1 for p in feedback if p.get("source") == "user_correction")
    accepted = sum(1 for p in feedback if p.get("source") == "accepted")
    rejected = sum(1 for p in feedback if p.get("source") == "rejected")

    # Combined report
    report = {
        "checkpoint": "feedback_quality",
        "timestamp": datetime.now().isoformat(),
        "total_feedback_pairs": len(feedback),
        "unique_pairs": unique,
        "duplicate_pairs": duplicate_count,
        "source_distribution": {
            "model_output": model_output,
            "user_correction": user_correction,
            "accepted": accepted,
            "rejected": rejected
        },
        "quality": quality_report,
        "drift": drift_report,
        "pass": quality_report["pass"] and drift_report["pass"]
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"{REPORT_DIR}/feedback_quality_{ts}.json"
    save_report(report, report_path)
    upload_report(report_path, f"quality_reports/feedback/feedback_quality_{ts}.json")

    print(f"\n  Total pairs:     {len(feedback)}")
    print(f"  Unique pairs:    {unique}")
    print(f"  Duplicates:      {duplicate_count}")
    print(f"  Quality pass:    {quality_report['pass']}")
    print(f"  Drift pass:      {drift_report['pass']}")
    print(f"\n  Source distribution (fairness):")
    print(f"  Model output:    {model_output}")
    print(f"  User correction: {user_correction}")
    print(f"  Accepted:        {accepted}")
    print(f"  Rejected:        {rejected}")
    print(f"\n  RESULT: {'✅ PASSED' if report['pass'] else '❌ FAILED'}")

if __name__ == "__main__":
    main()
