
#!/usr/bin/env python3
"""
02_quality_training.py
Checkpoint 2: Validate training pairs before retraining.
Run before kicking off LoRA retraining.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from quality.checks import check_training_set_quality, save_report

PROCESSED_DIR = os.path.expanduser("~/recipe-scraper-mlops/data/processed")
REPORT_DIR = os.path.expanduser("~/recipe-scraper-mlops/data/reports")

def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def main():
    print("\n=== Checkpoint 2: Training Set Quality Check ===")

    aligned_path = f"{PROCESSED_DIR}/aligned_pairs.jsonl"
    augmented_path = f"{PROCESSED_DIR}/augmented_pairs.jsonl"

    if not os.path.exists(aligned_path):
        print(f"ERROR: {aligned_path} not found!")
        sys.exit(1)

    print("Loading aligned pairs...")
    aligned = load_jsonl(aligned_path)

    augmented = []
    if os.path.exists(augmented_path):
        print("Loading augmented pairs...")
        augmented = load_jsonl(augmented_path)

    all_pairs = aligned + augmented
    print(f"  Total pairs loaded: {len(all_pairs)}")

    report = check_training_set_quality(all_pairs)
    save_report(report, f"{REPORT_DIR}/qc_training_set.json")

    print(f"\n  Total pairs:          {report['total_pairs']}")
    print(f"  Missing input:        {report['missing_input']}")
    print(f"  Missing target:       {report['missing_target']}")
    print(f"  Identical pairs:      {report['identical_pairs']} ({report['identical_pct']}%)")
    print(f"  Too short pairs:      {report['too_short_pairs']}")
    print(f"  Missing fix prefix:   {report.get('missing_fix_prefix', 0)}")
    print(f"\n  RESULT: {'✅ PASSED' if report['pass'] else '❌ FAILED'}")

    if not report['pass']:
        print("  WARNING: Training set quality check failed! Do not retrain.")
        sys.exit(1)

if __name__ == "__main__":
    main()
