#!/usr/bin/env python3
"""
07_generate_mock_inference_log.py
Generate a mock inference log from eval.jsonl for testing drift monitoring.
Uses the eval set as simulated production traffic.
"""
import json
import os
from datetime import datetime, timedelta
import random

EVAL_PATH = os.path.expanduser("~/recipe-scraper-mlops/data/processed/aligned_pairs.jsonl")
OUTPUT_PATH = os.path.expanduser("~/recipe-scraper-mlops/data/inference_log.jsonl")

def main():
    print("\n=== Generating Mock Inference Log ===")

    if not os.path.exists(EVAL_PATH):
        print(f"ERROR: {EVAL_PATH} not found!")
        return

    # Load a sample of pairs
    records = []
    with open(EVAL_PATH) as f:
        for i, line in enumerate(f):
            if i >= 500:  # take 500 samples
                break
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"  Loaded {len(records)} eval pairs")

    # Write as mock inference log with timestamps
    base_time = datetime.now() - timedelta(hours=len(records))
    with open(OUTPUT_PATH, "w") as f:
        for i, record in enumerate(records):
            entry = {
                "input": record.get("input", ""),
                "timestamp": (base_time + timedelta(minutes=i*5)).isoformat(),
                "source": random.choice(["web_scrape", "ocr", "manual"])
            }
            f.write(json.dumps(entry) + "\n")

    print(f"  Mock inference log written: {OUTPUT_PATH}")
    print(f"  Total entries: {len(records)}")
    print("\n  Done! Run 06_monitor_drift.py to test drift monitoring.")

if __name__ == "__main__":
    main()
