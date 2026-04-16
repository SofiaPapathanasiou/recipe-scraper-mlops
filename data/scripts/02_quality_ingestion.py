#!/usr/bin/env python3
"""
02_quality_ingestion.py
Checkpoint 1: Validate raw recipe data at ingestion.
Run after raw data is loaded, before any processing.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from quality.checks import check_ingestion_quality, save_report

RAW_DIR = os.path.expanduser("~/mealie-data/data/raw")
REPORT_DIR = os.path.expanduser("~/mealie-data/data/quality_reports")

def main():
    print("\n=== Checkpoint 1: Ingestion Quality Check ===")

    # Load Recipe1M
    layer1_path = f"{RAW_DIR}/layer1.json"
    if not os.path.exists(layer1_path):
        print(f"ERROR: {layer1_path} not found!")
        sys.exit(1)

    print("Loading layer1.json...")
    with open(layer1_path) as f:
        records = json.load(f)

    report = check_ingestion_quality(records)
    save_report(report, f"{REPORT_DIR}/qc_ingestion.json")

    print(f"\n  Total records:        {report['total_records']}")
    print(f"  Missing title:        {report['missing_title']} ({report['missing_title_pct']}%)")
    print(f"  Missing ingredients:  {report['missing_ingredients']}")
    print(f"  Missing instructions: {report['missing_instructions']}")
    print(f"  Duplicate titles:     {report['duplicate_titles']}")
    print(f"  Empty recipes:        {report['empty_recipes']}")
    print(f"\n  RESULT: {'✅ PASSED' if report['pass'] else '❌ FAILED'}")

    if not report['pass']:
        print("  WARNING: Ingestion quality check failed! Review before processing.")
        sys.exit(1)

if __name__ == "__main__":
    main()
