import json
from pathlib import Path

pairs_path = Path("data/processed/aligned_pairs.jsonl")
report_path = Path("data/reports/ingestion_quality_report.json")

total = 0
bad_json = 0
empty_dirty = 0
empty_clean = 0
duplicates = 0
seen = set()

with open(pairs_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        total += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            bad_json += 1
            continue

        dirty = str(row.get("dirty", "")).strip()
        clean = str(row.get("clean", "")).strip()

        if not dirty:
            empty_dirty += 1
        if not clean:
            empty_clean += 1

        key = (dirty, clean)
        if key in seen:
            duplicates += 1
        else:
            seen.add(key)

report = {
    "total_rows": total,
    "bad_json_rows": bad_json,
    "empty_dirty_rows": empty_dirty,
    "empty_clean_rows": empty_clean,
    "duplicate_pairs": duplicates,
    "status": "PASS" if bad_json == 0 and empty_dirty == 0 and empty_clean == 0 else "WARNING"
}

with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print(json.dumps(report, indent=2))
