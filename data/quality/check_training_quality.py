import json
from pathlib import Path

train_path = Path("data/training/train.jsonl")
eval_path = Path("data/training/eval.jsonl")
report_path = Path("data/reports/training_quality_report.json")

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append({"__bad_json__": True})
    return rows

train_rows = load_jsonl(train_path)
eval_rows = load_jsonl(eval_path)

bad_train = sum(1 for r in train_rows if "__bad_json__" in r)
bad_eval = sum(1 for r in eval_rows if "__bad_json__" in r)

train_inputs = {r["input"] for r in train_rows if "input" in r}
eval_inputs = {r["input"] for r in eval_rows if "input" in r}
overlap = len(train_inputs.intersection(eval_inputs))

report = {
    "train_rows": len(train_rows),
    "eval_rows": len(eval_rows),
    "bad_train_json_rows": bad_train,
    "bad_eval_json_rows": bad_eval,
    "train_eval_input_overlap": overlap,
    "status": "PASS" if bad_train == 0 and bad_eval == 0 and overlap == 0 else "WARNING"
}

with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print(json.dumps(report, indent=2))
