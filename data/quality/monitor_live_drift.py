import json
from pathlib import Path

train_path = Path("data/training/train.jsonl")
report_path = Path("data/reports/live_drift_report.json")

def avg_input_length(path):
    lengths = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "input" in row:
                lengths.append(len(row["input"]))
    return sum(lengths) / len(lengths) if lengths else 0

baseline_avg = avg_input_length(train_path)

sample_live_inputs = [
    "fix recipe: 12 cup sugar 1 12 tsp vanilla mix bake cool",
    "fix recipe: <br> 1 cup flour &amp; 2 eggs </div>",
    "fix recipe: 112 tsp salt and broken scraped text"
]

live_avg = sum(len(x) for x in sample_live_inputs) / len(sample_live_inputs)
html_artifact_count = sum(1 for x in sample_live_inputs if any(tok in x for tok in ["<br>", "&amp;", "</div>"]))
fraction_error_count = sum(1 for x in sample_live_inputs if any(tok in x for tok in ["12 cup", "112 tsp", "1 12 tsp"]))

drift_detected = live_avg > baseline_avg * 1.5 or html_artifact_count > 0 or fraction_error_count > 0

report = {
    "baseline_avg_input_length": baseline_avg,
    "live_avg_input_length": live_avg,
    "html_artifact_count": html_artifact_count,
    "fraction_error_count": fraction_error_count,
    "drift_detected": drift_detected,
    "status": "WARNING" if drift_detected else "PASS"
}

with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print(json.dumps(report, indent=2))
