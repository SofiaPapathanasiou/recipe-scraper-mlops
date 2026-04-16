
import json
import logging
from datetime import datetime
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("data_quality")

def check_ingestion_quality(records: List[Dict]) -> Dict:
    total = len(records)
    if total == 0:
        return {"checkpoint": "ingestion", "pass": False, "error": "empty dataset"}

    missing_title = sum(1 for r in records if not r.get("title", "").strip())
    missing_ingr = sum(1 for r in records if not r.get("ingredients"))
    missing_instr = sum(1 for r in records if not r.get("instructions"))
    duplicates = total - len({r.get("title", "").lower() for r in records})
    empty_recipes = sum(1 for r in records if not r.get("title") and not r.get("ingredients"))

    report = {
        "checkpoint": "ingestion",
        "timestamp": datetime.now().isoformat(),
        "total_records": total,
        "missing_title": missing_title,
        "missing_title_pct": round(missing_title / total * 100, 2),
        "missing_ingredients": missing_ingr,
        "missing_instructions": missing_instr,
        "duplicate_titles": duplicates,
        "empty_recipes": empty_recipes,
        "pass": (missing_title / total < 0.05) and (empty_recipes / total < 0.01)
    }
    logger.info(f"[Ingestion QC] pass={report['pass']} missing_title={missing_title} dupes={duplicates}")
    return report

def check_training_set_quality(pairs: List[Dict]) -> Dict:
    total = len(pairs)
    if total == 0:
        return {"checkpoint": "training_set", "pass": False, "error": "empty dataset"}

    missing_input = sum(1 for p in pairs if not p.get("input", "").strip())
    missing_target = sum(1 for p in pairs if not p.get("target", "").strip())
    identical = sum(1 for p in pairs if p.get("input") == p.get("target"))
    too_short = sum(1 for p in pairs if len(p.get("input", "")) < 20 or len(p.get("target", "")) < 20)

    report = {
        "checkpoint": "training_set",
        "timestamp": datetime.now().isoformat(),
        "total_pairs": total,
        "missing_input": missing_input,
        "missing_target": missing_target,
        "identical_pairs": identical,
        "identical_pct": round(identical / total * 100, 2),
        "too_short_pairs": too_short,
        "pass": missing_input == 0 and missing_target == 0 and (identical / total < 0.1)
    }
    logger.info(f"[Training QC] pass={report['pass']} identical={identical} too_short={too_short}")
    return report

def check_inference_drift(baseline: List[str], current: List[str]) -> Dict:
    def avg_len(texts):
        return sum(len(t) for t in texts) / max(len(texts), 1)
    def title_missing_rate(texts):
        return sum(1 for t in texts if "Title:" not in t) / max(len(texts), 1)
    def ingr_missing_rate(texts):
        return sum(1 for t in texts if "Ingredients:" not in t) / max(len(texts), 1)

    baseline_avg = avg_len(baseline)
    current_avg = avg_len(current)
    length_drift = abs(current_avg - baseline_avg) / max(baseline_avg, 1)
    title_miss = title_missing_rate(current)
    ingr_miss = ingr_missing_rate(current)

    report = {
        "checkpoint": "inference_drift",
        "timestamp": datetime.now().isoformat(),
        "baseline_samples": len(baseline),
        "current_samples": len(current),
        "baseline_avg_length": round(baseline_avg, 2),
        "current_avg_length": round(current_avg, 2),
        "length_drift_pct": round(length_drift * 100, 2),
        "title_missing_rate_pct": round(title_miss * 100, 2),
        "ingredients_missing_rate_pct": round(ingr_miss * 100, 2),
        "pass": length_drift < 0.3 and title_miss < 0.1
    }
    logger.info(f"[Drift QC] pass={report['pass']} drift={report['length_drift_pct']}% title_miss={report['title_missing_rate_pct']}%")
    return report

def save_report(report: Dict, path: str):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Quality report saved: {path}")
