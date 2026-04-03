#!/usr/bin/env python3
"""
05_batch_training_pipeline.py
Batch pipeline that compiles versioned training and evaluation datasets
from production data, with temporal splitting and candidate selection.

Sources:
1. Aligned (dirty, clean) pairs from object storage (external data)
2. User corrections from recipe_imports table (production feedback)

Rules:
- Temporal split: corrections before cutoff -> train, after -> eval
- Candidate selection: only recipes the model actually processed
- No data leakage: features available at prediction time only
"""

import json
import os
import subprocess
import psycopg2
import psycopg2.extras
from datetime import datetime, timedelta

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "mealie",
    "user": "mealie",
    "password": "mealie_password"
}

CONTAINER = "proj22-data"
VERSION = datetime.now().strftime("v_%Y%m%d")
OUT_DIR = os.path.expanduser("~/mealie-data/data/training")
PROCESSED_DIR = os.path.expanduser("~/mealie-data/data/processed")

def swift_upload(local_path, object_name):
    print(f"  [Swift] Uploading -> {CONTAINER}/{object_name}")
    subprocess.run(
        ["swift", "upload", CONTAINER, local_path, "--object-name", object_name],
        check=True, capture_output=True
    )

def load_external_pairs():
    """Load aligned + augmented pairs from processed data."""
    pairs = []
    for fname in ["aligned_pairs.jsonl", "augmented_pairs.jsonl"]:
        fpath = os.path.join(PROCESSED_DIR, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                for line in f:
                    pairs.append(json.loads(line))
    print(f"  External pairs loaded: {len(pairs)}")
    return pairs

def load_production_feedback():
    """
    Load user corrections from recipe_imports table.
    Only includes rows where user_correction IS NOT NULL
    (i.e., the model processed it AND the user corrected it).
    """
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT raw_text, model_output, user_correction, source_type, created_at
                FROM recipe_imports
                WHERE user_correction IS NOT NULL
                ORDER BY created_at
            """)
            rows = cur.fetchall()
    finally:
        conn.close()

    feedback_pairs = []
    for row in rows:
        feedback_pairs.append({
            "input": f"fix recipe: {row['raw_text']} <source:{row['source_type']}>",
            "target": row["user_correction"],
            "created_at": row["created_at"].isoformat()
        })
    print(f"  Production feedback pairs: {len(feedback_pairs)}")
    return feedback_pairs

def temporal_split(feedback_pairs, lookback_days=7):
    """
    Split feedback by time:
    - Train: corrections before cutoff
    - Eval: corrections after cutoff
    """
    cutoff = datetime.now() - timedelta(days=lookback_days)
    cutoff_str = cutoff.isoformat()

    train = [p for p in feedback_pairs if p["created_at"] < cutoff_str]
    eval_set = [p for p in feedback_pairs if p["created_at"] >= cutoff_str]

    print(f"  Temporal cutoff: {cutoff.strftime('%Y-%m-%d')}")
    print(f"  Feedback train: {len(train)}, Feedback eval: {len(eval_set)}")
    return train, eval_set

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Batch Training Pipeline - Versioned Dataset Compilation")
    print("=" * 60)

    # --- Step 1: Load external pairs ---
    print("\n=== Step 1: Load external (dirty, clean) pairs ===")
    external_pairs = load_external_pairs()

    # --- Step 2: Load production feedback ---
    print("\n=== Step 2: Load production feedback ===")
    feedback_pairs = load_production_feedback()

    # --- Step 3: Temporal split on feedback ---
    print("\n=== Step 3: Temporal split ===")
    feedback_train, feedback_eval = temporal_split(feedback_pairs, lookback_days=7)

    # --- Step 4: Combine into final train/eval sets ---
    print("\n=== Step 4: Compile final datasets ===")

    # Train = external pairs + feedback before cutoff
    # Use 90% of external for train, 10% for eval
    import random
    random.seed(42)
    random.shuffle(external_pairs)
    split_idx = int(len(external_pairs) * 0.9)
    ext_train = external_pairs[:split_idx]
    ext_eval = external_pairs[split_idx:]

    # Remove timestamp field from feedback for final output
    def clean_pair(p):
        return {"input": p["input"], "target": p["target"]}

    train_set = ext_train + [clean_pair(p) for p in feedback_train]
    eval_set = ext_eval + [clean_pair(p) for p in feedback_eval]

    print(f"  Train set: {len(train_set)} ({len(ext_train)} external + {len(feedback_train)} feedback)")
    print(f"  Eval set:  {len(eval_set)} ({len(ext_eval)} external + {len(feedback_eval)} feedback)")

    # --- Step 5: Save datasets ---
    print("\n=== Step 5: Save versioned datasets ===")
    train_path = os.path.join(OUT_DIR, "train.jsonl")
    eval_path = os.path.join(OUT_DIR, "eval.jsonl")

    with open(train_path, "w") as f:
        for item in train_set:
            f.write(json.dumps(item) + "\n")

    with open(eval_path, "w") as f:
        for item in eval_set:
            f.write(json.dumps(item) + "\n")

    # Manifest with full lineage
    manifest = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(),
        "model": "recipe_cleaner_t5",
        "split_strategy": "temporal",
        "temporal_cutoff": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
        "sources": {
            "external_aligned_pairs": "processed/cleaner/pairs/v_20260401/aligned_pairs.jsonl",
            "external_augmented_pairs": "processed/cleaner/augmented/v_20260401/augmented_pairs.jsonl",
            "production_feedback": "recipe_imports table (user_correction IS NOT NULL)"
        },
        "train_rows": len(train_set),
        "eval_rows": len(eval_set),
        "train_composition": {
            "external": len(ext_train),
            "production_feedback": len(feedback_train)
        },
        "eval_composition": {
            "external": len(ext_eval),
            "production_feedback": len(feedback_eval)
        },
        "candidate_selection": "Only recipes processed by the model and shown to users (recipe_imports with user_correction)",
        "leakage_prevention": "Temporal split on production feedback; external data randomly split"
    }

    manifest_path = os.path.join(OUT_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  Saved: {train_path} ({len(train_set)} rows)")
    print(f"  Saved: {eval_path} ({len(eval_set)} rows)")
    print(f"  Saved: {manifest_path}")

    # --- Step 6: Upload to object storage ---
    print("\n=== Step 6: Upload to object storage ===")
    swift_upload(train_path, f"training/cleaner/{VERSION}/train.jsonl")
    swift_upload(eval_path, f"training/cleaner/{VERSION}/eval.jsonl")
    swift_upload(manifest_path, f"training/cleaner/{VERSION}/manifest.json")

    # --- Verify ---
    print("\n=== Verifying object storage ===")
    result = subprocess.run(
        ["swift", "list", CONTAINER, "--prefix", "training/"],
        capture_output=True, text=True
    )
    print(result.stdout)

    # --- Print manifest ---
    print("=== Manifest ===")
    print(json.dumps(manifest, indent=2))

    print("\n=== Batch pipeline complete! ===")

if __name__ == "__main__":
    main()
