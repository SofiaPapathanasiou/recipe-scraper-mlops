#!/usr/bin/env python3
"""
01_ingest_and_transform.py
Ingestion + Transform pipeline for Recipe Cleaner (T5)

Steps:
1. Load Recipe1M+ (layer1.json) and RecipeNLG (recipenlg.csv)
2. Align by title to get (dirty, clean) pairs
3. Apply synthetic corruptions to expand dataset
4. Upload raw + processed data to object storage
"""

import json
import os
import re
import random
import subprocess
import tempfile
import pandas as pd
from datetime import datetime
from tqdm import tqdm

CONTAINER = "proj22-data"
VERSION = datetime.now().strftime("v_%Y%m%d")
RAW_DIR = os.path.expanduser("~/mealie-data/data/raw")
OUT_DIR = os.path.expanduser("~/mealie-data/data/processed")

# ============================================================
# Corruption functions (simulate real scraping/OCR errors)
# ============================================================

def mangle_fractions(text):
    """'1/2 cup' -> '12 cup' or 'cup' or '1 2 cup'"""
    def replace_fn(m):
        n, d = m.group(1), m.group(2)
        choice = random.choice(["remove_slash", "drop_all", "space"])
        if choice == "remove_slash":
            return n + d
        elif choice == "drop_all":
            return ""
        else:
            return n + " " + d
    return re.sub(r"(\d+)/(\d+)", replace_fn, text)

def ocr_substitutions(text):
    """Simulate OCR errors: O<->0, l<->1, rn->m"""
    subs = [("O", "0"), ("0", "O"), ("l", "1"), ("1", "l"), ("rn", "m")]
    sub_from, sub_to = random.choice(subs)
    return text.replace(sub_from, sub_to, 1)

def drop_units(text):
    """Drop a measurement unit from the text."""
    units = ["cup", "cups", "tablespoon", "tablespoons", "tbsp",
             "teaspoon", "teaspoons", "tsp", "ounce", "ounces", "oz",
             "pound", "pounds", "lb", "lbs"]
    for unit in units:
        pattern = re.compile(r'\b' + re.escape(unit) + r'\b', re.IGNORECASE)
        if pattern.search(text):
            return pattern.sub("", text, count=1).strip()
    return text

def merge_lines(lines):
    """Merge two random adjacent lines."""
    if len(lines) < 2:
        return lines
    result = list(lines)
    idx = random.randint(0, len(result) - 2)
    result[idx] = result[idx] + " " + result[idx + 1]
    result.pop(idx + 1)
    return result

def add_html_artifacts(text):
    """Add HTML remnants."""
    artifacts = ["&amp;", "&nbsp;", "<br>", "&#39;", "&quot;", "<p>", "</p>"]
    pos = random.randint(0, max(0, len(text) - 1))
    artifact = random.choice(artifacts)
    return text[:pos] + artifact + text[pos:]

def corrupt_recipe(title, ingredients, instructions):
    """Apply 1-3 random corruptions to a clean recipe, return dirty text."""
    corruption_fns = [
        ("fraction", lambda t, i, s: (t, [mangle_fractions(x) for x in i], [mangle_fractions(x) for x in s])),
        ("ocr", lambda t, i, s: (ocr_substitutions(t), [ocr_substitutions(x) for x in i], [ocr_substitutions(x) for x in s])),
        ("drop_unit", lambda t, i, s: (t, [drop_units(x) for x in i], s)),
        ("merge", lambda t, i, s: (t, merge_lines(i), merge_lines(s))),
        ("html", lambda t, i, s: (add_html_artifacts(t), i, s)),
    ]
    num = random.randint(1, 3)
    chosen = random.sample(corruption_fns, num)
    t, i, s = title, list(ingredients), list(instructions)
    for _, fn in chosen:
        t, i, s = fn(t, i, s)
    return t, i, s

def format_recipe_text(title, ingredients, instructions):
    """Format a recipe as flat text for the T5 model."""
    parts = [f"Title: {title}"]
    if ingredients:
        parts.append("Ingredients: " + " | ".join(ingredients))
    if instructions:
        parts.append("Instructions: " + " | ".join(instructions))
    return "\n".join(parts)

# ============================================================
# Swift upload helper
# ============================================================

def swift_upload(local_path, object_name):
    """Upload a file to Swift object storage."""
    print(f"  [Swift] Uploading -> {CONTAINER}/{object_name}")
    subprocess.run(
        ["swift", "upload", CONTAINER, local_path, "--object-name", object_name],
        check=True, capture_output=True
    )

# ============================================================
# Main pipeline
# ============================================================

def main():
    random.seed(42)
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- Step 1: Upload raw data to object storage ---
    print("\n=== Step 1: Uploading raw data to object storage ===")
    swift_upload(f"{RAW_DIR}/layer1.json", "raw/recipe1m/layer1.json")
    swift_upload(f"{RAW_DIR}/recipenlg.csv", "raw/recipenlg/full_dataset.csv")
    print("Raw data uploaded.")

    # --- Step 2: Load datasets ---
    print("\n=== Step 2: Loading datasets ===")
    print("Loading Recipe1M+...")
    with open(f"{RAW_DIR}/layer1.json") as f:
        r1m_data = json.load(f)

    r1m_by_title = {}
    for r in r1m_data:
        key = r["title"].strip().lower()
        r1m_by_title[key] = r
    print(f"  Recipe1M+ unique titles: {len(r1m_by_title)}")

    print("Loading RecipeNLG...")
    df = pd.read_csv(f"{RAW_DIR}/recipenlg.csv")
    df_r1m = df[df["source"] == "Recipes1M"].copy()
    print(f"  RecipeNLG (Recipes1M source): {len(df_r1m)}")

    # --- Step 3: Align (dirty, clean) pairs ---
    print("\n=== Step 3: Aligning (dirty, clean) pairs ===")
    pairs = []
    for _, row in tqdm(df_r1m.iterrows(), total=len(df_r1m), desc="Aligning"):
        key = row["title"].strip().lower()
        if key not in r1m_by_title:
            continue
        r1m_recipe = r1m_by_title[key]

        # Dirty version (Recipe1M+)
        dirty_ingr = [i["text"] for i in r1m_recipe.get("ingredients", [])]
        dirty_instr = [i["text"] for i in r1m_recipe.get("instructions", [])]
        dirty_text = format_recipe_text(r1m_recipe["title"], dirty_ingr, dirty_instr)

        # Clean version (RecipeNLG)
        try:
            clean_ingr = json.loads(row["ingredients"]) if isinstance(row["ingredients"], str) else []
            clean_instr = json.loads(row["directions"]) if isinstance(row["directions"], str) else []
        except json.JSONDecodeError:
            continue
        clean_text = format_recipe_text(row["title"], clean_ingr, clean_instr)

        # Only keep pairs where dirty != clean (there's an actual difference)
        if dirty_text != clean_text:
            pairs.append({
                "input": "fix recipe: " + dirty_text + " <source:web_scrape>",
                "target": clean_text
            })

    print(f"  Aligned pairs with differences: {len(pairs)}")

    # --- Step 4: Synthetic augmentation ---
    print("\n=== Step 4: Synthetic augmentation ===")
    # Take clean RecipeNLG recipes and corrupt them
    augmented = []
    aug_count = min(50000, len(df_r1m))  # generate up to 50K extra pairs
    sample_rows = df_r1m.sample(n=aug_count, random_state=42)

    for _, row in tqdm(sample_rows.iterrows(), total=aug_count, desc="Augmenting"):
        try:
            clean_ingr = json.loads(row["ingredients"]) if isinstance(row["ingredients"], str) else []
            clean_instr = json.loads(row["directions"]) if isinstance(row["directions"], str) else []
        except json.JSONDecodeError:
            continue

        clean_text = format_recipe_text(row["title"], clean_ingr, clean_instr)
        dirty_title, dirty_ingr, dirty_instr = corrupt_recipe(
            row["title"], clean_ingr, clean_instr
        )

        source_tag = random.choice(["web_scrape", "ocr", "manual"])
        dirty_text = format_recipe_text(dirty_title, dirty_ingr, dirty_instr)

        augmented.append({
            "input": "fix recipe: " + dirty_text + f" <source:{source_tag}>",
            "target": clean_text
        })

    print(f"  Augmented pairs: {len(augmented)}")

    # --- Step 5: Save processed data ---
    print("\n=== Step 5: Saving processed data ===")
    pairs_path = os.path.join(OUT_DIR, "aligned_pairs.jsonl")
    with open(pairs_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"  Aligned pairs saved: {pairs_path} ({len(pairs)} rows)")

    aug_path = os.path.join(OUT_DIR, "augmented_pairs.jsonl")
    with open(aug_path, "w") as f:
        for a in augmented:
            f.write(json.dumps(a) + "\n")
    print(f"  Augmented pairs saved: {aug_path} ({len(augmented)} rows)")

    # Manifest
    manifest = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(),
        "sources": {
            "recipe1m": "raw/recipe1m/layer1.json",
            "recipenlg": "raw/recipenlg/full_dataset.csv"
        },
        "aligned_pairs": len(pairs),
        "augmented_pairs": len(augmented),
        "total_training_pairs": len(pairs) + len(augmented),
        "corruption_types": ["fraction_mangling", "ocr_substitution", "drop_units", "merge_lines", "html_artifacts"],
        "augmentation_source_tags": ["web_scrape", "ocr", "manual"]
    }
    manifest_path = os.path.join(OUT_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest saved: {manifest_path}")

    # --- Step 6: Upload processed data to object storage ---
    print("\n=== Step 6: Uploading processed data to object storage ===")
    swift_upload(pairs_path, f"processed/cleaner/pairs/{VERSION}/aligned_pairs.jsonl")
    swift_upload(aug_path, f"processed/cleaner/augmented/{VERSION}/augmented_pairs.jsonl")
    swift_upload(manifest_path, f"processed/cleaner/{VERSION}/manifest.json")

    # --- Done ---
    print("\n=== Pipeline complete! ===")
    print(f"  Aligned pairs: {len(pairs)}")
    print(f"  Augmented pairs: {len(augmented)}")
    print(f"  Total: {len(pairs) + len(augmented)}")
    print(f"  Version: {VERSION}")

    # Verify upload
    print("\n=== Verifying object storage ===")
    result = subprocess.run(
        ["swift", "list", CONTAINER, "--prefix", "processed/"],
        capture_output=True, text=True
    )
    print(result.stdout)

if __name__ == "__main__":
    main()
