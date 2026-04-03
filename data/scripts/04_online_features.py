#!/usr/bin/env python3
"""
04_online_features.py
Online feature computation path for the Recipe Cleaner (T5).

Given a raw recipe import, computes the model input string:
  "fix recipe: <raw_text> <source:SOURCE_TYPE>"

This is integrate-able with the Mealie service — when a user imports
a recipe via the scraper, this function is called to prepare the
input for T5 inference.
"""

import json
import psycopg2
import psycopg2.extras

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "mealie",
    "user": "mealie",
    "password": "mealie_password"
}

def compute_t5_input(raw_text, source_type):
    """
    Online feature computation for the T5 recipe cleaner.
    
    Args:
        raw_text: str, the raw/dirty recipe text from import
        source_type: str, one of 'web_scrape', 'ocr', 'manual'
    
    Returns:
        str: formatted input string ready for T5 tokenization
    """
    return f"fix recipe: {raw_text} <source:{source_type}>"


def demo_from_database():
    """
    Demo: pull recent recipe imports from Postgres and compute
    the T5 input feature for each one.
    """
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT import_id, user_id, raw_text, source_type, model_output, user_correction, created_at
                FROM recipe_imports
                ORDER BY created_at DESC
                LIMIT 5
            """)
            rows = cur.fetchall()
    finally:
        conn.close()

    print("=" * 70)
    print("Online Feature Computation - T5 Recipe Cleaner")
    print("=" * 70)

    for i, row in enumerate(rows):
        print(f"\n--- Import #{i+1} ---")
        print(f"  Import ID:    {row['import_id']}")
        print(f"  User ID:      {row['user_id']}")
        print(f"  Source Type:   {row['source_type']}")
        print(f"  Created At:    {row['created_at']}")
        print(f"  Raw Text:      {row['raw_text'][:150]}...")

        # Compute the T5 input feature
        t5_input = compute_t5_input(row['raw_text'], row['source_type'])
        print(f"\n  >> T5 Model Input:")
        print(f"     {t5_input[:200]}...")

        # Show expected output
        print(f"\n  >> Model Output (stored):")
        print(f"     {row['model_output'][:150]}...")

        if row['user_correction']:
            print(f"\n  >> User Correction:")
            print(f"     {row['user_correction'][:150]}...")

    # Also demo a raw call without DB
    print("\n" + "=" * 70)
    print("Direct API-style call (no DB):")
    print("=" * 70)
    raw = "Title: Choco1ate Chip Cook1es\nIngredients: 12 cup butter | 1 egg | 2 sugar\nInstructions: Mix a11 ingredients. Bake at 35O degrees."
    source = "ocr"
    result = compute_t5_input(raw, source)
    print(f"\n  Input raw_text: {raw}")
    print(f"  Input source:   {source}")
    print(f"\n  >> Computed T5 input:")
    print(f"     {result}")


if __name__ == "__main__":
    demo_from_database()
