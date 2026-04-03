#!/usr/bin/env python3
"""
03_data_generator.py
Synthetic data generator that simulates users importing recipes
via Mealie's web scraper and providing feedback.

Writes to PostgreSQL tables: users, recipes, recipe_imports
"""

import json
import os
import random
import time
import psycopg2
import psycopg2.extras
from datetime import datetime, timedelta
from faker import Faker
from tqdm import tqdm

fake = Faker()
random.seed(42)
Faker.seed(42)

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "mealie",
    "user": "mealie",
    "password": "mealie_password"
}

NUM_USERS = 50
RECIPES_PER_USER = 10
DELAY = 0.05  # seconds between requests

# Sample dirty recipes (from our aligned pairs)
SAMPLE_DIRTY_RECIPES = None
SAMPLE_CLEAN_RECIPES = None

def load_sample_recipes():
    """Load a sample of aligned pairs for generating realistic imports."""
    global SAMPLE_DIRTY_RECIPES, SAMPLE_CLEAN_RECIPES
    pairs_path = os.path.expanduser("~/mealie-data/data/processed/aligned_pairs.jsonl")
    SAMPLE_DIRTY_RECIPES = []
    SAMPLE_CLEAN_RECIPES = []
    with open(pairs_path) as f:
        for i, line in enumerate(f):
            if i >= 2000:
                break
            pair = json.loads(line)
            # Extract just the recipe text after "fix recipe: " and before " <source:"
            dirty = pair["input"]
            dirty = dirty.replace("fix recipe: ", "", 1)
            dirty = dirty.rsplit(" <source:", 1)[0]
            SAMPLE_DIRTY_RECIPES.append(dirty)
            SAMPLE_CLEAN_RECIPES.append(pair["target"])

def get_connection():
    return psycopg2.connect(**DB_CONFIG)

def create_users(conn):
    """Create synthetic users with realistic profiles."""
    print(f"\n--- Creating {NUM_USERS} synthetic users ---")
    dietary_options = [
        [], ["vegetarian"], ["vegan"], ["gluten-free"],
        ["nut-allergy"], ["dairy-free"], ["vegetarian", "gluten-free"]
    ]
    users = []
    with conn.cursor() as cur:
        for i in tqdm(range(NUM_USERS), desc="Creating users"):
            username = fake.user_name() + str(i)
            dietary = random.choice(dietary_options)
            goals = {
                "calories": random.choice([1500, 1800, 2000, 2200, 2500]),
                "protein": random.choice([50, 60, 80, 100, 120])
            }
            budget = round(random.uniform(50, 150), 2)
            cur.execute("""
                INSERT INTO users (username, dietary_restrictions, nutrition_goals, budget_constraint)
                VALUES (%s, %s, %s, %s)
                RETURNING user_id
            """, (username, json.dumps(dietary), json.dumps(goals), budget))
            user_id = cur.fetchone()[0]
            users.append(user_id)
            time.sleep(DELAY)
    conn.commit()
    print(f"  Created {len(users)} users")
    return users

def create_recipes(conn):
    """Insert a base set of recipes into the catalog from RecipeNLG."""
    print(f"\n--- Loading recipes into catalog ---")
    import pandas as pd
    df = pd.read_csv(os.path.expanduser("~/mealie-data/data/raw/recipenlg.csv"), nrows=500)
    recipe_ids = []
    with conn.cursor() as cur:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading recipes"):
            try:
                ingr = json.loads(row["ingredients"]) if isinstance(row["ingredients"], str) else []
                instr_list = json.loads(row["directions"]) if isinstance(row["directions"], str) else []
                instr = " ".join(instr_list)
            except json.JSONDecodeError:
                continue
            cur.execute("""
                INSERT INTO recipes (title, ingredients, instructions, source)
                VALUES (%s, %s, %s, %s)
                RETURNING recipe_id
            """, (row["title"], json.dumps(ingr), instr, "recipenlg"))
            recipe_ids.append(cur.fetchone()[0])
    conn.commit()
    print(f"  Loaded {len(recipe_ids)} recipes")
    return recipe_ids

def simulate_recipe_imports(conn, users):
    """Simulate users importing dirty recipes via the web scraper."""
    print(f"\n--- Simulating recipe imports ---")
    total_imports = 0
    with conn.cursor() as cur:
        for user_id in tqdm(users, desc="Simulating imports"):
            num_imports = random.randint(3, RECIPES_PER_USER)
            for _ in range(num_imports):
                idx = random.randint(0, len(SAMPLE_DIRTY_RECIPES) - 1)
                raw_text = SAMPLE_DIRTY_RECIPES[idx]
                model_output = SAMPLE_CLEAN_RECIPES[idx]
                source_type = random.choice(["web_scrape", "web_scrape", "web_scrape", "ocr", "manual"])

                # 30% of users correct the model output
                user_correction = None
                if random.random() < 0.3:
                    user_correction = model_output + " (user edited)"

                # Spread imports over the last 4 weeks
                days_ago = random.randint(0, 28)
                created = datetime.now() - timedelta(days=days_ago, hours=random.randint(0, 23))

                cur.execute("""
                    INSERT INTO recipe_imports
                    (user_id, raw_text, model_output, user_correction, source_type, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (str(user_id), raw_text, model_output, user_correction, source_type, created))
                total_imports += 1
                time.sleep(DELAY)
    conn.commit()
    print(f"  Total imports: {total_imports}")

def simulate_meal_plan_interactions(conn, users, recipe_ids):
    """Simulate meal plan feedback (kept for schema completeness)."""
    print(f"\n--- Simulating meal plan interactions ---")
    actions = ["cooked", "cooked", "kept", "skipped", "swapped"]
    total = 0
    with conn.cursor() as cur:
        for user_id in tqdm(users, desc="Meal plan feedback"):
            for week_offset in range(4):
                plan_week = (datetime.now() - timedelta(weeks=week_offset)).date()
                # 5 recipes per week
                week_recipes = random.sample(recipe_ids, min(5, len(recipe_ids)))
                for rid in week_recipes:
                    action = random.choice(actions)
                    days_ago = week_offset * 7 + random.randint(0, 6)
                    created = datetime.now() - timedelta(days=days_ago)
                    cur.execute("""
                        INSERT INTO meal_plan_interactions
                        (user_id, recipe_id, plan_week, action, created_at)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (str(user_id), str(rid), plan_week, action, created))
                    total += 1
                    time.sleep(DELAY)
    conn.commit()
    print(f"  Total interactions: {total}")

def print_summary(conn):
    """Print table row counts."""
    print("\n=== Database Summary ===")
    with conn.cursor() as cur:
        for table in ["users", "recipes", "recipe_imports", "meal_plan_interactions"]:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            count = cur.fetchone()[0]
            print(f"  {table}: {count} rows")

def main():
    print("=" * 60)
    print("Data Generator - Simulating Mealie user traffic")
    print("=" * 60)

    load_sample_recipes()
    print(f"Loaded {len(SAMPLE_DIRTY_RECIPES)} sample recipe pairs")

    conn = get_connection()
    try:
        users = create_users(conn)
        recipe_ids = create_recipes(conn)
        simulate_recipe_imports(conn, users)
        simulate_meal_plan_interactions(conn, users, recipe_ids)
        print_summary(conn)
    finally:
        conn.close()

    print("\n=== Data generation complete! ===")

if __name__ == "__main__":
    main()
