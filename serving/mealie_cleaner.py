#!/usr/bin/env python3
import json
import time
import subprocess
import os
import requests

MEALIE_URL = "http://129.114.26.25:30900"
TRITON_URL = "http://129.114.26.25:30910"
MEALIE_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJsb25nX3Rva2VuIjp0cnVlLCJpZCI6IjE3MWIyMTVmLTg5OTQtNDU0Ny1hZjFjLTUxNTc5NzFlNDdhMCIsIm5hbWUiOiJtbG9wcyIsImludGVncmF0aW9uX2lkIjoiZ2VuZXJpYyIsImV4cCI6MTkzNDIyNTMwMH0.Oum8pAQDTnttoM55AOR5OOJZUfrItbPCaqwQKU9FDhw"
FEEDBACK_FILE = "/tmp/feedback_pairs.jsonl"
CONTAINER = "ObjStore_proj22"
HEADERS = {"Authorization": f"Bearer {MEALIE_TOKEN}", "Content-Type": "application/json"}
PROCESSED_FILE = "/tmp/processed_slugs.json"

def load_processed():
    if os.path.exists(PROCESSED_FILE):
        with open(PROCESSED_FILE) as f:
            return json.load(f)
    return {}

def save_processed(processed):
    with open(PROCESSED_FILE, "w") as f:
        json.dump(processed, f)

def format_recipe_for_triton(recipe):
    title = recipe.get("name", "")
    ingredients = " | ".join([
        i.get("display", "") or i.get("note", "") 
        for i in recipe.get("recipeIngredient", [])
        if i.get("display") or i.get("note")
    ])
    instructions = " | ".join([
        s.get("text", "") 
        for s in recipe.get("recipeInstructions", [])
        if isinstance(s, dict) and s.get("text")
    ])
    return f"fix recipe: Title: {title}\nIngredients: {ingredients}\nInstructions: {instructions}"

def call_triton(text):
    payload = {
        "inputs": [{
            "name": "INPUT_TEXT",
            "shape": [1, 1],
            "datatype": "BYTES",
            "data": [text]
        }]
    }
    response = requests.post(f"{TRITON_URL}/v2/models/recipe_model/infer", json=payload)
    if response.status_code == 200:
        return response.json()["outputs"][0]["data"][0]
    return None

def get_recipes():
    response = requests.get(f"{MEALIE_URL}/api/recipes", headers=HEADERS)
    if response.status_code == 200:
        return response.json().get("items", [])
    return []

def get_recipe(slug):
    response = requests.get(f"{MEALIE_URL}/api/recipes/{slug}", headers=HEADERS)
    if response.status_code == 200:
        return response.json()
    return None

def save_feedback(input_text, target_text, source="model_output"):
    pair = {
        "input": input_text,
        "target": target_text,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": source
    }
    with open(FEEDBACK_FILE, "a") as f:
        f.write(json.dumps(pair) + "\n")
    subprocess.run([
        "swift", "upload", CONTAINER,
        FEEDBACK_FILE,
        "--object-name", "feedback/feedback_pairs.jsonl"
    ], capture_output=True)
    print(f"Saved [{source}] training pair!")

def check_for_user_edits(slug, cleaned_at, original_input):
    recipe = get_recipe(slug)
    if not recipe:
        return
    updated_at = recipe.get("updatedAt", "")
    if updated_at > cleaned_at:
        title = recipe.get("name", "")
        ingredients = " | ".join([
            i.get("display", "") for i in recipe.get("recipeIngredient", [])
            if i.get("display")
        ])
        instructions = " | ".join([
            s.get("text", "") for s in recipe.get("recipeInstructions", [])
            if isinstance(s, dict) and s.get("text")
        ])
        user_version = f"Title: {title}\nIngredients: {ingredients}\nInstructions: {instructions}"
        save_feedback(original_input, user_version, source="user_correction")
        print(f"User correction captured for: {title}")
        return updated_at
    return None

def main():
    import os
    print("Starting Mealie recipe cleaner...")
    processed = load_processed()

    while True:
        recipes = get_recipes()
        for r in recipes:
            slug = r.get("slug")

            if slug in processed:
                new_time = check_for_user_edits(
                    slug,
                    processed[slug]["cleaned_at"],
                    processed[slug]["original_input"]
                )
                if new_time:
                    processed[slug]["cleaned_at"] = new_time
                    save_processed(processed)
                continue

            recipe = get_recipe(slug)
            if not recipe:
                continue

            print(f"Processing: {recipe.get('name')}")
            original_input = format_recipe_for_triton(recipe)
            cleaned = call_triton(original_input)

            if cleaned:
                print(f"Cleaned: {cleaned[:100]}...")
                save_feedback(original_input, cleaned, source="model_output")
                processed[slug] = {
                    "cleaned_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "original_input": original_input
                }
                save_processed(processed)
            else:
                print(f"Failed to clean: {slug}")

        time.sleep(30)

if __name__ == "__main__":
    main()
