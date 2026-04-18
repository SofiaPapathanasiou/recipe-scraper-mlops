#!/usr/bin/env python3
import json
import time
import subprocess
import requests

MEALIE_URL = "http://129.114.26.25:30900"
TRITON_URL = "http://129.114.26.25:30910"
MEALIE_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJsb25nX3Rva2VuIjp0cnVlLCJpZCI6IjE3MWIyMTVmLTg5OTQtNDU0Ny1hZjFjLTUxNTc5NzFlNDdhMCIsIm5hbWUiOiJtbG9wcyIsImludGVncmF0aW9uX2lkIjoiZ2VuZXJpYyIsImV4cCI6MTkzNDIyNTMwMH0.Oum8pAQDTnttoM55AOR5OOJZUfrItbPCaqwQKU9FDhw"
FEEDBACK_FILE = "/tmp/feedback_pairs.jsonl"
CONTAINER = "ObjStore_proj22"

HEADERS = {"Authorization": f"Bearer {MEALIE_TOKEN}"}

def format_recipe_for_triton(recipe):
    title = recipe.get("name", "")
    ingredients = " | ".join([i.get("display", "") for i in recipe.get("recipeIngredient", [])])
    instructions = " | ".join([s.get("text", "") for s in recipe.get("recipeInstructions", [])])
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

def save_feedback(text, cleaned):
    pair = {
        "input": text,
        "target": cleaned,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": "user_feedback"
    }
    with open(FEEDBACK_FILE, "a") as f:
        f.write(json.dumps(pair) + "\n")
    
    # Upload to object store
    subprocess.run([
        "swift", "upload", CONTAINER,
        FEEDBACK_FILE,
        "--object-name", "feedback/feedback_pairs.jsonl"
    ], capture_output=True)
    print("Saved and uploaded to object store!")

def main():
    print("Starting Mealie recipe cleaner...")
    processed = set()

    while True:
        recipes = get_recipes()
        for r in recipes:
            slug = r.get("slug")
            if slug in processed:
                continue
            recipe = get_recipe(slug)
            if not recipe:
                continue
            print(f"Processing: {recipe.get('name')}")
            text = format_recipe_for_triton(recipe)
            cleaned = call_triton(text)
            if cleaned:
                print(f"Cleaned: {cleaned[:100]}...")
                save_feedback(text, cleaned)
                processed.add(slug)
            else:
                print(f"Failed to clean: {slug}")
        time.sleep(30)

if __name__ == "__main__":
    main()
