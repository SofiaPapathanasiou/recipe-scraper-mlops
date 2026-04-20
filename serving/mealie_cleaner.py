#!/usr/bin/env python3
import json
import os
import time
import subprocess
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
        try:
            with open(PROCESSED_FILE) as f:
                return json.load(f)
        except Exception:
            return {}
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
        s.get("text", "").strip()
        for s in recipe.get("recipeInstructions", [])
        if isinstance(s, dict) and s.get("text", "").strip()
    ])
    return f"fix recipe: Title: {title}\nIngredients: {ingredients}\nInstructions: {instructions}"

def format_note_text(original_input):
    """Format original recipe with bullet points."""
    lines = original_input.replace("fix recipe: ", "").split("\n")
    result = []
    for line in lines:
        line = line.strip()
        if line.startswith("Title:"):
            result.append(f"**Title:** {line.replace('Title:', '').strip()}")
            result.append("")
        elif line.startswith("Ingredients:"):
            result.append("**Ingredients:**")
            items = line.replace("Ingredients:", "").strip().split(" | ")
            for item in items:
                item = item.strip()
                if item:
                    result.append(f"- {item}")
            result.append("")
        elif line.startswith("Instructions:"):
            result.append("**Instructions:**")
            steps = line.replace("Instructions:", "").strip().split(" | ")
            for i, step in enumerate(steps, 1):
                step = step.strip()
                if step:
                    result.append(f"{i}. {step}")
        elif line:
            result.append(line)
    return "\n".join(result)

def format_cleaned_note(cleaned_text):
    """Format cleaned output with bullet points and dedup ingredients."""
    result = []
    for line in cleaned_text.split("\n"):
        line = line.strip()
        if line.startswith("Title:"):
            result.append(f"**Title:** {line.replace('Title:', '').strip()}")
            result.append("")
        elif line.startswith("Ingredients:"):
            result.append("**Ingredients:**")
            items = line.replace("Ingredients:", "").strip().split(" | ")
            seen = set()
            for item in items:
                item = item.strip()
                if item and item not in seen:
                    seen.add(item)
                    result.append(f"- {item}")
            result.append("")
        elif line.startswith("Instructions:"):
            result.append("**Instructions:**")
            steps = line.replace("Instructions:", "").strip().split(" | ")
            for i, step in enumerate(steps, 1):
                step = step.strip()
                if step:
                    result.append(f"{i}. {step}")
        elif line:
            result.append(line)
    return "\n".join(result)

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

def add_note_to_recipe(slug, original_input, cleaned_output):
    """Add original and cleaned recipe as notes so user can compare."""
    payload = {
        "notes": [
            {
                "title": "Original Recipe",
                "text": format_note_text(original_input)
            },
            {
                "title": "Triton Cleaned",
                "text": format_cleaned_note(cleaned_output)
            }
        ]
    }
    response = requests.patch(
        f"{MEALIE_URL}/api/recipes/{slug}",
        headers=HEADERS,
        json=payload
    )
    if response.status_code == 200:
        print("Added original and cleaned versions as notes!")
    else:
        print(f"Failed to add notes: {response.status_code}")

def save_feedback(input_text, target_text, source="model_output"):
    pair = {
        "input": input_text,
        "target": target_text,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": source
    }
    existing = set()
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE) as f:
            for line in f:
                if line.strip():
                    existing.add(json.loads(line)["input"])
    if input_text in existing:
        print("Skipping duplicate feedback pair!")
        return
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
        return None
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
        post_feedback(slug, "reject")
        print(f"User correction captured for: {title}")
        return updated_at
    return None
def check_feedback_endpoint(slug, original_input, cleaned_output):
    """Check if user accepted or rejected the cleaned recipe."""
    response = requests.get(
        f"{MEALIE_URL}/api/ml/feedback",
        headers=HEADERS,
        params={"slug": slug}
    )
    if response.status_code != 200:
        return
    
    data = response.json()
    rating = data.get("rating")
    
    if rating == "accept":
        save_feedback(original_input, cleaned_output, source="accepted")
        print(f"Recipe accepted: {slug}")
    elif rating == "reject":
        save_feedback(original_input, original_input, source="rejected")
        print(f"Recipe rejected: {slug}")

def post_feedback(slug, rating):
    """Post feedback signal to Prometheus monitoring endpoint."""
    response = requests.post(
        f"{MEALIE_URL}/api/ml/feedback",
        params={"slug": slug, "rating": rating}
    )
    if response.status_code == 200:
        print(f"Feedback posted: slug={slug} rating={rating}")
    else:
        print(f"Failed to post feedback: {response.status_code}")

def main():
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
                add_note_to_recipe(slug, original_input, cleaned)
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
