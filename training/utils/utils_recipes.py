from typing import Any

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset

from .utils_core import resolve_model_source

MOCK_RECIPE_BLUEPRINTS = [
    {
        "title": "Lemon Garlic Chicken Pasta",
        "yield": "4 servings",
        "prep_time": "15 minutes",
        "cook_time": "25 minutes",
        "ingredients": [
            "8 ounces spaghetti",
            "1 pound boneless chicken breast, diced",
            "2 tablespoons olive oil",
            "3 cloves garlic, minced",
            "1 lemon, zested and juiced",
            "1/2 cup grated parmesan",
            "2 cups baby spinach",
            "1/2 teaspoon kosher salt",
            "1/4 teaspoon black pepper",
        ],
        "instructions": [
            "Cook the spaghetti in salted water until al dente, then reserve 1/2 cup pasta water and drain.",
            "Heat the olive oil in a skillet and cook the chicken until browned and cooked through.",
            "Stir in the garlic, lemon zest, lemon juice, salt, and pepper, then cook for 1 minute.",
            "Add the spinach and cooked pasta, tossing with the parmesan and reserved pasta water until glossy.",
        ],
        "notes": "Serve with extra parmesan and lemon wedges.",
    },
    {
        "title": "One-Bowl Banana Muffins",
        "yield": "12 muffins",
        "prep_time": "10 minutes",
        "cook_time": "20 minutes",
        "ingredients": [
            "3 ripe bananas, mashed",
            "1/2 cup melted butter",
            "1/2 cup brown sugar",
            "1 egg",
            "1 teaspoon vanilla extract",
            "1 1/2 cups all-purpose flour",
            "1 teaspoon baking soda",
            "1/2 teaspoon cinnamon",
            "1/4 teaspoon salt",
        ],
        "instructions": [
            "Preheat the oven to 350F and line a 12-cup muffin tin.",
            "Whisk the bananas, melted butter, brown sugar, egg, and vanilla until smooth.",
            "Fold in the flour, baking soda, cinnamon, and salt just until no dry streaks remain.",
            "Divide the batter between the cups and bake until the tops spring back lightly.",
        ],
        "notes": "A few chocolate chips or chopped walnuts can be folded in with the dry ingredients.",
    },
    {
        "title": "Sheet Pan Sausage and Vegetables",
        "yield": "4 servings",
        "prep_time": "15 minutes",
        "cook_time": "30 minutes",
        "ingredients": [
            "12 ounces smoked sausage, sliced",
            "1 red bell pepper, chopped",
            "1 zucchini, sliced",
            "1 small red onion, cut into wedges",
            "12 ounces baby potatoes, halved",
            "2 tablespoons olive oil",
            "1 teaspoon paprika",
            "1/2 teaspoon garlic powder",
            "1/2 teaspoon salt",
        ],
        "instructions": [
            "Heat the oven to 425F and line a sheet pan with parchment.",
            "Toss the sausage, bell pepper, zucchini, onion, and potatoes with the olive oil and seasonings.",
            "Spread everything in an even layer and roast until the vegetables are tender and caramelized.",
            "Stir halfway through cooking so the potatoes brown on multiple sides.",
        ],
        "notes": "Finish with chopped parsley or a squeeze of lemon if you have it.",
    },
    {
        "title": "Tomato Basil Soup",
        "yield": "6 servings",
        "prep_time": "10 minutes",
        "cook_time": "35 minutes",
        "ingredients": [
            "2 tablespoons butter",
            "1 yellow onion, diced",
            "3 cloves garlic, minced",
            "2 tablespoons tomato paste",
            "2 cans crushed tomatoes",
            "2 cups vegetable broth",
            "1/3 cup heavy cream",
            "1/4 cup basil leaves",
            "3/4 teaspoon salt",
        ],
        "instructions": [
            "Melt the butter in a soup pot and cook the onion until softened.",
            "Add the garlic and tomato paste, stirring until fragrant and slightly darkened.",
            "Pour in the crushed tomatoes and broth, then simmer for 25 minutes.",
            "Blend until smooth, then stir in the cream, basil, and salt before serving.",
        ],
        "notes": "A grilled cheese sandwich makes a good side.",
    },
    {
        "title": "Honey Soy Salmon Bowls",
        "yield": "4 servings",
        "prep_time": "20 minutes",
        "cook_time": "15 minutes",
        "ingredients": [
            "4 salmon fillets",
            "3 tablespoons soy sauce",
            "2 tablespoons honey",
            "1 tablespoon rice vinegar",
            "1 teaspoon grated ginger",
            "2 cups cooked jasmine rice",
            "1 cucumber, sliced",
            "1 avocado, sliced",
            "2 green onions, thinly sliced",
        ],
        "instructions": [
            "Whisk the soy sauce, honey, rice vinegar, and ginger in a shallow dish.",
            "Marinate the salmon for 10 minutes while the oven heats to 400F.",
            "Bake the salmon until flaky, brushing with the leftover marinade halfway through.",
            "Build bowls with rice, cucumber, avocado, and salmon, then top with green onions.",
        ],
        "notes": "Sesame seeds add crunch if you want a garnish.",
    },
    {
        "title": "Creamy Chickpea Curry",
        "yield": "4 servings",
        "prep_time": "10 minutes",
        "cook_time": "25 minutes",
        "ingredients": [
            "1 tablespoon coconut oil",
            "1 onion, diced",
            "2 cloves garlic, minced",
            "1 tablespoon grated ginger",
            "2 tablespoons curry powder",
            "2 cans chickpeas, drained",
            "1 can coconut milk",
            "1 cup diced tomatoes",
            "1/2 teaspoon salt",
        ],
        "instructions": [
            "Warm the coconut oil in a skillet and cook the onion until translucent.",
            "Add the garlic, ginger, and curry powder, stirring until fragrant.",
            "Stir in the chickpeas, coconut milk, tomatoes, and salt.",
            "Simmer until slightly thickened, then serve over rice or with naan.",
        ],
        "notes": "Baby spinach can be stirred in during the last 2 minutes of cooking.",
    },
    {
        "title": "Classic Pancakes",
        "yield": "10 pancakes",
        "prep_time": "10 minutes",
        "cook_time": "15 minutes",
        "ingredients": [
            "1 1/2 cups all-purpose flour",
            "2 tablespoons sugar",
            "2 teaspoons baking powder",
            "1/4 teaspoon salt",
            "1 1/4 cups milk",
            "1 egg",
            "2 tablespoons melted butter",
            "1 teaspoon vanilla extract",
        ],
        "instructions": [
            "Whisk the flour, sugar, baking powder, and salt in a bowl.",
            "In a second bowl, whisk the milk, egg, melted butter, and vanilla.",
            "Pour the wet mixture into the dry ingredients and stir just until combined.",
            "Cook 1/4-cup portions on a greased skillet until bubbles form and the pancakes are golden on both sides.",
        ],
        "notes": "Do not overmix or the pancakes will be tough.",
    },
    {
        "title": "Roasted Broccoli Mac and Cheese",
        "yield": "6 servings",
        "prep_time": "15 minutes",
        "cook_time": "35 minutes",
        "ingredients": [
            "12 ounces elbow macaroni",
            "1 head broccoli, cut into florets",
            "2 tablespoons olive oil",
            "3 tablespoons butter",
            "3 tablespoons flour",
            "2 cups milk",
            "2 cups shredded cheddar cheese",
            "1/2 teaspoon salt",
            "1/4 teaspoon mustard powder",
        ],
        "instructions": [
            "Roast the broccoli with the olive oil at 425F until crisp-tender.",
            "Cook the macaroni until just shy of al dente and drain.",
            "Make a roux with the butter and flour, whisk in the milk, then melt in the cheddar, salt, and mustard powder.",
            "Fold in the macaroni and broccoli, then bake until bubbling if you want a casserole-style finish.",
        ],
        "notes": "For a stovetop version, skip the final bake and serve immediately.",
    },
]


def format_mock_recipe(recipe: dict[str, Any]) -> str:
    ingredient_lines = "\n".join(f"- {item}" for item in recipe["ingredients"])
    instruction_lines = "\n".join(f"{index}. {step}" for index, step in enumerate(recipe["instructions"], start=1))
    return (
        f"Title: {recipe['title']}\n"
        f"Yield: {recipe['yield']}\n"
        f"Prep time: {recipe['prep_time']}\n"
        f"Cook time: {recipe['cook_time']}\n"
        "Ingredients:\n"
        f"{ingredient_lines}\n"
        "Instructions:\n"
        f"{instruction_lines}\n"
        f"Notes: {recipe['notes']}"
    )


def apply_word_level_recipe_noise(text: str) -> str:
    replacements = {
        "Title:": "Ttle:",
        "Yield:": "Yeild:",
        "Prep time:": "Prep tm:",
        "Cook time:": "Cook tm:",
        "Ingredients:": "Ingrednts:",
        "Instructions:": "Instrctions:",
        "Notes:": "Note:",
        "ounces": "oz",
        "tablespoons": "tbsp",
        "teaspoons": "tsp",
        "minutes": "mins",
        "boneless": "bonless",
        "chicken": "chikcen",
        "parmesan": "parmasan",
        "vegetable": "vegtable",
        "broccoli": "brocoli",
        "through": "thru",
        "until": "till",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text.replace(", then", " then").replace(", and", " and")


def collapse_recipe_sections(text: str) -> str:
    collapsed = text.replace("\n- ", ", ").replace("\n", " | ")
    collapsed = collapsed.replace("Instructions: | 1. ", "Directions: ")
    collapsed = collapsed.replace(" | 2. ", " Next, ")
    collapsed = collapsed.replace(" | 3. ", " Then ")
    collapsed = collapsed.replace(" | 4. ", " Finally ")
    collapsed = collapsed.replace(" | Notes: ", " | Note ")
    return collapsed


def remove_recipe_punctuation(text: str) -> str:
    stripped = text.replace(":", "").replace(",", "").replace(".", "")
    stripped = stripped.replace("1/2", "1-2").replace("1/4", "1-4")
    stripped = stripped.replace("350F", "350 f").replace("425F", "425 f").replace("400F", "400 f")
    return stripped


def add_shorthand_recipe_noise(text: str) -> str:
    replacements = {
        "Preheat": "Pre-heat",
        "Whisk": "Mix up",
        "stir": "mix",
        "until": "til",
        "with the": "w/",
        "and": "&",
        "because": "bc",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text.replace("Instructions:", "Steps:").replace("Notes:", "Tips:")


def build_mock_recipe_pairs() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for recipe in MOCK_RECIPE_BLUEPRINTS:
        target = format_mock_recipe(recipe)
        pairs.extend(
            [
                (apply_word_level_recipe_noise(target), target),
                (collapse_recipe_sections(apply_word_level_recipe_noise(target)), target),
                (remove_recipe_punctuation(target), target),
                (add_shorthand_recipe_noise(collapse_recipe_sections(target)), target),
            ]
        )
    return pairs


CORRUPTIONS = build_mock_recipe_pairs()

def make_split(pairs: list[tuple[str, str]], target_size: int) -> list[tuple[str, str]]:
    pool = pairs * (target_size // len(pairs) + 1)
    generator = torch.Generator().manual_seed(0)
    order = torch.randperm(len(pool), generator=generator).tolist()
    shuffled = [pool[index] for index in order]
    return shuffled[:target_size]


class RecipeTextDataset(Dataset):
    def __init__(
        self,
        pairs: list[tuple[str, str]],
        tokenizer: Any,
        task_prefix: str,
        max_input_length: int,
        max_target_length: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.inputs = [task_prefix + corrupted for corrupted, _ in pairs]
        self.targets = [target for _, target in pairs]
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        model_inputs = self.tokenizer(
            self.inputs[index],
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = self.tokenizer(
            self.targets[index],
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids
        # T5 ignores labels set to -100 when computing the loss.
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": model_inputs.input_ids.squeeze(0),
            "attention_mask": model_inputs.attention_mask.squeeze(0),
            "labels": labels.squeeze(0),
        }


class MockRecipeDataset(RecipeTextDataset):
    def __init__(self, tokenizer: Any, cfg: dict[str, Any], split: str) -> None:
        if split == "train":
            size = cfg["data"]["mock_train_size"]
        elif split == "val":
            size = cfg["data"]["mock_val_size"]
        else:
            raise ValueError(f"Unsupported split {split!r}")

        super().__init__(
            pairs=make_split(CORRUPTIONS, size),
            tokenizer=tokenizer,
            task_prefix=cfg["model"]["task_prefix"],
            max_input_length=cfg["tokenization"]["max_input_length"],
            max_target_length=cfg["tokenization"]["max_target_length"],
        )


def build_datasets(cfg: dict[str, Any], tokenizer: Any) -> tuple[Dataset, Dataset]:
    if cfg["data"]["source"] != "mock":
        raise ValueError(
            f"Unsupported data.source {cfg['data']['source']!r}; only 'mock' is supported in this setup."
        )
    return MockRecipeDataset(tokenizer, cfg, "train"), MockRecipeDataset(tokenizer, cfg, "val")


def build_dataloaders(
    cfg: dict[str, Any], tokenizer: Any, accelerator: Accelerator
) -> tuple[DataLoader, DataLoader, Dataset, Dataset]:
    train_dataset, val_dataset = build_datasets(cfg, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["per_device_train_batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=accelerator.device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["per_device_eval_batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=accelerator.device.type == "cuda",
    )
    return train_loader, val_loader, train_dataset, val_dataset


def summarize_training_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_name": cfg["model"]["name"],
        "model_source": resolve_model_source(cfg),
        "data_source": cfg["data"]["source"],
        "num_epochs": cfg["training"]["num_epochs"],
        "train_batch_size": cfg["training"]["per_device_train_batch_size"],
        "eval_batch_size": cfg["training"]["per_device_eval_batch_size"],
        "gradient_accumulation_steps": cfg["training"]["gradient_accumulation_steps"],
        "learning_rate": cfg["training"]["learning_rate"],
        "warmup_ratio": cfg["training"]["warmup_ratio"],
        "checkpoint_dir": cfg["checkpointing"]["checkpoint_dir"],
        "tracking_uri": cfg["mlflow"]["tracking_uri"],
    }


def summarize_batch(batch: dict[str, torch.Tensor]) -> str:
    parts: list[str] = []
    for key, value in batch.items():
        shape = tuple(value.shape)
        parts.append(f"{key}=shape{shape},dtype={value.dtype}")
    return "; ".join(parts)
