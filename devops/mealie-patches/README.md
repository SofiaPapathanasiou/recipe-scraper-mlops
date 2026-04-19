# Mealie Patches

Changes applied to Mealie v3.16.0 (ghcr.io/mealie-recipes/mealie) for Triton inference integration.

## Files
- `triton_cleaner.py` → add to `mealie/services/scraper/`
- `recipe_scraper.py` → replace `mealie/services/scraper/recipe_scraper.py`
- `pyproject.toml` → adds `tritonclient==2.50.0` dependency

## Custom image
`graceritamcgrath/mealie-custom:latest`
