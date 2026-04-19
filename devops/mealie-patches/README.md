# Mealie Patches
Changes applied to Mealie v3.16.0 (ghcr.io/mealie-recipes/mealie) for Triton inference and feedback monitoring integration.

## Files
- `triton_cleaner.py` → copy to `mealie/services/scraper/triton_cleaner.py`
- `recipe_scraper.py` → replace `mealie/services/scraper/recipe_scraper.py`
- `recipe_cleaning_feedback.py` → copy to `mealie/routes/recipe_cleaning_feedback.py`
- `__init__.py` → replace `mealie/routes/__init__.py`
- `pyproject.toml` → replace `pyproject.toml` (adds `tritonclient==2.50.0` and `prometheus_client>=0.20.0` dependencies)

## Applying patches
```bash
cp triton_cleaner.py <mealie-src>/mealie/services/scraper/
cp recipe_scraper.py <mealie-src>/mealie/services/scraper/
cp recipe_cleaning_feedback.py <mealie-src>/mealie/routes/
cp __init__.py <mealie-src>/mealie/routes/
cp pyproject.toml <mealie-src>/
```

## Rebuilding the custom image
```bash
cd <mealie-src>
docker build -t graceritamcgrath/mealie-custom:v1.2 --target production -f docker/Dockerfile .
docker push graceritamcgrath/mealie-custom:v1.2
```

## Custom image
`graceritamcgrath/mealie-custom:v1.2`

## Environment variables required
- `TRITON_SERVER_URL` — URL of the Triton inference server e.g. `http://recipe-triton-svc:8000`
- `TRITON_MODEL_NAME` — model name e.g. `recipe_model`
- `MODEL_VERSION` — model version for Prometheus labels e.g. `v1.2`
- `API_DOCS` — set to `true` to enable Swagger UI at `/docs`
