cle# recipe-scraper-mlops

This project implements an end-to-end ML pipeline that , built around the oen-source recipe manager Mealie (https://github.com/mealie-recipes/mealie). We introduce an ML feature that takes scraped recipes from the internet that may contain formatting issues or mistakes and produces neatly formatted and corrected recipes for users to save in their Mealie app.
This is a course project for *Machine Learning Systems Engineering & Operations (ECE-GY 9183, Spring 2026, NYU Tandon)*.

## Project structure
```text
recipe-scraper-mlops/ 
├── devops/ # Docker, deployment, and infrastructure setup 
├── serving/ # Inference API (FastAPI) 
├── training/ # Model training pipeline 
├── data/ # Scraping and data processing scripts 
├── shared/ # Example inputs/outputs and shared schemas 
└── README.md
```

## Current Status
Initial implementation in progress. We are currently setting up the core components for data processing, model training, inference serving, and deployment.

## Argo Workflows

The Argo Workflows control plane is managed from
[`devops/k8s/argo-workflows`](/home/cc/recipe-scraper-mlops/devops/k8s/argo-workflows/kustomization.yaml:1)
and applied through a dedicated Argo CD app.

Training workflow resources live under
[`devops/workflows`](/home/cc/recipe-scraper-mlops/devops/workflows/README.md:1).

Manual model training should be launched from the Argo Workflows dashboard UI
using the `recipe-model-training` template. The Helm `trainingJob` remains in
the platform chart only as a simple smoke-test or fallback batch job.

## Team
- **Training:** Yathin Reddy Duvuru
- **Serving:** Grace McGrath
- **Data:** Shruti Sridhar
- **DevOps/Platform:** Sofia Papathanasiou
