cle# recipe-scraper-mlops

This project implements an end-to-end ML pipeline built around the open-source recipe manager Mealie (https://github.com/mealie-recipes/mealie). We introduce an ML feature that takes scraped recipes from the internet that may contain formatting issues or mistakes and produces neatly formatted and corrected recipes for users to save in their Mealie app.

This is a course project for *Machine Learning Systems Engineering & Operations (ECE-GY 9183, Spring 2026, NYU Tandon)*.

## Project structure

    recipe-scraper-mlops/
    ├── devops/                  # Kubernetes, Helm charts, Ansible, ArgoCD, workflows
    │   ├── ansible/             # Cluster provisioning
    │   ├── argocd/              # ArgoCD application configs
    │   ├── k8s/                 # Helm charts for platform, mealie, argo-workflows
    │   ├── mealie-patches/      # Mealie source code modifications
    │   └── workflows/           # Argo Workflows for MLflow model promotion
    ├── serving/                 # Serving layer — Triton, Mealie integration, monitoring
    │   ├── mealie_cleaner.py    # Recipe polling and feedback collection
    │   └── README.md            # Serving layer setup and operations guide
    ├── training/                # Model training pipeline
    ├── data/                    # Scraping and data processing scripts
    ├── shared/                  # Example inputs/outputs and shared schemas
    └── README.md

## Documentation

- **Serving layer** — setup, deployment, monitoring, and rollback: [serving/README.md](serving/README.md)

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
