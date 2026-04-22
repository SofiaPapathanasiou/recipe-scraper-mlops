# recipe-scraper-mlops

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

## Recreate the environment (repo-root notebooks)

Use the repo-root setup notebooks in order to rebuild infrastructure and cluster state end-to-end.

1. From your local machine, run `01_terraform_provisioning.ipynb` from the repo root.
2. Complete Terraform validation in the notebook (`plan`/`apply` checks and expected outputs).
3. SSH into `node1`, clone this repo there, and `cd` to the repo root on `node1`.
4. Run `02_node1_cluster_bootstrap.ipynb` on `node1` phase by phase:
   - Ansible connectivity (`hello_host.yml`)
   - Pre-K8s preparation (`pre_k8s_configure.yml`)
   - Kubespray cluster creation (`cluster.yml`)
   - Post-K8s configuration (`post_k8s_configure.yml`)
   - Argo CD bootstrap apps (`argocd_bootstrap_apps.yml`)

5. (Optional) Submit a one-off manual training run via the Argo CLI using `03_manual_training_workflow_cli.ipynb`.
   - Purpose: wraps `argo submit` for `devops/workflows/manual-training-workflow.yaml` with a single place to override workflow parameters.
   - Prereqs: `argo` + `kubectl` installed and pointed at the cluster; the `recipe-model-training` `WorkflowTemplate` exists in the `recipe-scraper-platform` namespace.
   - Common overrides: `TRAINING_IMAGE` (cluster-local training image), `TRAIN_JSONL_PATH` / `EVAL_JSONL_PATH`, `NUM_PROCESSES`, `TRAIN_EXTRA_ARGS`, `MLFLOW_TRACKING_URI`.
   - If you prefer the UI: manual training can also be launched from the Argo Workflows dashboard using the `recipe-model-training` template.

### Required prerequisite before `pre_k8s`

Before running the `pre_k8s` phase/playbook, ensure the MLflow PostgreSQL persistent volume is mounted and ready on `node5` at `/mnt/mlflow_persist/postgres_data`. This must be in place before `devops/ansible/pre_k8s/pre_k8s_configure.yml` runs.

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
