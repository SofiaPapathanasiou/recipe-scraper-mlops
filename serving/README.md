# Serving Layer — README

## Overview
The serving layer consists of:
- **Triton Inference Server** — serves the T5 recipe cleaning model
- **Mealie integration** — sends scraped recipes to Triton for cleaning
- **Feedback endpoint** — collects user rejection signals
- **Prometheus + Grafana** — monitors all three signal types
- **Argo Rollouts** — manages canary deployments with automated promotion/rollback
- **MLflow watcher** — detects better models and triggers deployments
- **mealie_cleaner.py** — polls Mealie for recipe edits and posts feedback

---

## Prerequisites
- Kubernetes cluster running (see `devops/ansible/`)
- Chameleon Cloud openrc credentials available
- GitHub personal access token with repo scope

---

## 1. Install Argo Rollouts

    kubectl create namespace argo-rollouts
    kubectl apply -n argo-rollouts -f https://github.com/argoproj/argo-rollouts/releases/latest/download/install.yaml

### Argo Rollouts kubectl plugin

    curl -LO https://github.com/argoproj/argo-rollouts/releases/latest/download/kubectl-argo-rollouts-linux-amd64
    chmod +x kubectl-argo-rollouts-linux-amd64
    sudo mv kubectl-argo-rollouts-linux-amd64 /usr/local/bin/kubectl-argo-rollouts

---

## 2. Install Argo Workflows

    kubectl create namespace argo-workflows
    kubectl apply -n argo-workflows -f https://github.com/argoproj/argo-workflows/releases/latest/download/install.yaml

### Argo Workflows CLI

    curl -sLO https://github.com/argoproj/argo-workflows/releases/download/v3.5.4/argo-linux-amd64.gz
    gunzip argo-linux-amd64.gz
    chmod +x argo-linux-amd64
    sudo mv argo-linux-amd64 /usr/local/bin/argo

---

## 3. Configure ArgoCD

### Install ArgoCD

    kubectl create namespace argocd
    kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

### Wait for ArgoCD to be ready

    kubectl wait --for=condition=available deployment/argocd-server -n argocd --timeout=120s

### Get initial admin password

    kubectl get secret argocd-initial-admin-secret -n argocd \
      -o jsonpath='{.data.password}' | base64 -d

### Apply the ArgoCD application

    kubectl apply -f devops/k8s/platform/templates/argocd-application.yaml

### Verify application is synced

    kubectl get applications -n argocd

---

## 4. Create Required Secrets

    kubectl create secret generic triton-openrc \
      --from-literal=openrc="$(cat /path/to/openrc)" \
      -n recipe-scraper-platform

    kubectl create secret generic mlflow-s3-credentials \
      --from-literal=AWS_ACCESS_KEY_ID=<key> \
      --from-literal=AWS_SECRET_ACCESS_KEY=<secret> \
      -n recipe-scraper-platform

    kubectl create secret generic git-credentials \
      --from-literal=token='<github-pat>' \
      -n recipe-scraper-platform

---

## 5. Deploy All Environments via ArgoCD

    kubectl apply -f devops/argocd/project.yaml
    kubectl apply -f devops/argocd/applications/

This deploys all environments and services via ArgoCD:
- **platform** — Triton, Prometheus, Grafana, MLflow
- **staging** — Mealie staging environment
- **canary** — Mealie canary environment
- **production** — Mealie production environment
- **workflows-stack** — Argo Workflows
- **workflows-jobs** — MLflow watcher CronWorkflow

ArgoCD will keep all environments in sync with Git automatically. Verify:

    kubectl get applications -n argocd

---

## 6. Deploy MLflow Watcher CronWorkflow

    kubectl apply -f devops/workflows/mlflow-model-promoter.yaml

---

## 7. Mealie Integration

### Apply patches to Mealie source

    cp devops/mealie-patches/triton_cleaner.py <mealie-src>/mealie/services/scraper/
    cp devops/mealie-patches/recipe_scraper.py <mealie-src>/mealie/services/scraper/
    cp devops/mealie-patches/recipe_cleaning_feedback.py <mealie-src>/mealie/routes/
    cp devops/mealie-patches/__init__.py <mealie-src>/mealie/routes/
    cp devops/mealie-patches/pyproject.toml <mealie-src>/

### Build and push custom Mealie image

    cd <mealie-src>
    docker build -t graceritamcgrath/mealie-custom:v1.2 \
      --target production \
      -f docker/Dockerfile .
    docker push graceritamcgrath/mealie-custom:v1.2

### Required environment variables for Mealie

    TRITON_SERVER_URL=http://recipe-triton-svc:8000
    TRITON_MODEL_NAME=recipe_model
    MODEL_VERSION=<version>
    API_DOCS=true

---

## 8. Running mealie_cleaner.py

Polls Mealie every 30 seconds for new recipes and user edits.

    export MEALIE_URL=http://<mealie-host>:<port>
    export MEALIE_TOKEN=<mealie-api-token>
    python3 serving/mealie_cleaner.py

The script:
- Sends new recipes to Triton for cleaning
- Adds original and cleaned versions as notes in Mealie
- Detects when users edit a cleaned recipe (negative feedback signal)
- Posts rejection to /api/ml/feedback endpoint
- Saves feedback pairs to ObjStore_proj22 Swift container for retraining

---

## Monitoring

### Check Prometheus targets

    curl http://<node-ip>:30902/api/v1/targets | python3 -m json.tool | grep -E "job|health"

### Check feedback metrics

    curl http://<mealie-host>:<port>/api/ml/metrics | grep feedback

### Submit manual feedback

    curl -X POST "http://<mealie-host>:<port>/api/ml/feedback?slug=<recipe-slug>&rating=reject"
    curl -X POST "http://<mealie-host>:<port>/api/ml/feedback?slug=<recipe-slug>&rating=accept"

### Access Grafana

    http://<node-ip>:30903
    Default credentials: admin/admin

---

## Argo Rollouts Operations

### Check rollout status

    kubectl argo rollouts get rollout recipe-triton -n recipe-scraper-platform

### Watch rollout in real time

    kubectl argo rollouts get rollout recipe-triton -n recipe-scraper-platform --watch

### Force promote (testing only)

    kubectl argo rollouts promote recipe-triton -n recipe-scraper-platform --full

### Abort and rollback

    kubectl argo rollouts abort recipe-triton -n recipe-scraper-platform

### Retry after abort

    kubectl argo rollouts retry rollout recipe-triton -n recipe-scraper-platform

---

## MLflow Watcher Operations

### Check CronWorkflow

    kubectl get cronworkflow -n recipe-scraper-platform

### Check recent runs

    kubectl get workflows -n recipe-scraper-platform | grep promoter

### View logs of latest run

    argo logs <workflow-name> -n recipe-scraper-platform

### Trigger manual run

    argo submit --from cronworkflow/mlflow-model-promoter \
      -n recipe-scraper-platform \
      --name mlflow-promoter-manual

---

## Promotion/Rollback Thresholds

| Signal | Threshold | Justification |
|--------|-----------|---------------|
| User feedback rejection rate | < 25% | More than 1 in 4 users rejecting output indicates model regression |
| Triton error rate | < 5% | Above 5% means users are getting errors instead of cleaned recipes |
| Inference latency | < 10s | Recipe cleaning is synchronous — above 10s degrades UX unacceptably |
| Feedback window | 12h | Feedback collected every 6h, 12h window spans two cycles to avoid false positives |
| Failure limit | 2 consecutive | Single bad batch should not trigger rollback |

---

## Key Files

| File | Purpose |
|------|---------|
| devops/k8s/platform/templates/triton-rollout.yaml | Argo Rollouts canary strategy |
| devops/k8s/platform/templates/triton-analysis-template.yaml | Promotion/rollback triggers |
| devops/k8s/platform/templates/prometheus.yaml | Prometheus scrape config |
| devops/k8s/platform/values.yaml | Platform configuration including modelPrefix |
| devops/workflows/mlflow-model-promoter.yaml | MLflow watcher CronWorkflow |
| devops/workflows/promote.py | Model promotion logic |
| devops/mealie-patches/ | Mealie source code modifications |
| serving/mealie_cleaner.py | Recipe polling and feedback collection |