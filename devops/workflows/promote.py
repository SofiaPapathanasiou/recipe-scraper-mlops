import requests
import yaml
import os
import subprocess

MLFLOW_URL = os.environ.get("MLFLOW_URL", "http://mlflow.recipe-scraper-platform.svc.cluster.local:8000")
EXPERIMENT_IDS = ["1", "2"]
PROMOTION_THRESHOLD = 0.45

def get_best_run():
    best_run = None
    best_score = 0
    for exp_id in EXPERIMENT_IDS:
        resp = requests.post(
            f"{MLFLOW_URL}/api/2.0/mlflow/runs/search",
            json={
                "experiment_ids": [exp_id],
                "max_results": 20,
                "order_by": ["metrics.best_eval_rougeL DESC"]
            }
        )
        runs = resp.json().get("runs", [])
        for run in runs:
            metrics = {m["key"]: m["value"] for m in run["data"].get("metrics", [])}
            score = metrics.get("best_eval_rougeL", 0)
            if run["info"]["status"] == "FINISHED" and score > best_score:
                best_score = score
                best_run = run
    return best_run, best_score

best_run, best_score = get_best_run()

if not best_run:
    print("No finished runs found")
    exit(0)

if best_score < PROMOTION_THRESHOLD:
    print(f"Best score {best_score:.4f} below threshold, not promoting")
    exit(0)

run_id = best_run["info"]["run_id"]
exp_id = best_run["info"]["experiment_id"]
model_prefix = f"{exp_id}/{run_id}/artifacts/checkpoints/best"

print(f"Best run: {run_id}, rougeL: {best_score:.4f}")

with open("/workspace/devops/k8s/platform/values.yaml") as f:
    values = yaml.safe_load(f)

current_prefix = values.get("triton", {}).get("modelPrefix", "")
if current_prefix == model_prefix:
    print("Already on best model, nothing to do")
    exit(0)

values["triton"]["modelPrefix"] = model_prefix
values["triton"]["modelVersion"] = run_id[:8]

with open("/workspace/devops/k8s/platform/values.yaml", "w") as f:
    yaml.dump(values, f, default_flow_style=False)

subprocess.run(["git", "config", "user.email", "grace.mcgrath@nyu.edu"], cwd="/workspace", check=True)
subprocess.run(["git", "config", "user.name", "Grace McGrath"], cwd="/workspace", check=True)
subprocess.run(["git", "add", "devops/k8s/platform/values.yaml"], cwd="/workspace", check=True)
subprocess.run(["git", "commit", "-m", f"auto: promote model {run_id[:8]} rougeL={best_score:.4f}"], cwd="/workspace", check=True)
subprocess.run(["git", "push"], cwd="/workspace", check=True)
print("Pushed to Git — ArgoCD will sync automatically")
