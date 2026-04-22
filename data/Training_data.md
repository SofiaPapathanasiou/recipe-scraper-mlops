# Loading Training Data from Object Store

This file explains how to get the training JSONL files from the object store and mount them to the data PVC.

## Prerequisites

- kubectl access to the cluster
- Object store credentials sourced

## Step 1: Source credentials

```bash
source ~/app-cred-proj22_model_app_credential-openrc.sh
```

## Step 2: Run the loader script

```bash
bash ~/recipe-scraper-mlops/data/scripts/load_training_data.sh
```

This script will:
1. Download `train.jsonl` and `eval.jsonl` from `ObjStore_proj22`
2. Spin up a temporary pod with the PVC mounted
3. Copy the files to `/data/training/` in the PVC
4. Clean up the pod

## Manual Steps (if script fails)

```bash
# Download files
swift download ObjStore_proj22 training/cleaner/v_20260401/train.jsonl -o /tmp/train.jsonl
swift download ObjStore_proj22 training/cleaner/v_20260401/eval.jsonl -o /tmp/eval.jsonl

# Start loader pod
kubectl run data-loader --image=192.168.1.11:5000/recipe-scraper-data:latest \
  --restart=Never \
  --overrides='{"spec":{"volumes":[{"name":"data","persistentVolumeClaim":{"claimName":"data-pvc-gpu"}}],"containers":[{"name":"data-loader","image":"192.168.1.11:5000/recipe-scraper-data:latest","command":["sleep","3600"],"volumeMounts":[{"name":"data","mountPath":"/data"}]}]}}' \
  -n recipe-scraper-platform

# Copy files
kubectl exec -n recipe-scraper-platform data-loader -- mkdir -p /data/training
kubectl cp /tmp/train.jsonl recipe-scraper-platform/data-loader:/data/training/train.jsonl
kubectl cp /tmp/eval.jsonl recipe-scraper-platform/data-loader:/data/training/eval.jsonl

# Verify
kubectl exec -n recipe-scraper-platform data-loader -- ls -lh /data/training/

# Cleanup
kubectl delete pod data-loader -n recipe-scraper-platform
```

## File Locations in Object Store

| File | Object Store Path |
|---|---|
| train.jsonl (313MB) | `training/cleaner/v_20260401/train.jsonl` |
| eval.jsonl (34MB) | `training/cleaner/v_20260401/eval.jsonl` |
| aligned_pairs.jsonl (250MB) | `processed/cleaner/pairs/v_20260401/aligned_pairs.jsonl` |
| augmented_pairs.jsonl | `processed/cleaner/augment/v_20260401/augmented_pairs.jsonl` |

## File Location in PVC

Once loaded, files will be at:
```
/data/training/train.jsonl
/data/training/eval.jsonl
```
```
