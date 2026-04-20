# Argo Workflow Resources

This directory holds the workflow resources that the GitOps stack applies into
`recipe-scraper-platform`.

Preferred operator path:

- Submit manual training runs from the Argo Workflows dashboard UI using the
  `recipe-model-training` template.
- Use `manual-training-workflow.yaml` only as a CLI submission example or as a
  reference for the default arguments exposed in the UI.

Files:

- `rbac.yaml`: service account and minimum executor RBAC for workflow pods
- `workflow-template-training.yaml`: reusable training `WorkflowTemplate`
- `cronworkflow-retraining.yaml`: scheduled retraining `CronWorkflow`
- `manual-training-workflow.yaml`: on-demand workflow submission example
- `kustomization.yaml`: the GitOps entrypoint ArgoCD syncs for long-lived workflow resources

Important current default:

- Training workflows default to `data-pvc-gpu`, the PVC mounted on
  `gpu-node`, so training pods can access the dataset from the same node where
  the GPU is scheduled.
- The `CronWorkflow` is suspended by default because the only cluster GPU is
  already reserved by Triton on `gpu-node`. Unsuspend it only after GPU
  capacity or serving allocation changes.

GitOps note:

- `manual-training-workflow.yaml` is intentionally not included in
  `kustomization.yaml`. It uses `generateName` and is meant to be submitted
  manually, not continuously reconciled by ArgoCD.
- `cronworkflow-retraining.yaml` is the long-lived scheduled entrypoint. It
  reuses the same `recipe-model-training` `WorkflowTemplate` as manual runs so
  both paths stay aligned.
