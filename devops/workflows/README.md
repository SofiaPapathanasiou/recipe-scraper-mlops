# Argo Workflow Resources

This directory holds the workflow resources that the GitOps stack applies into
`recipe-scraper-platform`.

Files:

- `rbac.yaml`: service account and minimum executor RBAC for workflow pods
- `workflow-template-training.yaml`: reusable training `WorkflowTemplate`
- `cronworkflow-retraining.yaml`: scheduled retraining `CronWorkflow`
- `manual-training-workflow.yaml`: on-demand workflow submission example

Important current default:

- The `CronWorkflow` is suspended by default because the only cluster GPU is
  already reserved by Triton on `gpu-node`. Unsuspend it only after GPU
  capacity or serving allocation changes.
