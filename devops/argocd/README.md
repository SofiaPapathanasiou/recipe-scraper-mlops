# ArgoCD Bootstrap Manifests

This directory is the Git-tracked source of truth for ArgoCD application
registration in this repository.

Bootstrap once:

```bash
kubectl apply -k devops/argocd
```

After that, ArgoCD should reconcile the applications declared here instead of
relying on imperative `argocd app create` commands.
