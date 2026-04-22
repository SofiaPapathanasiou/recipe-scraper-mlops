# Argo CD + Kubernetes Ops Commands

Command cheat sheet for day-2 operations: check Argo CD apps, Kubernetes workload status, and Argo Workflows (including live logs).

## Conventions

- Set your namespace(s) once:

```bash
export NS_PLATFORM=recipe-scraper-platform
export NS_ARGOCD=argocd
export NS_WORKFLOWS=argo
```

- Handy aliases (optional):

```bash
alias k=kubectl
```

## Context, Cluster, Nodes

```bash
kubectl config get-contexts
kubectl config current-context
kubectl cluster-info

kubectl get nodes -o wide
kubectl top nodes
kubectl describe node <node>
```

## Namespaces

```bash
kubectl get ns
kubectl get all -n "$NS_PLATFORM"
```

## Argo CD (Kubernetes View)

Argo CD runs in-cluster; these commands inspect it through `kubectl`.

```bash
kubectl get pods -n "$NS_ARGOCD" -o wide
kubectl get svc -n "$NS_ARGOCD"
kubectl get ingress -n "$NS_ARGOCD"  # if you expose it via ingress

# Argo CD Applications (CRD)
kubectl get applications.argoproj.io -n "$NS_ARGOCD"
kubectl describe application.argoproj.io/<app> -n "$NS_ARGOCD"

# See application conditions/health/sync details in YAML
kubectl get application.argoproj.io/<app> -n "$NS_ARGOCD" -o yaml

# What Argo CD deployed for a given app (helpful to spot drift)
kubectl get application.argoproj.io/<app> -n "$NS_ARGOCD" -o jsonpath='{.status.resources[*].kind}'
kubectl get application.argoproj.io/<app> -n "$NS_ARGOCD" -o jsonpath='{.status.resources[*].name}'

# Argo CD controller logs
kubectl logs -n "$NS_ARGOCD" deploy/argocd-application-controller --since=30m
kubectl logs -n "$NS_ARGOCD" deploy/argocd-repo-server --since=30m
kubectl logs -n "$NS_ARGOCD" deploy/argocd-server --since=30m
```

## Argo CD (CLI)

Use this when you have `argocd` installed and can reach the Argo CD API.

```bash
argocd version
argocd login <argocd-host>

argocd app list
argocd app get <app>

# Current diff between desired and live
argocd app diff <app>

# Sync operations
argocd app sync <app>
argocd app wait <app> --health --sync --timeout 600

# App history / rollbacks
argocd app history <app>
argocd app rollback <app> <id>
```

## Pods, Deployments, StatefulSets, DaemonSets

```bash
kubectl get pods -n "$NS_PLATFORM" -o wide
kubectl describe pod -n "$NS_PLATFORM" <pod>

kubectl get deploy -n "$NS_PLATFORM"
kubectl describe deploy -n "$NS_PLATFORM" <deploy>
kubectl rollout status deploy/<deploy> -n "$NS_PLATFORM"
kubectl rollout history deploy/<deploy> -n "$NS_PLATFORM"
kubectl rollout restart deploy/<deploy> -n "$NS_PLATFORM"

kubectl get sts -n "$NS_PLATFORM"
kubectl rollout status sts/<sts> -n "$NS_PLATFORM"

kubectl get ds -n "$NS_PLATFORM"
kubectl rollout status ds/<ds> -n "$NS_PLATFORM"
```

## Services, Endpoints, Ingress

```bash
kubectl get svc -n "$NS_PLATFORM" -o wide
kubectl describe svc -n "$NS_PLATFORM" <svc>

kubectl get endpoints -n "$NS_PLATFORM" <svc>
kubectl get endpointSlices -n "$NS_PLATFORM" -l kubernetes.io/service-name=<svc>

kubectl get ingress -n "$NS_PLATFORM"
kubectl describe ingress -n "$NS_PLATFORM" <ingress>
```

## Events (Debugging)

```bash
kubectl get events -n "$NS_PLATFORM" --sort-by=.metadata.creationTimestamp
kubectl get events -n "$NS_PLATFORM" --field-selector type=Warning --sort-by=.metadata.creationTimestamp

# Narrow to a single object
kubectl get events -n "$NS_PLATFORM" --field-selector involvedObject.name=<name> --sort-by=.metadata.creationTimestamp
```

## Live Logs (Pods)

```bash
# Single container
kubectl logs -n "$NS_PLATFORM" <pod> -f

# If the pod has multiple containers
kubectl logs -n "$NS_PLATFORM" <pod> -c <container> -f

# Previous container (crash-loop debugging)
kubectl logs -n "$NS_PLATFORM" <pod> -c <container> --previous

# Follow logs for a whole deployment (all matching pods)
kubectl logs -n "$NS_PLATFORM" deploy/<deploy> --all-containers=true -f
```

## Exec, Port-Forward, Copy

```bash
kubectl exec -n "$NS_PLATFORM" -it <pod> -- sh

kubectl port-forward -n "$NS_PLATFORM" svc/<svc> 8080:80
kubectl port-forward -n "$NS_PLATFORM" deploy/<deploy> 8080:8080

kubectl cp -n "$NS_PLATFORM" <pod>:/path/in/pod ./local-path
kubectl cp -n "$NS_PLATFORM" ./local-path <pod>:/path/in/pod
```

## Persistent Volumes / Claims

```bash
kubectl get pv
kubectl get pvc -A
kubectl get pvc -n "$NS_PLATFORM"
kubectl describe pvc -n "$NS_PLATFORM" <pvc>
```

## Resources, Requests/Limits, HPA

```bash
kubectl top pods -n "$NS_PLATFORM"

kubectl get hpa -n "$NS_PLATFORM"
kubectl describe hpa -n "$NS_PLATFORM" <hpa>
```

## Port-Forward UIs (Common)

```bash
# Argo CD UI/API (service name may vary)
kubectl -n "$NS_ARGOCD" get svc
kubectl -n "$NS_ARGOCD" port-forward svc/argocd-server 8080:443

# Argo Workflows UI/API (service name may vary)
kubectl -n "$NS_WORKFLOWS" get svc
kubectl -n "$NS_WORKFLOWS" port-forward svc/argo-server 2746:2746
```

## SSH Tunnel Hopping (Access Dashboards From Your Laptop)

Typical setup: your `kubectl` access is on a cluster node (or bastion), not on your local machine.

Pattern:

1. SSH to a node that has `kubectl` configured.
2. On that node, run a `kubectl port-forward ...` that binds to `127.0.0.1`.
3. From your laptop, create an SSH tunnel (with jump hosts) that forwards your local port to that node’s `127.0.0.1:<port>`.

Replace placeholders:

- `<user>`: your SSH user
- `<bastion>`: public jump host (optional)
- `<node>`: cluster node you can SSH into and run `kubectl`

### Argo CD (via kubectl port-forward)

On the cluster node:

```bash
kubectl -n "$NS_ARGOCD" port-forward svc/argocd-server 8080:443
```

On your laptop (single jump host):

```bash
ssh -J <user>@<bastion> <user>@<node> -N -L 8080:127.0.0.1:8080
```

On your laptop (multiple jump hosts):

```bash
ssh -J <user>@<bastion1>,<user>@<bastion2> <user>@<node> -N -L 8080:127.0.0.1:8080
```

Then open `http://127.0.0.1:8080`.

### Argo Workflows (via kubectl port-forward)

On the cluster node:

```bash
kubectl -n "$NS_WORKFLOWS" port-forward svc/argo-server 2746:2746
```

On your laptop:

```bash
ssh -J <user>@<bastion> <user>@<node> -N -L 2746:127.0.0.1:2746
```

Then open `http://127.0.0.1:2746`.

### Argo Rollouts Dashboard

Two common deployments exist; use whichever matches your cluster.

Option A: Rollouts dashboard via kubectl plugin (runs a local web server on the node).

On the cluster node:

```bash
kubectl argo rollouts dashboard -n "$NS_PLATFORM" --host 127.0.0.1 --port 3100
```

On your laptop:

```bash
ssh -J <user>@<bastion> <user>@<node> -N -L 3100:127.0.0.1:3100
```

Then open `http://127.0.0.1:3100`.

Option B: Rollouts dashboard as a Service (if installed).

On the cluster node:

```bash
kubectl -n "$NS_PLATFORM" get svc | grep -i rollout
kubectl -n "$NS_PLATFORM" port-forward svc/<rollouts-dashboard-svc> 3100:3100
```

On your laptop:

```bash
ssh -J <user>@<bastion> <user>@<node> -N -L 3100:127.0.0.1:3100
```

## CRDs / API Resources (Sanity Checks)

```bash
kubectl get crd | grep -E 'argoproj|argo'
kubectl api-resources | grep -i argo
```

## Network Debug (Quick Checks)

```bash
# DNS resolution from inside the cluster (requires a pod with tools)
kubectl run -n "$NS_PLATFORM" -it --rm dnsutils --image=registry.k8s.io/e2e-test-images/jessie-dnsutils:1.3 --restart=Never -- sh

# Inside that shell:
# nslookup kubernetes.default
# nslookup <svc>.<ns>.svc.cluster.local
```

## Argo Workflows (Kubernetes View)

These require the Argo Workflows CRDs installed (e.g., `workflows.argoproj.io`).

```bash
# Controller/UI status
kubectl get pods -n "$NS_WORKFLOWS" -o wide
kubectl get svc -n "$NS_WORKFLOWS"

# WorkflowTemplates / CronWorkflows
kubectl get workflowtemplates -n "$NS_PLATFORM"
kubectl get cronworkflows -n "$NS_PLATFORM"
kubectl describe cronworkflow -n "$NS_PLATFORM" <cron>

# Workflows
kubectl get workflows -n "$NS_PLATFORM"
kubectl get workflows -n "$NS_PLATFORM" --sort-by=.metadata.creationTimestamp
kubectl describe workflow -n "$NS_PLATFORM" <wf>
kubectl get workflow -n "$NS_PLATFORM" <wf> -o yaml

# Pods created by a workflow
kubectl get pods -n "$NS_PLATFORM" -l workflows.argoproj.io/workflow=<wf>
```

## Argo Workflows (CLI)

If you have `argo` installed, this is usually the fastest way to operate workflows.

```bash
argo version

# List
argo list -n "$NS_PLATFORM"
argo get <wf> -n "$NS_PLATFORM"

# Submit from a template (typical for manual runs)
argo submit --from workflowtemplate/<template> -n "$NS_PLATFORM"
argo submit --from workflowtemplate/<template> -n "$NS_PLATFORM" -p key=value

# Watch a workflow live
argo watch <wf> -n "$NS_PLATFORM"

# Live logs (workflow-wide)
argo logs <wf> -n "$NS_PLATFORM" -f

# Live logs for a single step/node
argo logs <wf> -n "$NS_PLATFORM" --node-id <node-id> -f

# Resubmit / retry
argo retry <wf> -n "$NS_PLATFORM"
argo resubmit <wf> -n "$NS_PLATFORM"

# Terminate/stop
argo terminate <wf> -n "$NS_PLATFORM"
```

## Workflow Live Logs (kubectl-only)

If you cannot use `argo logs`, you can follow logs from workflow pods directly.

```bash
# List workflow pods
kubectl get pods -n "$NS_PLATFORM" -l workflows.argoproj.io/workflow=<wf> -o wide

# Follow all containers for one pod
kubectl logs -n "$NS_PLATFORM" <pod> --all-containers=true -f

# Common container names are executor-related; list containers first if unsure
kubectl get pod -n "$NS_PLATFORM" <pod> -o jsonpath='{.spec.containers[*].name}'
```

## Common “What’s Broken?” Commands

```bash
# Everything not Running/Completed
kubectl get pods -A | awk '$4 != "Running" && $4 != "Completed" {print}'

# Describe and events for a failing pod
kubectl describe pod -n "$NS_PLATFORM" <pod>
kubectl get events -n "$NS_PLATFORM" --field-selector involvedObject.name=<pod> --sort-by=.metadata.creationTimestamp

# Image pull / crash loop evidence
kubectl logs -n "$NS_PLATFORM" <pod> --all-containers=true --previous
```

## Repo-Specific Notes

- Argo CD bootstrap manifests live at `devops/argocd` and can be applied with `kubectl apply -k devops/argocd`.
- Workflow resources live at `devops/workflows` (templates, cron workflows, RBAC).
