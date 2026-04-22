# Cluster Bootstrap Notebook Design

## Goal

Create a single runnable Bash notebook at the repository root that brings the platform up from scratch when executed from the `node1` control host. The notebook should guide an operator through provisioning, cluster bootstrap, Argo CD setup, and application deployment using the automation already present in this repository.

## Scope

The notebook will cover these stages in order:

1. Provision infrastructure with Terraform in `devops/tf/kvm`
2. Verify Ansible connectivity with `devops/ansible/general/hello_host.yml`
3. Prepare nodes with `devops/ansible/pre_k8s/pre_k8s_configure.yml`
4. Create the Kubernetes cluster with Kubespray `cluster.yml`
5. Configure post-cluster tooling with `devops/ansible/post_k8s/post_k8s_configure.yml`
6. Bootstrap Argo CD applications with `devops/ansible/argocd/argocd_bootstrap_apps.yml`

The notebook will also include prerequisite checks, operator inputs, validation commands after each major phase, and short recovery notes for common rerun scenarios.

## Non-Goals

- Replacing existing Terraform, Ansible, or Kubespray automation
- Refactoring playbooks or inventory layout as part of this task
- Providing a full teardown notebook
- Generalizing the workflow for execution from a laptop or any host other than `node1`

## Execution Assumptions

- The notebook runs on `node1`, which is the control host and public entrypoint.
- The repository is available on `node1` at a stable repo-root path.
- The operator has shell access, the SSH key referenced by the inventories, and a valid `~/.config/openstack/clouds.yaml`.
- Terraform, Ansible, `kubectl`, and required supporting CLIs are either already installed or can be checked before execution.
- Kubespray is available in the environment expected by this repo, and the operator can invoke its `cluster.yml` playbook from `node1`.

## Notebook Format

- File location: repo root
- File type: Jupyter notebook with a Bash kernel
- Cell mix:
  - Markdown cells for context, warnings, expected outcomes, and rerun notes
  - Bash cells for executable steps
- Bash cells should begin with `set -euo pipefail` unless a cell intentionally captures failures for diagnostics

## Notebook Structure

### 1. Title and outcome

Explain:

- what the notebook provisions
- that it is intended for clean cluster bring-up from `node1`
- the high-level stage order
- the expected manual supervision points

### 2. Prerequisites and operator inputs

Provide one early parameter cell that defines reusable variables such as:

- `REPO_ROOT`
- `TF_DIR`
- `ANSIBLE_DIR`
- `KUBESPRAY_DIR`
- `KUBESPRAY_INVENTORY`
- `TF_OPENSTACK_CLOUD`
- `TF_OPENSTACK_REGION`
- `TF_OPENSTACK_ENDPOINT_TYPE`
- `TF_SUFFIX`
- `TF_CPU_FLAVOR_ID`
- `TF_CREATE_GPU_NODE`

Add checks for:

- running on `node1`
- required commands present
- `clouds.yaml` present
- expected inventory files present

### 3. Terraform phase

Include cells for:

- changing into `devops/tf/kvm`
- `terraform init`
- `terraform plan` using notebook variables
- `terraform apply` using notebook variables
- optional `terraform output`

Validation:

- confirm apply success
- remind operator that Ansible and Kubespray inventory IPs must match provisioned hosts

Recovery note:

- mention partial apply cleanup guidance already documented in `devops/tf/kvm/README.md`

### 4. Ansible connectivity phase

Run:

- `ansible-playbook -i devops/ansible/inventory.yml devops/ansible/general/hello_host.yml`

Validation:

- all hosts return hostnames successfully

### 5. Pre-K8s preparation phase

Run:

- `ansible-playbook -i devops/ansible/inventory.yml devops/ansible/pre_k8s/pre_k8s_configure.yml`

Validation:

- playbook completes without unreachable hosts
- optional follow-up ad hoc checks for package manager and networking state if needed

### 6. Kubespray cluster creation phase

Run Kubespray against `devops/ansible/k8s/inventory/mycluster/hosts.yaml` using the repo’s chosen invocation path for `cluster.yml`.

Validation:

- `kubectl get nodes -o wide`
- `kubectl get pods -A`
- note that `post_k8s` later copies `admin.conf` into `/home/cc/.kube/config`, so pre-validation may need explicit kubeconfig or root access depending on cluster state

### 7. Post-K8s configuration phase

Run:

- `ansible-playbook -i devops/ansible/inventory.yml devops/ansible/post_k8s/post_k8s_configure.yml`

Validation:

- `kubectl get nodes -o wide`
- `argocd version --client`
- `argo version --client`
- show `argocd` namespace resources

Operator note:

- highlight that this playbook reveals the initial Argo CD admin password in output

### 8. Argo CD bootstrap apps phase

Run:

- `ansible-playbook devops/ansible/argocd/argocd_bootstrap_apps.yml`

Validation:

- `kubectl get applications -n argocd`
- `kubectl get pods -n argocd`
- optional sync/health inspection using `argocd` CLI if login is configured

### 9. Final verification and next checks

Summarize:

- cluster node health
- Argo CD application registration
- suggested next commands for watching workloads converge

Include watch commands such as:

- `kubectl get pods -A`
- `kubectl get applications -n argocd -w`

## Design Choices

### Single notebook instead of split notebooks

This workflow is sequential and operator-driven. A single notebook reduces context switching and makes it easier to recover from partial progress.

### Parameterized commands instead of hard-coded values

The Terraform README currently shows example values such as `suffix=proj22` and a sample flavor ID. The notebook should expose these as editable variables so the runbook remains reusable.

### Thin wrapper over existing automation

The notebook should call existing Terraform and Ansible entrypoints directly rather than reimplementing provisioning logic in notebook cells. This keeps repository automation as the source of truth.

## Risks and Mitigations

- Inventory drift after Terraform apply
  - Mitigation: add an explicit checkpoint reminding the operator to confirm host IPs before Ansible and Kubespray steps.
- Missing Bash kernel on `node1`
  - Mitigation: include a prerequisite note that the notebook requires a Bash-capable Jupyter kernel.
- Long-running or flaky infrastructure steps
  - Mitigation: keep phases separated into rerunnable cells and add short recovery notes instead of one monolithic cell.
- Secret exposure in notebook output
  - Mitigation: warn that the post-K8s playbook prints the initial Argo CD admin password and that notebook outputs may need clearing before sharing.

## Testing Strategy

Implementation will verify:

- notebook JSON is valid
- notebook opens as a root-level `.ipynb`
- cells reference real repository paths and playbooks
- command order matches the intended bring-up flow
- no stage assumes local-laptop execution
