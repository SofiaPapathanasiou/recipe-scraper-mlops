# Cluster Bootstrap Notebook Design

## Goal

Create two runnable Bash notebooks at the repository root:

1. A Terraform provisioning notebook that can be run by a third party before the cluster exists
2. A cluster bootstrap notebook that is run from the `node1` control host after infrastructure is available

Together, the notebooks should guide an operator through provisioning, cluster bootstrap, Argo CD setup, and application deployment using the automation already present in this repository.

## Scope

The deliverable will be split into two notebooks.

Notebook 1 will cover:

1. Chameleon credential prerequisite guidance
2. Provision infrastructure with Terraform in `devops/tf/kvm`
3. Output-oriented handoff instructions for accessing `node1`

Notebook 2 will cover:

1. SSH-to-`node1` and repo clone prerequisites
2. Verify Ansible connectivity with `devops/ansible/general/hello_host.yml`
3. Prepare nodes with `devops/ansible/pre_k8s/pre_k8s_configure.yml`
4. Create the Kubernetes cluster with Kubespray `cluster.yml`
5. Configure post-cluster tooling with `devops/ansible/post_k8s/post_k8s_configure.yml`
6. Bootstrap Argo CD applications with `devops/ansible/argocd/argocd_bootstrap_apps.yml`

Both notebooks will include prerequisite checks, operator inputs, validation commands after each major phase, and short recovery notes for common rerun scenarios.

## Non-Goals

- Replacing existing Terraform, Ansible, or Kubespray automation
- Refactoring playbooks or inventory layout as part of this task
- Providing a full teardown notebook
- Generalizing the workflow for execution from a laptop or any host other than `node1`

## Execution Assumptions

- Notebook 1 runs from a machine that can reach Chameleon Cloud APIs and has Terraform installed.
- The operator running Notebook 1 has Chameleon application credentials in `clouds.yaml` form and places them in `~/.config/openstack/clouds.yaml`.
- Notebook 2 runs on `node1`, which is the control host and public entrypoint after provisioning completes.
- Before running Notebook 2, the operator SSHs into `node1` and clones this repository locally.
- The operator has shell access, the SSH key referenced by the inventories, and the CLIs needed for the relevant notebook.
- Kubespray is available in the environment expected by this repo, and the operator can invoke its `cluster.yml` playbook from `node1`.

## Notebook Format

- File location: repo root
- File type: Jupyter notebooks with a Bash kernel
- Deliverables:
  - Terraform provisioning notebook
  - `node1` cluster bootstrap notebook
- Cell mix:
  - Markdown cells for context, warnings, expected outcomes, and rerun notes
  - Bash cells for executable steps
- Bash cells should begin with `set -euo pipefail` unless a cell intentionally captures failures for diagnostics

## Notebook 1: Terraform Provisioning

### 1. Title and outcome

Explain:

- that this notebook is run before `node1` is accessible
- what infrastructure it provisions in Chameleon
- that successful completion produces a `node1` entrypoint for the remaining setup
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

- required commands present
- `clouds.yaml` present
- expected Terraform files present

Add a dedicated markdown cell that explains:

- the operator must obtain Chameleon Cloud application credentials
- those credentials must be available as a `clouds.yaml`
- the file should be placed at `~/.config/openstack/clouds.yaml`
- Terraform provisioning cannot proceed without that file

### 3. Terraform phase

Include cells for:

- changing into `devops/tf/kvm`
- `terraform init`
- `terraform plan` using notebook variables
- `terraform apply` using notebook variables
- optional `terraform output`

Validation:

- confirm apply success
- show outputs relevant to accessing `node1`

Recovery note:

- mention partial apply cleanup guidance already documented in `devops/tf/kvm/README.md`

### 4. Handoff to node1

Include a markdown section that tells the operator to:

- SSH into `node1`
- clone this repository onto `node1`
- switch to the repo root there
- continue with Notebook 2 for all remaining steps

## Notebook 2: node1 Cluster Bootstrap

### 1. Title and outcome

Explain:

- that this notebook must be run from inside `node1`
- that Terraform provisioning must already be complete
- that the repo should already be cloned on `node1`
- the remaining cluster/bootstrap stage order

### 2. Prerequisites and operator inputs

Provide one early parameter cell that defines reusable variables such as:

- `REPO_ROOT`
- `ANSIBLE_DIR`
- `KUBESPRAY_DIR`
- `KUBESPRAY_INVENTORY`

Add checks for:

- running on `node1`
- required commands present
- expected inventory files present
- repository path exists locally on `node1`

### 3. Ansible connectivity phase

Run:

- `ansible-playbook -i devops/ansible/inventory.yml devops/ansible/general/hello_host.yml`

Validation:

- all hosts return hostnames successfully

### 4. Pre-K8s preparation phase

Run:

- `ansible-playbook -i devops/ansible/inventory.yml devops/ansible/pre_k8s/pre_k8s_configure.yml`

Validation:

- playbook completes without unreachable hosts
- optional follow-up ad hoc checks for package manager and networking state if needed

### 5. Kubespray cluster creation phase

Run Kubespray against `devops/ansible/k8s/inventory/mycluster/hosts.yaml` using the repo’s chosen invocation path for `cluster.yml`.

Validation:

- `kubectl get nodes -o wide`
- `kubectl get pods -A`
- note that `post_k8s` later copies `admin.conf` into `/home/cc/.kube/config`, so pre-validation may need explicit kubeconfig or root access depending on cluster state

### 6. Post-K8s configuration phase

Run:

- `ansible-playbook -i devops/ansible/inventory.yml devops/ansible/post_k8s/post_k8s_configure.yml`

Validation:

- `kubectl get nodes -o wide`
- `argocd version --client`
- `argo version --client`
- show `argocd` namespace resources

Operator note:

- highlight that this playbook reveals the initial Argo CD admin password in output

### 7. Argo CD bootstrap apps phase

Run:

- `ansible-playbook devops/ansible/argocd/argocd_bootstrap_apps.yml`

Validation:

- `kubectl get applications -n argocd`
- `kubectl get pods -n argocd`
- optional sync/health inspection using `argocd` CLI if login is configured

### 8. Final verification and next checks

Summarize:

- cluster node health
- Argo CD application registration
- suggested next commands for watching workloads converge

Include watch commands such as:

- `kubectl get pods -A`
- `kubectl get applications -n argocd -w`

## Design Choices

### Split notebooks instead of a single notebook

Provisioning happens before `node1` exists, while the remaining automation is intended to run from inside `node1`. Splitting the runbook along that boundary makes the execution context explicit and avoids mixing off-cluster and on-cluster assumptions in one notebook.

### Parameterized commands instead of hard-coded values

The Terraform README currently shows example values such as `suffix=proj22` and a sample flavor ID. The notebook should expose these as editable variables so the runbook remains reusable.

### Thin wrapper over existing automation

The notebook should call existing Terraform and Ansible entrypoints directly rather than reimplementing provisioning logic in notebook cells. This keeps repository automation as the source of truth.

## Risks and Mitigations

- Inventory drift after Terraform apply
  - Mitigation: add an explicit checkpoint in the `node1` notebook reminding the operator to confirm host IPs before Ansible and Kubespray steps.
- Missing or invalid Chameleon credentials
  - Mitigation: add an early notebook cell explaining the `clouds.yaml` requirement and fail fast if `~/.config/openstack/clouds.yaml` is missing.
- Missing Bash kernel on `node1`
  - Mitigation: include a prerequisite note that both notebooks require a Bash-capable Jupyter kernel in their execution environment.
- Long-running or flaky infrastructure steps
  - Mitigation: keep phases separated into rerunnable cells and add short recovery notes instead of one monolithic cell.
- Secret exposure in notebook output
  - Mitigation: warn that the post-K8s playbook prints the initial Argo CD admin password and that notebook outputs may need clearing before sharing.

## Testing Strategy

Implementation will verify:

- both notebook JSON files are valid
- both notebooks open as root-level `.ipynb` files
- cells reference real repository paths and playbooks
- command order matches the intended bring-up flow
- Terraform notebook does not assume `node1` access
- bootstrap notebook assumes execution from `node1`
