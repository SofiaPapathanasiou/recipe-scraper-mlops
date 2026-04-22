# KVM Terraform

This Terraform stack creates the recipe scraper infrastructure on Chameleon OpenStack.

Current defaults in this folder are set up so that:

- `node1` is the only CPU node attached to `sharednet1`
- `node2` through `node5` are private-network-only
- GPU nodes are skipped when `create_gpu_node=false`
- `cpu_flavor_id` and `gpu_flavor_id` are the preferred variable names
- `gpu_flavor_ids` can override GPU flavor per node (for example `gpu-node1` and `gpu-node2` with different flavors)

## Prerequisites

- A valid `clouds.yaml` entry for the target site in `~/.config/openstack/clouds.yaml`
- Terraform initialized in this folder
- A valid CPU flavor UUID for `cpu_flavor_id`

Backward compatibility note: `reservation_cpu` and `reservation_gpu` still work, but they are deprecated aliases for `cpu_flavor_id` and `gpu_flavor_id`.

## Initialize Terraform

From the repository root:

```bash
cd /home/cc/recipe-scraper-mlops/devops/tf/kvm
terraform init
```

## Create Infrastructure On KVM@TACC

This is the current working command set for `KVM@TACC` using:

- NetID: `proj22`
- CPU flavor ID: `7df7c35e-b47d-4164-a9b7-148ba76885f3`
- Two GPU nodes enabled with explicit flavor mapping via env vars

Set vars once:

```bash
cd /home/cc/recipe-scraper-mlops/devops/tf/kvm

export TF_VAR_openstack_cloud="kvm_tacc"
export TF_VAR_openstack_region="KVM@TACC"
export TF_VAR_openstack_endpoint_type="public"
export TF_VAR_suffix="proj22"
export TF_VAR_cpu_flavor_id="7df7c35e-b47d-4164-a9b7-148ba76885f3"
export TF_VAR_create_gpu_node="true"

# Optional fallback flavor used by every GPU node unless overridden below.
export TF_VAR_gpu_flavor_id=""

# Per-node override map: explicit different flavors for two GPU nodes.
export TF_VAR_gpu_flavor_ids='{"gpu-node1":"<gpu-flavor-uuid-1>","gpu-node2":"<gpu-flavor-uuid-2>"}'
```

Plan:

```bash
cd /home/cc/recipe-scraper-mlops/devops/tf/kvm

terraform plan
```

Apply:

```bash
cd /home/cc/recipe-scraper-mlops/devops/tf/kvm

terraform apply
```

## Clean Up A Partial Failed Apply

If a previous apply partially succeeded on `KVM@TACC`, delete the known orphaned shared network port first, then destroy the tracked resources:

```bash
cd /home/cc/recipe-scraper-mlops/devops/tf/kvm

OS_CLOUD=kvm_tacc openstack port delete sharednet1-node2-recipe-proj22

terraform destroy \
  -var="openstack_cloud=kvm_tacc" \
  -var="openstack_region=KVM@TACC" \
  -var="openstack_endpoint_type=public" \
  -var="suffix=proj22" \
  -var="cpu_flavor_id=7df7c35e-b47d-4164-a9b7-148ba76885f3" \
  -var="create_gpu_node=false"
```

## GPU Flavor Resolution

Terraform resolves GPU flavors in this order:

1. `gpu_flavor_ids["<gpu-node-name>"]` (per-node explicit override)
2. `gpu_flavor_id` (shared fallback for all GPU nodes)
3. `reservation_gpu` (deprecated fallback)

If `create_gpu_node=true`, Terraform requires a resolved non-empty flavor for every GPU node in `gpu_nodes`.

## Notes

- `suffix` should be your NetID.
- `openstack_cloud` must match the cloud entry name in `~/.config/openstack/clouds.yaml`.
- `openstack_region` should match the target Chameleon site exactly, for example `KVM@TACC`.
- `node1` gets the floating IP and serves as the public entrypoint.
- Other CPU nodes are expected to be reached over the private network through `node1`.
