# KVM Terraform

This Terraform stack creates the recipe scraper infrastructure on Chameleon OpenStack.

Current defaults in this folder are set up so that:

- `node1` is the only CPU node attached to `sharednet1`
- `node2` through `node5` are private-network-only
- the GPU node is skipped when `create_gpu_node=false`
- `cpu_flavor_id` and `gpu_flavor_id` are the preferred variable names

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
- No GPU node creation

Plan:

```bash
cd /home/cc/recipe-scraper-mlops/devops/tf/kvm

terraform plan \
  -var="openstack_cloud=kvm_tacc" \
  -var="openstack_region=KVM@TACC" \
  -var="openstack_endpoint_type=public" \
  -var="suffix=proj22" \
  -var="cpu_flavor_id=7df7c35e-b47d-4164-a9b7-148ba76885f3" \
  -var="create_gpu_node=false"
```

Apply:

```bash
cd /home/cc/recipe-scraper-mlops/devops/tf/kvm

terraform apply \
  -var="openstack_cloud=kvm_tacc" \
  -var="openstack_region=KVM@TACC" \
  -var="openstack_endpoint_type=public" \
  -var="suffix=proj22" \
  -var="cpu_flavor_id=7df7c35e-b47d-4164-a9b7-148ba76885f3" \
  -var="create_gpu_node=false"
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

## Notes

- `suffix` should be your NetID.
- `openstack_cloud` must match the cloud entry name in `~/.config/openstack/clouds.yaml`.
- `openstack_region` should match the target Chameleon site exactly, for example `KVM@TACC`.
- `node1` gets the floating IP and serves as the public entrypoint.
- Other CPU nodes are expected to be reached over the private network through `node1`.
