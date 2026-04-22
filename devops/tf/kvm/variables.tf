variable "openstack_cloud" {
  description = "Cloud entry name from clouds.yaml"
  type        = string
  default     = "openstack"
}

variable "openstack_region" {
  description = "OpenStack region name"
  type        = string
  default     = "CHI@TACC"
}

variable "openstack_endpoint_type" {
  description = "OpenStack endpoint interface to use"
  type        = string
  default     = "public"
}

variable "suffix" {
  description = "Suffix for resource names (use net ID)"
  type        = string
  nullable    = false
}

variable "key" {
  description = "Name of key pair"
  type        = string
  default     = "id_rsa_chameleon"
}

variable "cpu_flavor_id" {
  description = "Flavor UUID to use for CPU nodes"
  type        = string
  default     = null
}

variable "reservation_cpu" {
  description = "Deprecated: use cpu_flavor_id instead"
  type        = string
  default     = null
}

variable "gpu_flavor_id" {
  description = "Flavor UUID to use for GPU nodes"
  type        = string
  default     = null
}

variable "gpu_flavor_ids" {
  description = "Per-GPU-node flavor UUID map; overrides gpu_flavor_id per node when provided"
  type        = map(string)
  default     = {}
}

variable "reservation_gpu" {
  description = "Deprecated: use gpu_flavor_id instead"
  type        = string
  default     = null
}

variable "create_gpu_node" {
  description = "Whether to create the GPU node in this stack"
  type        = bool
  default     = true
}

variable "nodes" {
  description = "CPU nodes for the cluster: node1-node3 control plane, node4-node5 workers"
  type        = map(string)
  default = {
    "node1" = "192.168.1.11"
    "node2" = "192.168.1.12"
    "node3" = "192.168.1.13"
    "node4" = "192.168.1.14"
    "node5" = "192.168.1.15"
  }
}

variable "sharednet1_nodes" {
  description = "CPU nodes that should attach to sharednet1; default keeps node1 as the public entrypoint"
  type        = list(string)
  default     = ["node1"]
}

variable "gpu_nodes" {
  description = "GPU nodes brought up separately using gpu_flavor_id or per-node gpu_flavor_ids"
  type        = map(string)
  default = {
    "gpu-node1" = "192.168.1.16"
    "gpu-node2" = "192.168.1.17"
  }
}
