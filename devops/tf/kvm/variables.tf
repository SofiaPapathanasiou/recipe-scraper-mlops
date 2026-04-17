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

variable "reservation_cpu" {
  description = "UUID of the reservation used for the CPU nodes (m1.xlarge)"
  type        = string
}

variable "reservation_gpu" {
  description = "UUID of the reservation used for the GPU node"
  type        = string
  default     = ""
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

variable "gpu_nodes" {
  description = "GPU node brought up separately using reservation_gpu"
  type        = map(string)
  default = {
    "gpu-node" = "192.168.1.16"
  }
}
