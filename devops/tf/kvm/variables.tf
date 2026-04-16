variable "suffix" {
  description = "Suffix for resource names (use net ID)"
  type        = string
  nullable = false
}

variable "key" {
  description = "Name of key pair"
  type        = string
  default     = "id_rsa_chameleon"
}

variable "reservation" {
  description = "UUID of the reservation for node1 (m1.large)"
  type        = string
}

variable "reservation_node2" {
  description = "UUID of the reservation for node2 (m1.medium)"
  type        = string
}

variable "reservation_gpu" {
  description = "UUID of the reservation for gpu node (g1.h100.pci.1)"
  type        = string
  default     = ""
}

variable "nodes" {
  type = map(string)
  default = {
    "node1" = "192.168.1.11"
    "node2" = "192.168.1.12"
  }
}

variable "gpu_nodes" {
  type = map(string)
  default = {
    "gpu-node" = "192.168.1.13"
  }
}
