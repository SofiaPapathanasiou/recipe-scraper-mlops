output "floating_ip_out" {
  description = "Floating IP assigned to node1"
  value       = openstack_networking_floatingip_v2.floating_ip.address
}

output "control_plane_private_ips" {
  description = "Private IPs for control plane nodes"
  value = {
    for node_name, ip in var.nodes : node_name => ip
    if contains(["node1", "node2", "node3"], node_name)
  }
}

output "worker_private_ips" {
  description = "Private IPs for worker nodes"
  value = {
    for node_name, ip in var.nodes : node_name => ip
    if contains(["node4", "node5"], node_name)
  }
}

output "gpu_private_ips" {
  description = "Private IPs for GPU nodes when enabled"
  value       = var.create_gpu_node ? var.gpu_nodes : {}
}
