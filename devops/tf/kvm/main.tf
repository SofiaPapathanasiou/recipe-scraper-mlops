locals {
  effective_cpu_flavor_id = coalesce(var.cpu_flavor_id, var.reservation_cpu)
  effective_gpu_flavor_id = coalesce(var.gpu_flavor_id, var.reservation_gpu, "")
}

resource "openstack_networking_network_v2" "private_net" {
  name                  = "private-net-recipe-${var.suffix}"
  port_security_enabled = true
}

resource "openstack_networking_subnet_v2" "private_subnet" {
  name       = "private-subnet-recipe-${var.suffix}"
  network_id = openstack_networking_network_v2.private_net.id
  cidr       = "192.168.1.0/24"
  no_gateway = true
}

resource "openstack_networking_port_v2" "private_net_ports" {
  for_each              = var.nodes
  name                  = "port-${each.key}-recipe-${var.suffix}"
  network_id            = openstack_networking_network_v2.private_net.id
  port_security_enabled = true

  fixed_ip {
    subnet_id  = openstack_networking_subnet_v2.private_subnet.id
    ip_address = each.value
  }
}

resource "openstack_networking_port_v2" "sharednet1_ports" {
  for_each   = { for name, ip in var.nodes : name => ip if contains(var.sharednet1_nodes, name) }
  name       = "sharednet1-${each.key}-recipe-${var.suffix}"
  network_id = data.openstack_networking_network_v2.sharednet1.id
  security_group_ids = [
    data.openstack_networking_secgroup_v2.allow_ssh.id,
    data.openstack_networking_secgroup_v2.allow_http_80.id
  ]
}

resource "openstack_compute_instance_v2" "nodes" {
  for_each = var.nodes

  name       = "${each.key}-recipe-${var.suffix}"
  image_name = "CC-Ubuntu24.04"
  flavor_id  = local.effective_cpu_flavor_id
  key_pair   = var.key

  dynamic "network" {
    for_each = contains(var.sharednet1_nodes, each.key) ? [1] : []
    content {
      port = openstack_networking_port_v2.sharednet1_ports[each.key].id
    }
  }

  network {
    port = openstack_networking_port_v2.private_net_ports[each.key].id
  }

  user_data = <<-EOF
    #! /bin/bash
    sudo echo "127.0.1.1 ${each.key}-recipe-${var.suffix}" >> /etc/hosts
    su cc -c /usr/local/bin/cc-load-public-keys
  EOF

}

# GPU node — only created when gpu_flavor_id is provided
resource "openstack_networking_port_v2" "private_net_ports_gpu" {
  for_each              = var.create_gpu_node && local.effective_gpu_flavor_id != "" ? var.gpu_nodes : {}
  name                  = "port-${each.key}-recipe-${var.suffix}"
  network_id            = openstack_networking_network_v2.private_net.id
  port_security_enabled = true

  fixed_ip {
    subnet_id  = openstack_networking_subnet_v2.private_subnet.id
    ip_address = each.value
  }
}

resource "openstack_networking_port_v2" "sharednet1_ports_gpu" {
  for_each   = var.create_gpu_node && local.effective_gpu_flavor_id != "" ? var.gpu_nodes : {}
  name       = "sharednet1-${each.key}-recipe-${var.suffix}"
  network_id = data.openstack_networking_network_v2.sharednet1.id
  security_group_ids = [
    data.openstack_networking_secgroup_v2.allow_ssh.id,
    data.openstack_networking_secgroup_v2.allow_http_80.id
  ]
}

resource "openstack_compute_instance_v2" "gpu_node" {
  for_each = var.create_gpu_node && local.effective_gpu_flavor_id != "" ? var.gpu_nodes : {}

  name       = "${each.key}-recipe-${var.suffix}"
  image_name = "CC-Ubuntu24.04"
  flavor_id  = local.effective_gpu_flavor_id
  key_pair   = var.key

  network {
    port = openstack_networking_port_v2.sharednet1_ports_gpu[each.key].id
  }

  network {
    port = openstack_networking_port_v2.private_net_ports_gpu[each.key].id
  }

  user_data = <<-EOF
    #! /bin/bash
    sudo echo "127.0.1.1 ${each.key}-recipe-${var.suffix}" >> /etc/hosts
    su cc -c /usr/local/bin/cc-load-public-keys
  EOF
}

resource "openstack_networking_floatingip_v2" "floating_ip" {
  pool        = "public"
  description = "Recipe scraper IP for control plane node1 (${var.suffix})"
  port_id     = openstack_networking_port_v2.sharednet1_ports["node1"].id
}
