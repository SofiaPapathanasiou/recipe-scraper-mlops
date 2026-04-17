provider "openstack" {
  cloud         = var.openstack_cloud
  region        = var.openstack_region
  endpoint_type = var.openstack_endpoint_type
}
