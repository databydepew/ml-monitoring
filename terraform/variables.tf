variable "project_id" {
  description = "The project ID to host the cluster in"
  default     = "mdepew-assets"
}

variable "region" {
  description = "The region to host the cluster in"
  default     = "us-central1"
}

variable "network" {
  description = "The VPC network to host the cluster in"
  default     = "vertexai"
}

variable "subnetwork" {
  description = "The subnetwork to host the cluster in"
  default     = "workbench"
}
