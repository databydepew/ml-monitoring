terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
  }
}

provider "google" {
  project = "mdepew-assets"
  region  = "us-central1"
}

# Create custom service account for GKE
resource "google_service_account" "gke_sa" {
  account_id   = "gke-workload-sa"
  display_name = "GKE Workload Identity Service Account"
}

# Grant BigQuery access
resource "google_project_iam_member" "bigquery_access" {
  project = "mdepew-assets"
  role    = "roles/bigquery.dataEditor"
  member  = "serviceAccount:${google_service_account.gke_sa.email}"
}

# Grant GCS access
resource "google_project_iam_member" "storage_access" {
  project = "mdepew-assets"
  role    = "roles/storage.objectViewer"
  member  = "serviceAccount:${google_service_account.gke_sa.email}"
}

# Create GKE cluster
resource "google_container_cluster" "primary" {
  name     = "vertexai-gke-cluster"
  location = "us-central1"
  network  = "vertexai"
  subnetwork = "workbench"

  # We can't create a cluster with no node pool defined, but we want to only use
  # separately managed node pools. So we create the smallest possible default
  # node pool and immediately delete it.
  remove_default_node_pool = true
  initial_node_count       = 1

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }
}

# Create node pool
resource "google_container_node_pool" "primary_nodes" {
  name       = "primary-node-pool"
  location   = "us-central1"
  cluster    = google_container_cluster.primary.name
  
  node_count = 6

  node_config {
    machine_type = "n1-standard-4"

    # Google recommends custom service accounts that have cloud-platform scope and permissions granted via IAM Roles.
    service_account = google_service_account.gke_sa.email
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    workload_metadata_config {
      mode = "GKE_METADATA"
    }
  }
}
