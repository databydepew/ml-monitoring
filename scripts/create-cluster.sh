#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define variables
PROJECT_ID="mdepew-assets"
CLUSTER_NAME="test-inference"
REGION="us-central1"
NETWORK="vertexai"
SUBNETWORK="workbench"
GCP_SA_EMAIL="gke-workload-ksa@mdepew-assets.iam.gserviceaccount.com"
K8S_SA_NAME="gke-workload-ksa"
NAMESPACE="monitoring"
MACHINE_TYPE="e2-standard-4" # Adjust based on your needs
NODE_COUNT=3 # Adjust as needed

# Authenticate with GCP
echo "Setting GCP project to $PROJECT_ID"
gcloud config set project $PROJECT_ID

# Enable necessary APIs
echo "Enabling required Google Cloud services..."
gcloud services enable container.googleapis.com iam.googleapis.com

# Create the Standard GKE cluster with Workload Identity enabled
# echo "Creating Standard GKE cluster '$CLUSTER_NAME' in network '$NETWORK' and subnetwork '$SUBNETWORK'..."
# gcloud container clusters create "$CLUSTER_NAME" \
#     --region "$REGION" \
#     --num-nodes "$NODE_COUNT" \
#     --machine-type "$MACHINE_TYPE" \
#     --workload-pool="$PROJECT_ID.svc.id.goog" \
#     --enable-ip-alias \
#     --network "$NETWORK" \
#     --subnetwork "$SUBNETWORK" \
#     --release-channel "regular"

# Get credentials for the new cluster
echo "Fetching cluster credentials..."
gcloud container clusters get-credentials "$CLUSTER_NAME" --region "$REGION"

# Create a Kubernetes service account
echo "Creating Kubernetes service account..."
kubectl create serviceaccount $K8S_SA_NAME --namespace $NAMESPACE

# Annotate the Kubernetes service account to use the existing GCP service account
echo "Annotating Kubernetes service account..."
kubectl annotate serviceaccount $K8S_SA_NAME \
    --namespace $NAMESPACE \
    iam.gke.io/gcp-service-account=$GCP_SA_EMAIL

# Allow the GCP service account to impersonate Kubernetes workloads
echo "Binding IAM roles to the service account..."
gcloud iam service-accounts add-iam-policy-binding $GCP_SA_EMAIL \
    --role roles/iam.workloadIdentityUser \
    --member "serviceAccount:$PROJECT_ID.svc.id.goog[$NAMESPACE/$K8S_SA_NAME]"

# Display success message
echo "Standard Kubernetes cluster '$CLUSTER_NAME' with Workload Identity enabled has been created successfully!"
