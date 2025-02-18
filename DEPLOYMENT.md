# ML Model Monitoring System Deployment Guide

This guide provides step-by-step instructions for deploying the refinance prediction model monitoring system on Google Kubernetes Engine (GKE) with Prometheus monitoring.

## Prerequisites

1. Google Cloud Platform (GCP) account with the following APIs enabled:
   - Google Kubernetes Engine API
   - BigQuery API
   - IAM API

2. Local tools installed:
   - `gcloud` CLI
   - `kubectl`
   - `helm` (v3+)
   - `terraform` (v1.0+)

## Deployment Steps

### Step 1: Configure GCP Project

```bash
# Login to GCP
gcloud auth login

# Set your project ID
export PROJECT_ID="mdepew-assets"
gcloud config set project $PROJECT_ID

# Enable required APIs
gcloud services enable container.googleapis.com bigquery.googleapis.com iam.googleapis.com
```

### Step 2: Configure Service Account

```bash
# Create Workload Identity service account
gcloud iam service-accounts create gke-workload-ksa \
    --description="GKE Workload Identity SA" \
    --display-name="GKE Workload Identity"

# Grant necessary permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:gke-workload-ksa@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/bigquery.user"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:gke-workload-ksa@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/bigquery.dataViewer"
```

### Step 3: Deploy Infrastructure with Terraform

```bash
# Initialize Terraform
cd terraform
terraform init

# Review the deployment plan
terraform plan

# Apply the configuration
terraform apply
```

### Step 4: Configure kubectl

```bash
# Get cluster credentials
gcloud container clusters get-credentials ml-monitoring-cluster \
    --region us-central1 \
    --project $PROJECT_ID
```

### Step 5: Deploy Kubernetes Service Account

```bash
# Create namespace
kubectl create namespace default

# Apply service account
kubectl apply -f k8s/service-account.yaml
```

### Step 6: Deploy Prometheus

```bash
# Add Prometheus Helm repository
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Create namespace for Prometheus
kubectl create namespace prometheus

# Deploy Prometheus with custom configuration
helm install prometheus prometheus-community/prometheus \
    -n prometheus \
    -f prometheus/values.yaml
```

### Step 7: Deploy Model Service

```bash
# Deploy the refinance predictor service
kubectl apply -f k8s/app.yaml
```

### Step 8: Verify Deployment

```bash
# Check all pods are running
kubectl get pods -A

# Get Prometheus server URL
export SERVICE_IP=$(kubectl get svc --namespace prometheus prometheus-server -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
echo http://$SERVICE_IP:80

# Access AlertManager UI (in a separate terminal)
kubectl --namespace prometheus port-forward svc/prometheus-alertmanager 9093
# Access at http://localhost:9093
```

## Monitoring Components

### Metrics
- `refinance_prediction_requests_total`: Total prediction requests
- `refinance_prediction_latency_seconds`: Prediction latency
- `refinance_prediction_probability`: Prediction probabilities
- `refinance_model_health`: Model health status
- `refinance_system_memory_usage_bytes`: Memory usage
- `refinance_bigquery_errors_total`: BigQuery errors
- `refinance_bigquery_query_latency_seconds`: BigQuery query latency

### Alerts
1. Model Performance
   - High Model Latency (>1s)
   - High Error Rate (>5%)
   - Model Health Check

2. Data Source
   - BigQuery Connection Failure
   - High BigQuery Latency (>10s)

## Troubleshooting

1. Check pod status:
```bash
kubectl get pods -n prometheus
kubectl get pods -n default
```

2. View pod logs:
```bash
kubectl logs <pod-name> -n <namespace>
```

3. Check Prometheus targets:
```bash
kubectl port-forward svc/prometheus-server 9090:80 -n prometheus
# Access http://localhost:9090/targets
```

## Cleanup

To remove all resources:

```bash
# Delete Kubernetes resources
helm uninstall prometheus -n prometheus
kubectl delete -f k8s/

# Delete infrastructure
cd terraform
terraform destroy
```
