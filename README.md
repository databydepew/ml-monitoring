# ML Model Monitoring Platform

A comprehensive MLOps monitoring solution for a refinance prediction model deployed on Google Kubernetes Engine (GKE) with advanced distribution monitoring and drift detection capabilities.

## Features

### Model Monitoring
- Real-time performance tracking (accuracy, precision, recall, F1)
- Feature drift detection using multiple methods:
  - KL divergence
  - Kolmogorov-Smirnov test
  - Jensen-Shannon divergence
  - Population Stability Index (PSI)
- Prediction distribution monitoring
- Feature importance tracking
- Latency monitoring

### Data Integration
- BigQuery integration for reference data
- Automated data quality checks
- Feature statistics logging
- Correlation analysis
- Data freshness monitoring

### Infrastructure
- GKE deployment with Workload Identity
- Prometheus metrics and alerting
- Custom recording rules
- Comprehensive alert system
- Scalable architecture

### Monitoring Stack
- Real-time metrics collection
- Customizable alert thresholds
- Historical data comparison
- Automated drift detection

## Prerequisites

### Google Cloud Platform
- GCP account with enabled APIs:
  - Google Kubernetes Engine API
  - BigQuery API
  - IAM API
- Project ID: mdepew-assets
- Region: us-central1

### Local Tools
- `gcloud` CLI
- `kubectl`
- `helm` (v3+)
- `terraform` (v1.0+)

## Quick Start

For detailed deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md)

1. Configure GCP:
```bash
gcloud auth login
export PROJECT_ID="mdepew-assets"
gcloud config set project $PROJECT_ID
```

2. Deploy Infrastructure:
```bash
cd terraform
terraform init
terraform apply
```

3. Configure kubectl:
```bash
gcloud container clusters get-credentials ml-monitoring-cluster \
    --region us-central1 \
    --project $PROJECT_ID
```

4. Deploy Monitoring Stack:
```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
kubectl create namespace prometheus
helm install prometheus prometheus-community/prometheus -n prometheus -f prometheus/values.yaml
```

5. Deploy Model Service:
```bash
kubectl apply -f k8s/app.yaml
```

