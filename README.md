# ML Model Monitoring Platform

A comprehensive MLOps monitoring solution for a refinance prediction model deployed on Google Kubernetes Engine (GKE) with advanced distribution monitoring and drift detection capabilities.

## Features

### Model Monitoring
- Real-time performance tracking (accuracy, precision, recall, F1)
- Feature drift detection using multiple statistical methods:
  - KL divergence with sliding window analysis
  - Kolmogorov-Smirnov test with p-value tracking
  - Configurable drift thresholds (warning/critical)
- Prediction distribution monitoring
- Feature importance tracking
- Latency monitoring

### Real-time Evaluation
- On-demand model evaluation via REST API
- Hourly performance tracking
- Comprehensive evaluation reports including:
  - Classification metrics
  - Feature drift analysis
  - Sample size validation
  - Confidence score distribution

### Data Integration
- BigQuery integration for:
  - Ground truth data comparison
  - Prediction storage and tracking
  - Feature distribution analysis
- Automated data quality checks
- Feature statistics logging
- Real-time drift metrics storage

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

## API Endpoints

### Model Prediction
```bash
POST /predict
Content-Type: application/json

{
    "interest_rate": 3.5,
    "loan_amount": 300000,
    "loan_balance": 290000,
    "loan_to_value_ratio": 0.8,
    "credit_score": 750,
    "debt_to_income_ratio": 0.35,
    "income": 120000,
    "loan_term": 30,
    "loan_age": 2,
    "home_value": 375000,
    "current_rate": 4.5,
    "rate_spread": 1.0
}
```

Response includes:
- Prediction (0/1)
- Confidence score
- Feature drift metrics
- KL divergence values
- KS test statistics

### Model Evaluation
```bash
GET /evaluate?hours_back=1&min_samples=10
```

Parameters:
- `hours_back`: Evaluation period in hours (default: 1)
- `min_samples`: Minimum required samples (default: 10)

Returns comprehensive evaluation report including:
- Performance metrics
- Drift analysis
- Sample statistics
- Confidence distributions

### Metrics
```bash
GET /metrics
```
Prometheus metrics endpoint exposing:
- Prediction counts and latencies
- Feature distributions
- Drift metrics
- Performance metrics

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

6. Create BigQuery Tables:
```bash
# Create predictions table
bq mk --table $PROJECT_ID:ml_monitoring.model_predictions predictions_schema.json

# Create ground truth table (if not exists)
bq mk --table $PROJECT_ID:ml_monitoring.ground_truth_data ground_truth_schema.json
```

