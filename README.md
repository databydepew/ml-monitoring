# ML Model Monitoring Platform

A robust MLOps platform for deploying and monitoring machine learning models in production, with advanced distribution monitoring and drift detection capabilities.

## Features

- **Real-time Model Monitoring**
  - Feature drift detection
  - Prediction distribution monitoring
  - Model performance tracking
  - Latency monitoring

- **Infrastructure**
  - Kubernetes-based deployment
  - Prometheus metrics integration
  - Grafana dashboards
  - Scalable architecture

- **Advanced Analytics**
  - Distribution monitoring with KL divergence
  - Reference data integration with BigQuery
  - Automated alerts for drift detection
  - Performance degradation tracking

## Getting Started

### Prerequisites

- Docker
- Kubernetes cluster (Kind or Minikube)
- kubectl
- Helm 3.x
- Python 3.10+
- Google Cloud account with BigQuery access (optional)
- Google Cloud SDK (optional)

### Quick Start

1. Clone the repository:
```bash
git clone https://github.com/databydepew/ml-monitoring.git
cd ml-monitoring
```

2. Create Kubernetes namespace:
```bash
kubectl create namespace monitoring
```

3. Set up the monitoring stack:
```bash
./create-cluster.sh
```

4. Deploy the model:
```bash
kubectl apply -f k8s/
```

## Monitoring Components

### 1. Model Metrics
- Prediction latency
- Request volume
- Error rates
- Model confidence scores

### 2. Distribution Monitoring
- Feature drift detection using KL divergence
- Prediction distribution analysis
- Automated drift alerts
- Historical distribution comparisons

### 3. Performance Tracking
- Accuracy metrics
- Feature importance
- Model degradation detection
- A/B test monitoring

## Project Structure

```
.
├── app/                    # Main application code
├── k8s/                   # Kubernetes manifests
├── grafana/               # Grafana dashboards
├── model/                 # ML model artifacts
├── monitoring/            # Monitoring configuration
├── tests/                 # Test suite
└── scripts/               # Utility scripts
```

## Metrics and Dashboards

Access the monitoring dashboards:
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090

Key metrics available:
- `model_prediction_latency_seconds`
- `model_prediction_values`
- `model_predictions_by_class_total`
- `feature_distribution_metrics`

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

Simulate model decay:
```bash
python simulate_decay.py
```

## Documentation

Detailed documentation is available in:
- [Monitoring Guide](README-monitoring.md)
- [Challenges and Solutions](README-challenges.md)

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
