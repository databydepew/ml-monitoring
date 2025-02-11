# ML Model Monitoring with Prometheus

This project demonstrates how to deploy and monitor a machine learning model using Kubernetes and Prometheus. The model predicts loan approvals and tracks various metrics including model accuracy and prediction latency.

## Prerequisites

- Docker
- Kubernetes cluster (Kind or Minikube)
- kubectl
- Helm 3.x
- Python 3.8+

## Step 1: Kubernetes Setup

1. Create a monitoring namespace:
```bash
kubectl create namespace monitoring
```

## Step 2: Install Prometheus

1. Add Prometheus Helm repository:
```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
```

2. Install Prometheus:
```bash
helm upgrade --install prometheus prometheus-community/prometheus \
  --namespace monitoring \
  --set server.persistentVolume.size=4Gi \
  --set alertmanager.persistentVolume.size=1Gi \
  --set server.global.scrape_interval=15s
```

3. Install ServiceMonitor CRD:
```bash
kubectl apply -f https://raw.githubusercontent.com/prometheus-operator/prometheus-operator/main/example/prometheus-operator-crd/monitoring.coreos.com_servicemonitors.yaml
```

## Step 3: Deploy the Application

1. Build and load the Docker image:
```bash
# Build the image
docker build -t loan-model-api:latest .

# Load into Kind (if using Kind)
kind load docker-image loan-model-api:latest
```

2. Deploy the application:
```bash
kubectl apply -f manifests/deployment.yaml
```

3. Configure monitoring:
```bash
kubectl apply -f manifests/service-monitor.yaml
```

## Step 4: Access the Services

1. Port forward the application and metrics:
```bash
# Application and metrics endpoints
kubectl port-forward -n monitoring svc/loan-model-api 5000:5000 8001:8001 &

# Prometheus UI
kubectl port-forward -n monitoring svc/prometheus-server 9090:80 &
```

## Step 5: Test the Application

### Manual Testing

1. Health check:
```bash
curl -X GET http://localhost:5000/health
```

2. Make a prediction:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "income": 75000,
    "loan_amount": 25000,
    "loan_term": 36,
    "credit_score": 720,
    "employment_status": 1,
    "loan_purpose": 1
  }'
```

3. Provide feedback:
```bash
curl -X POST http://localhost:5000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "prediction_id": "0",
    "actual_outcome": 1,
    "predicted_outcome": 1
  }'
```

### Automated Testing

1. Install test requirements:
```bash
pip install -r test_requirements.txt
```

2. Run the test script:
```bash
python test_app.py
```

## Step 6: Monitor Metrics

Access Prometheus UI at `http://localhost:9090` to view the following metrics:

### Model Performance Metrics

#### Model Accuracy Calculation
The model accuracy is calculated using a rolling window approach:

1. The system maintains a history of the last 100 predictions using a fixed-size deque:
```python
PREDICTION_WINDOW = 100  # Track last 100 predictions
prediction_history = deque(maxlen=PREDICTION_WINDOW)
```

2. When feedback is received via the `/feedback` endpoint, it compares:
   - `actual_outcome`: The true outcome (0 or 1)
   - `predicted_outcome`: The model's prediction (0 or 1)

3. The accuracy calculation:
   - Each correct prediction (actual = predicted) adds a True to the history
   - Each incorrect prediction adds a False
   - Current accuracy = (number of True values) / (total predictions in window)
   ```python
   prediction_history.append(actual_outcome == predicted_outcome)
   current_accuracy = sum(prediction_history) / len(prediction_history)
   ```

4. This rolling window approach:
   - Provides a recent accuracy metric (last 100 predictions)
   - Automatically drops old predictions when new ones arrive
   - Helps identify recent model performance trends

Other performance metrics:
- `model_requests_total`: Total number of prediction requests
- `model_success_total`: Successful predictions
- `model_errors_total`: Failed predictions

### Prediction Metrics
- `model_prediction_latency_seconds`: Time spent processing prediction request
- `model_prediction_values`: Distribution of model predictions
- `model_predictions_by_class_total`: Total predictions by class

### Feature Metrics
- Feature value gauges for each input feature (age, income, etc.)

## Step 7: Clean Up

1. Stop port forwarding:
```bash
pkill -f "kubectl port-forward"
```

2. Remove application:
```bash
kubectl delete -f manifests/deployment.yaml
kubectl delete -f manifests/service-monitor.yaml
```

3. Uninstall Prometheus:
```bash
helm uninstall prometheus -n monitoring
```

4. Delete namespace:
```bash
kubectl delete namespace monitoring
```