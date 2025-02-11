# Deploying the Custom Model Inference on Minikube

This guide will walk you through deploying the custom model inference on Minikube and testing the endpoints.

## Prerequisites

- Docker installed on your machine
- Minikube installed on your machine
- kubectl installed on your machine

## Step 1: Build the Docker Image

- `cd` into the root directory of this repository
- Run `docker build -t loan-model-api:latest .` to build the Docker image

## Step 2: Apply the Kubernetes Deployment

- Run `kubectl apply -f manifests/deployment.yaml` to apply the Kubernetes deployment
- This will create a deployment named `loan-model-api` with 1 replica

## Step 3: Apply the Kubernetes Service

- Run `kubectl apply -f manifests/service.yaml` to apply the Kubernetes service
- This will create a service named `loan-model-api` that exposes port 5000

## Step 4: Port-Forward the Service

- Run `kubectl port-forward svc/loan-model-api 5000:5000 &` to port-forward the service
- This will allow you to access the service on `localhost:5000`

## Step 5: Test the Endpoints

- Run `curl -X GET http://localhost:5000/health` to test the health endpoint
- Run `curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"age": 35, "income": 75000, "loan_amount": 25000, "loan_term": 36, "credit_score": 720, "employment_status": 1, "loan_purpose": 1}'` to test the predict endpoint
- Run 
```
curl -X POST http://localhost:5000/feedback -H "Content-Type: application/json" -d '{"prediction_id": "123", "actual_outcome": 1}'

```
 to test the feedback endpoint

## Step 6: Clean Up

- Run `kubectl delete deployment loan-model-api` to delete the deployment
- Run `kubectl delete svc loan-model-api` to delete the service
------

installing prometheus 
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts && helm repo update



helm upgrade --install prometheus prometheus-community/prometheus \
  --namespace monitoring \
  --set server.persistentVolume.size=4Gi \
  --set alertmanager.persistentVolume.size=1Gi \
  --set server.global.scrape_interval=15s

kubectl apply -f https://raw.githubusercontent.com/prometheus-operator/prometheus-operator/main/example/prometheus-operator-crd/monitoring.coreos.com_servicemonitors.yaml

kubectl apply -f manifests/service-monitor.yaml

kubectl port-forward -n monitoring svc/prometheus-server 9090:80 5000:5000 &
  
kubectl port-forward svc/loan-model-api 8001:8001  -n monitoring &

The metrics being collected include:

model_requests_total: Total number of prediction requests
model_errors_total: Total number of prediction errors
model_success_total: Total number of successful predictions
model_prediction_latency_seconds: Time spent processing prediction requests
model_prediction_values: Distribution of model predictions
Feature value gauges for each feature
model_predictions_by_class_total: Total predictions by class
model_accuracy: Current accuracy of the model
You can query these metrics in the Prometheus UI using PromQL. For example, to see the total number of requests, you can query model_requests_total.