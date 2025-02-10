

Here's a summary of all the important commands we ran, in order:

Started Minikube and connected to Docker daemon:
bash
CopyInsert
minikube start
eval $(minikube docker-env)
Built and deployed the model:
bash
CopyInsert
docker build -t ml-monitoring:latest .
kubectl apply -f model-deployment.yaml
Set up port forwarding for the model service:
bash
CopyInsert in Terminal
kubectl port-forward service/loan-model 5000:80
Set up Prometheus monitoring:
bash
CopyInsert
# Create monitoring namespace (if not exists)
kubectl get namespace monitoring

# Apply model drift rules
kubectl apply -f model-drift-rules.yaml

# Add Prometheus Helm repo
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install Prometheus
helm install prometheus prometheus-community/prometheus \
  --namespace monitoring \
  --set server.retention=15d \
  --set server.persistentVolume.size=10Gi
Set up port forwarding for Prometheus and AlertManager:
bash
CopyInsert
kubectl port-forward -n monitoring svc/prometheus-server 9090:80
kubectl port-forward -n monitoring svc/prometheus-alertmanager 9093:9093
Test the model with sample predictions:
bash
CopyInsert
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
Useful commands for monitoring and debugging:

bash
CopyInsert
# Check pod status
kubectl get pods -n monitoring

# Check pod logs
kubectl logs -n monitoring <pod-name> -c <container-name>

# View service details
kubectl get svc -n monitoring

# Check Prometheus metrics
curl http://localhost:5000/metrics

# Delete resources if needed
kubectl delete configmap <configmap-name> -n monitoring
helm uninstall prometheus -n monitoring
Access URLs:

Model API: http://localhost:5000
Prometheus UI: http://localhost:9090
AlertManager: http://localhost:9093


-----
This PromQL expression is used to detect **significant deviations** in model predictions over time. Let's break it down step by step.

### **Expression:**
```promql
abs(
  rate(model_prediction_values_sum[1h]) / rate(model_prediction_values_count[1h]) -
  avg_over_time(model_prediction_values_sum[1d]) / avg_over_time(model_prediction_values_count[1d])
) > 0.2
```

---

### **Step-by-step breakdown:**

#### **1. rate(model_prediction_values_sum[1h])**
- `rate()` computes the per-second average increase in the `model_prediction_values_sum` metric over the past **1 hour**.
- `model_prediction_values_sum` represents the sum of all model predictions.
- This effectively gives an estimate of how the model's prediction values are changing in real time.

#### **2. rate(model_prediction_values_count[1h])**
- Similar to the previous step, this computes the per-second average increase in the count of predictions over **1 hour**.
- `model_prediction_values_count` tracks the total number of predictions made.

#### **3. rate(model_prediction_values_sum[1h]) / rate(model_prediction_values_count[1h])**
- This calculates the **average prediction value** over the last **1 hour**.
- This is done by dividing the total predicted sum by the number of predictions.

#### **4. avg_over_time(model_prediction_values_sum[1d])**
- `avg_over_time()` computes the **average** of `model_prediction_values_sum` over the past **24 hours**.

#### **5. avg_over_time(model_prediction_values_count[1d])**
- Similarly, this calculates the **average** of `model_prediction_values_count` over the last **24 hours**.

#### **6. avg_over_time(model_prediction_values_sum[1d]) / avg_over_time(model_prediction_values_count[1d])**
- This computes the **baseline average prediction value** over the past **24 hours**.

#### **7. abs(... difference ...) > 0.2**
- This calculates the absolute difference between the **current 1-hour average prediction value** and the **baseline 24-hour average prediction value**.
- If this difference is greater than **0.2** (20%), an alert is triggered.

---

### **Why is this useful?**
- It helps detect if the model's prediction values are **deviating significantly** from the past day’s trend.
- A sudden change in model predictions could indicate:
  - **Model drift** (e.g., changes in model behavior).
  - **Data drift** (e.g., incoming data distribution has shifted).
  - **Issues with model inputs** (e.g., feature anomalies or incorrect data processing).

---

### **Example Scenarios**
| Time Period | Avg Prediction Value | Baseline (24h) | Difference | Alert Triggered? |
|------------|---------------------|----------------|------------|----------------|
| Last 1 hour | **0.85** | **0.60** | **0.25 (25%)** | ✅ Yes (above 20%) |
| Last 1 hour | **0.75** | **0.70** | **0.05 (5%)** | ❌ No (below 20%) |

---

### **Final Summary**
This expression detects when the model's predictions **significantly deviate (more than 20%) from the past 24-hour average**, helping to monitor **model drift and potential issues**. 🚀

