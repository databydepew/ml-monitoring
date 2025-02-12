# ML Model Monitoring with Prometheus and Distribution Monitoring

This project demonstrates how to deploy and monitor a machine learning model using Kubernetes and Prometheus. The model predicts loan refinancing approvals and includes advanced monitoring features such as:

- Real-time distribution monitoring for feature drift detection
- Prediction distribution monitoring
- Model performance tracking
- Integration with BigQuery for reference data
- Prometheus metrics for all monitoring aspects

## Prerequisites

- Docker
- Kubernetes cluster (Kind or Minikube)
- kubectl
- Helm 3.x
- Python 3.10+
- Google Cloud account with BigQuery access (optional)
- Google Cloud SDK (optional)

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
    "interest_rate": 3.5,
    "loan_amount": 300000,
    "loan_balance": 290000,
    "loan_to_value_ratio": 0.8,
    "credit_score": 750,
    "debt_to_income_ratio": 0.3,
    "income": 120000,
    "loan_term": 30,
    "loan_age": 2,
    "home_value": 375000,
    "current_rate": 4.2,
    "rate_spread": 0.7
  }'
```

3. Provide feedback:
```bash
curl -X POST http://localhost:5000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "prediction_id": 0,
    "actual_outcome": 1
  }'
```

4. View Prometheus metrics:
```bash
curl http://localhost:8001/metrics
```

### Available Metrics

1. Distribution Monitoring:
   - `feature_kl_divergence`: KL divergence between training and production distributions
   - `feature_drift_p_value`: P-value for distribution drift test
   - `feature_drift_detected`: Whether significant drift was detected
   - `prediction_distribution_shift`: KL divergence in prediction distributions

2. Model Performance:
   - `model_prediction_latency_seconds`: Time spent processing prediction requests
   - `model_prediction_values`: Distribution of model predictions

3. Feature Tracking:
   - Individual gauges for each feature (e.g., `model_feature_credit_score`)

### BigQuery Integration

The application can use BigQuery as a source for reference data. To enable this:

1. Create a service account with BigQuery access
2. Create a Kubernetes secret with the credentials:
```bash
kubectl create secret generic bigquery-secret \
  --from-file=credentials.json=/path/to/service-account.json \
  -n monitoring
```

3. Configure BigQuery settings in the deployment:
```yaml
env:
  - name: GOOGLE_CLOUD_PROJECT
    value: "your-project-id"
  - name: BIGQUERY_DATASET
    value: "your-dataset"
  - name: BIGQUERY_TABLE
    value: "your-table"
```

## Monitoring Architecture

### Distribution Monitoring

The application uses KL divergence to detect distribution shifts in both features and predictions:

1. Feature Distribution Monitoring:
   - Maintains reference distributions from training data
   - Computes KL divergence between reference and production distributions
   - Uses bootstrap sampling for statistical significance testing
   - Alerts on significant distribution shifts

2. Prediction Distribution Monitoring:
   - Tracks shifts in model prediction distributions
   - Helps detect concept drift and model degradation

### BigQuery Integration

The system can use BigQuery as a source of truth for:
- Reference data distributions
- Historical predictions
- Model performance metrics

### Prometheus Integration

All monitoring metrics are exposed via Prometheus:
- Feature distribution metrics
- Prediction distribution metrics
- Model performance metrics
- Request latency metrics

These metrics can be visualized using Grafana or any Prometheus-compatible visualization tool.
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

### Distribution Monitoring with KL-Divergence

Kullback-Leibler (KL) divergence is a measure of how one probability distribution differs from another reference probability distribution. In our model monitoring context, we use it to detect when the distribution of production data starts to drift away from our training data distribution.

#### What is KL-Divergence?

KL-divergence (D_KL) measures the relative entropy between two probability distributions P and Q. For discrete probability distributions, it is defined as:

```
D_KL(P||Q) = Σ P(x) * log(P(x)/Q(x))
```

Where:
- P(x) is the reference (training) distribution
- Q(x) is the production distribution
- The sum is over all possible values x

#### Properties of KL-Divergence:

1. **Non-negative**: D_KL(P||Q) ≥ 0 for all P, Q
2. **Zero only when identical**: D_KL(P||Q) = 0 if and only if P = Q
3. **Asymmetric**: D_KL(P||Q) ≠ D_KL(Q||P)

#### How We Use KL-Divergence:

1. **Feature Distribution Monitoring**:
   ```python
   # Example of monitoring income distribution
   reference_dist = compute_distribution(training_data['income'])
   production_dist = compute_distribution(recent_data['income'])
   kl_div = entropy(reference_dist, production_dist)
   ```

2. **Statistical Testing**:
   - We use bootstrap sampling to determine if the KL-divergence is statistically significant
   - P-values < 0.05 indicate significant distribution drift

3. **Drift Detection**:
   - Monitor KL-divergence for each feature
   - Track prediction distribution shifts
   - Alert when divergence exceeds thresholds

#### Interpreting KL-Divergence Values:

- **0.0 - 0.1**: Minimal drift
- **0.1 - 0.5**: Moderate drift, monitor closely
- **> 0.5**: Significant drift, investigate and potentially retrain

#### Example Visualization:
```
Reference Distribution:     Production Distribution:
    ╭─╮                         ╭──╮
   ╭╯ ╰╮                       ╭╯  ╰╮
  ╭╯   ╰╮                     ╭╯    ╰╮
 ╭╯     ╰╮                   ╭╯      ╰╮
╭╯       ╰╮                 ╭╯        ╰╮
─────────────               ─────────────
KL-Divergence = 0.42 (Moderate Drift)
```

#### Using KL-Divergence to Monitor Model Accuracy

KL-divergence serves as an early warning system for potential accuracy degradation through several mechanisms:

1. **Leading Indicator**:
   - Distribution drift often precedes accuracy drops
   - Example: If income distribution shifts higher in production:
     ```python
     # Training data was mostly low-income
     training_income = [30k-50k: 60%, 50k-70k: 30%, 70k+: 10%]
     
     # Production shifts to high-income
     production_income = [30k-50k: 20%, 50k-70k: 40%, 70k+: 40%]
     
     # High KL-divergence warns of potential accuracy issues
     # Model may not perform well on high-income cases
     ```

2. **Feature Importance Correlation**:
   - Monitor KL-divergence against accuracy changes
   - Higher divergence in important features → larger accuracy impact
   ```python
   feature_impacts = {
       'credit_score': {'kl_div': 0.45, 'acc_change': -0.15},
       'income': {'kl_div': 0.30, 'acc_change': -0.08},
       'loan_term': {'kl_div': 0.10, 'acc_change': -0.02}
   }
   # credit_score drift has highest impact on accuracy
   ```

3. **Threshold-Based Monitoring**:
   ```python
   if kl_divergence > 0.5:  # Significant drift
       # Check accuracy more frequently
       # Consider model retraining
       alert_team("High drift detected - validate model accuracy")
   ```

4. **Multi-Metric Dashboard**:
   ```
   Feature KL-Divergence | Accuracy Change
   ----------------------------------------
   0.1-0.3              | -1% to -3%
   0.3-0.5              | -3% to -8%
   > 0.5                | > -8%
   ```

5. **Automated Response System**:
   ```python
   def monitor_model_health():
       if any(feature_kl_div > 0.5):
           # Increase accuracy monitoring frequency
           set_accuracy_monitoring(frequency='hourly')
           # Trigger validation on holdout set
           validate_on_holdout_set()
           # Alert if accuracy drops
           if accuracy_drop > 0.05:
               trigger_retraining()
   ```

6. **Root Cause Analysis**:
   - High KL-divergence helps identify which features are causing accuracy drops
   - Guides feature-specific interventions
   ```python
   # Example: Credit score drift analysis
   if credit_score_kl_div > threshold:
       analyze_credit_segments()
       update_credit_score_bins()
       retrain_on_new_distribution()
   ```

7. **Retraining Decisions**:
   - Use KL-divergence trends to decide when to retrain
   - Balance between model stability and accuracy
   ```python
   def should_retrain():
       return (
           max(feature_kl_divs) > 0.5 and
           accuracy_trend.is_declining() and
           days_since_last_training > 7
       )
   ```

By monitoring both KL-divergence and accuracy metrics, you can:
- Detect potential issues early
- Identify which features need attention
- Make data-driven retraining decisions
- Maintain model performance proactively

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

#### Alternative Methods for Ground Truth Collection

1. **Database Integration**:
   - Connect to a loan management system or CRM database
   - Periodically query for actual loan outcomes
   - Match outcomes with predictions using prediction_id
   ```python
   # Example with SQL database
   actual_outcomes = db.query("""
       SELECT prediction_id, loan_status 
       FROM loan_outcomes 
       WHERE decision_date >= :start_date
   """)
   ```

2. **Event-Driven Updates**:
   - Subscribe to loan status change events
   - Automatically update model metrics when loan status changes
   - Use message queues (e.g., Kafka, RabbitMQ) for real-time updates
   ```python
   @kafka.consumer('loan_status_updates')
   def process_loan_status(event):
       prediction_id = event['prediction_id']
       actual_outcome = event['final_status']
       update_model_accuracy(prediction_id, actual_outcome)
   ```

3. **Batch Processing**:
   - Run periodic jobs to fetch outcomes in batches
   - Update metrics in bulk
   - Useful for scenarios with delayed ground truth
   ```python
   @scheduled_job('cron', hour='0')
   def update_model_metrics():
       yesterday = datetime.now() - timedelta(days=1)
       outcomes = fetch_loan_outcomes(date=yesterday)
       bulk_update_metrics(outcomes)
   ```

4. **External API Integration**:
   - Connect to third-party credit reporting APIs
   - Fetch loan performance data automatically
   - Cross-reference with model predictions
   ```python
   @api_client.scheduled_fetch
   def fetch_credit_outcomes():
       responses = credit_api.get_loan_statuses(loan_ids)
       update_model_metrics(responses)
   ```

5. **Human-in-the-Loop**:
   - Web interface for loan officers to input outcomes
   - Quality assurance team reviews and validates
   - Combine automated and manual verification
   ```python
   @app.route('/validate', methods=['POST'])
   def validate_outcome():
       reviewer_id = request.json['reviewer_id']
       prediction_id = request.json['prediction_id']
       validated_outcome = request.json['validated_outcome']
       update_metrics_with_validation(prediction_id, validated_outcome, reviewer_id)
   ```

6. **Hybrid Approach**:
   - Combine multiple methods based on data availability
   - Use automated methods for quick feedback
   - Supplement with manual validation for accuracy
   - Implement confidence scores for different sources

Considerations for Ground Truth Collection:
- Data freshness vs. accuracy trade-off
- Handling missing or delayed feedback
- Data quality and validation
- Compliance and privacy requirements
- Scalability of collection method

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

