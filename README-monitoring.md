# Refinance Prediction Model Monitoring Guide

This guide details the monitoring setup for our refinance prediction model deployed on GKE. The model predicts whether a mortgage holder is likely to refinance based on various loan and financial features.

## Model Overview

### Model Type
- Binary Classification (Refinance Prediction)
- Implementation: Random Classifier (for demonstration)
- Endpoint: Exposed on port 5001

### Data Source
- BigQuery Table: `synthetic.synthetic_mortgage_data`
- Features:
  1. `interest_rate`: Current interest rate of the loan
  2. `loan_amount`: Original loan amount
  3. `loan_balance`: Current loan balance
  4. `loan_to_value_ratio`: Ratio of loan to property value
  5. `credit_score`: Borrower's credit score
  6. `debt_to_income_ratio`: Debt-to-income ratio
  7. `income`: Annual income
  8. `loan_term`: Original loan term
  9. `loan_age`: Age of the loan
  10. `home_value`: Current home value
  11. `current_rate`: Current market rate
  12. `rate_spread`: Difference between loan rate and market rate
- Target: `refinance` (binary: 0/1)

## Monitoring Components

### 1. Model Monitor (`model_monitor.py`)

Primary monitoring service that tracks model performance and health:

#### Performance Metrics
- `refinance_model_accuracy`: Overall model accuracy
- `refinance_model_precision`: Precision for refinance predictions
- `refinance_model_recall`: Recall for refinance predictions
- `refinance_model_f1`: F1 score
- `ground_truth_refinance_rate`: Actual refinance rate
- `prediction_error_rate`: Rate of incorrect predictions

#### Feature Monitoring
- Feature importance tracking
- Distribution analysis
- Data quality checks
- Periodic BigQuery data validation

### 2. Drift Monitor (`drift_monitor.py`)

Specialized drift detection service:

#### Feature Drift Detection
- KL divergence tracking
- Kolmogorov-Smirnov tests
- Jensen-Shannon divergence
- Population Stability Index (PSI)
- P-value based significance testing

#### Sliding Window Analysis
- Window Size: 50 samples
- Significance Level: 0.1
- Continuous reference distribution updates

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

## Prometheus Metrics

### Model Performance
```
refinance_model_accuracy
refinance_model_precision
refinance_model_recall
refinance_model_f1
refinance_prediction_latency_seconds
```

### Feature Monitoring
```
refinance_feature_statistics{feature_name="<feature>",statistic="<stat>"}
refinance_feature_drift{feature_name="<feature>"}
refinance_bigquery_row_count
refinance_bigquery_data_freshness_hours
refinance_bigquery_null_values{feature_name="<feature>"}
```

### Drift Detection
```
refinance_model_feature_kl_divergence{feature="<feature>"}
refinance_model_feature_drift_p_value{feature="<feature>"}
refinance_model_feature_drift_detected{feature="<feature>"}
refinance_model_prediction_distribution_shift
```

## Alerts Configuration

### Model Performance Alerts
1. High Model Latency
   - Threshold: > 1s for 5 minutes
   - Severity: Warning

2. High Error Rate
   - Threshold: > 5% for 5 minutes
   - Severity: Critical

3. Model Health Check
   - Condition: health == 0
   - Severity: Critical

### Data Quality Alerts
1. BigQuery Connection Failure
   - Condition: Any errors in 5 minutes
   - Severity: Critical

2. High BigQuery Latency
   - Threshold: > 10s
   - Severity: Warning

3. Feature Drift
   - Condition: KS statistic > threshold
   - Severity: Warning

## Accessing Monitoring

### Prometheus UI
```bash
kubectl port-forward svc/prometheus-server 9090:80 -n prometheus
# Access: http://localhost:9090
```

### AlertManager
```bash
kubectl port-forward svc/prometheus-alertmanager 9093 -n prometheus
# Access: http://localhost:9093
```

## Troubleshooting

### Common Issues
1. BigQuery Connection Failures
   - Check service account permissions
   - Verify Workload Identity setup
   - Check network connectivity

2. High Latency
   - Check resource utilization
   - Verify BigQuery query optimization
   - Check for concurrent request volume

3. Drift Detection Issues
   - Verify reference data availability
   - Check sliding window size
   - Validate statistical test parameters

## Best Practices

1. Regular Monitoring
   - Check alerts daily
   - Review drift metrics weekly
   - Validate data quality monthly

2. Model Updates
   - Track performance degradation
   - Monitor feature importance changes
   - Document all model versions

3. Alert Management
   - Set appropriate thresholds
   - Configure alert routing
   - Maintain runbooks for each alert

## BigQuery Integration

### Data Source Configuration
- Table: `synthetic.synthetic_mortgage_data`
- Project: `mdepew-assets`
- Region: `us-central1`

### Data Ingestion
1. Training Data
   ```sql
   SELECT 
     interest_rate, loan_amount, loan_balance,
     loan_to_value_ratio, credit_score,
     debt_to_income_ratio, income, loan_term,
     loan_age, home_value, current_rate,
     rate_spread, refinance
   FROM synthetic.synthetic_mortgage_data
   WHERE training_set = TRUE
   ```

2. Production Monitoring
   ```sql
   SELECT *
   FROM synthetic.synthetic_mortgage_data
   WHERE TIMESTAMP_TRUNC(prediction_time, HOUR) = TIMESTAMP_TRUNC(CURRENT_TIMESTAMP(), HOUR)
   ```

### Monitoring Tables

1. Feature Statistics
   ```
   monitoring.feature_statistics
   ├── feature_name (STRING)
   ├── mean (FLOAT64)
   ├── std_dev (FLOAT64)
   ├── min (FLOAT64)
   ├── max (FLOAT64)
   ├── timestamp (TIMESTAMP)
   └── window_size (INT64)
   ```

2. Drift Metrics
   ```
   monitoring.drift_metrics
   ├── feature_name (STRING)
   ├── kl_divergence (FLOAT64)
   ├── ks_statistic (FLOAT64)
   ├── p_value (FLOAT64)
   ├── drift_detected (BOOLEAN)
   └── timestamp (TIMESTAMP)
   ```

3. Model Performance
   ```
   monitoring.model_metrics
   ├── accuracy (FLOAT64)
   ├── precision (FLOAT64)
   ├── recall (FLOAT64)
   ├── f1_score (FLOAT64)
   ├── prediction_count (INT64)
   └── timestamp (TIMESTAMP)
   ```

### Data Quality Checks

1. Freshness Check
   ```sql
   SELECT
     TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), MAX(prediction_time), HOUR) as data_delay_hours
   FROM synthetic.synthetic_mortgage_data
   ```

2. Completeness Check
   ```sql
   SELECT
     feature_name,
     COUNT(*) - COUNT(feature_value) as null_count,
     COUNT(*) as total_count
   FROM monitoring.feature_statistics
   GROUP BY feature_name
   ```

### Workload Identity Setup

1. Create Service Account
   ```bash
   gcloud iam service-accounts create ml-monitoring-sa \
     --project=mdepew-assets
   ```

2. Grant BigQuery Access
   ```bash
   gcloud projects add-iam-policy-binding mdepew-assets \
     --member="serviceAccount:ml-monitoring-sa@mdepew-assets.iam.gserviceaccount.com" \
     --role="roles/bigquery.dataViewer"
   ```

3. Configure Kubernetes Service Account
   ```bash
   kubectl create serviceaccount ml-monitoring-ksa -n monitoring
   ```

4. Bind Service Accounts
   ```bash
   gcloud iam service-accounts add-iam-policy-binding \
     ml-monitoring-sa@mdepew-assets.iam.gserviceaccount.com \
     --role roles/iam.workloadIdentityUser \
     --member "serviceAccount:mdepew-assets.svc.id.goog[monitoring/ml-monitoring-ksa]"
   ```

### Query Optimization

1. Partitioning
   - Tables are partitioned by `prediction_time`
   - Reduces query costs and improves performance
   - Enables efficient historical analysis

2. Clustering
   - Clustered by `feature_name` for monitoring tables
   - Improves query performance for feature-specific analysis
   - Reduces data scanned per query

3. Materialized Views
   - Used for commonly accessed metrics
   - Updated every 60 minutes
   - Reduces computation overhead

### Cost Management

1. Query Optimization
   - Use partitioned tables
   - Implement appropriate clustering
   - Leverage materialized views

2. Data Retention
   - Raw data: 90 days
   - Aggregated metrics: 1 year
   - Feature statistics: 6 months

3. Query Quotas
   - Daily quota: 2TB
   - Concurrent queries: 100
   - Rate limits monitored via Prometheus
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

