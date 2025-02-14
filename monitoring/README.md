# Refinance Model Monitoring System

This system provides comprehensive monitoring for the refinance prediction model, tracking model performance, data drift, and prediction distributions in real-time.

## Ground Truth Data Collection

The system collects ground truth data from BigQuery for model monitoring and drift detection:

### Data Source
- Project: `mdepew-assets`
- Dataset: `synthetic`
- Table: `synthetic_mortgage_data`

### Collection Process
1. **Recent Data**
   ```sql
   SELECT
     interest_rate, loan_amount, loan_balance,
     loan_to_value_ratio, credit_score,
     debt_to_income_ratio, income, loan_term,
     loan_age, home_value, current_rate,
     rate_spread, refinance
   FROM synthetic_mortgage_data
   WHERE DATE(CURRENT_TIMESTAMP()) = CURRENT_DATE()
   ```
   - Used for current performance metrics
   - Calculates accuracy, precision, recall, F1 score

2. **Historical Data**
   ```sql
   SELECT [features]
   FROM synthetic_mortgage_data
   WHERE DATE(CURRENT_TIMESTAMP()) != CURRENT_DATE()
   LIMIT 10000
   ```
   - Used as baseline for drift detection
   - Compares feature distributions

### Monitored Features
- Interest Rate
- Loan Amount
- Loan Balance
- Loan-to-Value Ratio
- Credit Score
- Debt-to-Income Ratio
- Income
- Loan Term
- Loan Age
- Home Value
- Current Rate
- Rate Spread

### Update Frequency

#### Data Collection
- **Ground Truth Data**: Fetched from BigQuery every 60 minutes
  - Collects today's data for performance metrics
  - Uses `DATE(CURRENT_TIMESTAMP()) = CURRENT_DATE()`
  - Includes actual refinance outcomes

#### Metrics Updates
- **Prometheus Scraping**: Every 15 seconds
  - Collects all current metric values
  - Updates Grafana dashboards
  - Configured in `prometheus.yml`

#### Real-time Updates
- **Prediction Metrics**: Instant updates
  - Updated as each prediction request arrives
  - Includes prediction probabilities
  - Error rates and model outputs

#### Historical Comparisons
- **Baseline Data**: Up to 10,000 records
  - Excludes today's data
  - Used for drift calculations
  - Refreshed every 60 minutes

#### Retention Periods
- **Prometheus**: 15 days of metrics
- **BigQuery**: Full historical data
- **Grafana**: Configurable, default 90 days

## Model Accuracy Calculation

### Data Flow
1. **Ground Truth Collection**
   ```sql
   SELECT
     interest_rate, loan_amount, loan_balance,
     loan_to_value_ratio, credit_score,
     debt_to_income_ratio, income, loan_term,
     loan_age, home_value, current_rate,
     rate_spread, refinance
   FROM synthetic_mortgage_data
   WHERE DATE(CURRENT_TIMESTAMP()) = CURRENT_DATE()
   ```
   - Fetches today's data including actual refinance outcomes
   - `refinance` column contains true labels (0 or 1)

2. **Model Predictions**
   - Features are extracted from BigQuery data
   - Model makes predictions on these features
   - Generates both class predictions (0/1) and probabilities

3. **Metric Calculation**
   ```python
   # Using scikit-learn metrics
   accuracy = accuracy_score(y_true, y_pred)
   precision = precision_score(y_true, y_pred)
   recall = recall_score(y_true, y_pred)
   f1 = f1_score(y_true, y_pred)
   ```

### Performance Metrics
1. **Accuracy**
   - Ratio of correct predictions to total predictions
   - Updated every 60 minutes with new ground truth data
   - Tracked in Prometheus gauge: `refinance_model_accuracy`

2. **Error Rate**
   - Calculated as `1 - accuracy`
   - Shows percentage of incorrect predictions
   - Tracked in Prometheus gauge: `prediction_error_rate`

3. **Additional Metrics**
   - **Precision**: Accuracy of positive predictions
   - **Recall**: Percentage of actual refinances caught
   - **F1 Score**: Harmonic mean of precision and recall
   - **Refinance Rate**: Actual percentage of refinances (`GROUND_TRUTH_REFINANCE_RATE`)

### Monitoring Dashboard
- Real-time accuracy metrics
- Historical trend graphs
- Alert thresholds:
  - Warning: accuracy < 95%
  - Critical: accuracy < 90%

## Feature Drift Detection

### Kolmogorov-Smirnov (KS) Test

```python
ks_stat, _ = ks_2samp(historical_data[feature], current_data[feature])
```

The KS test is a statistical test that measures how different two distributions are by finding the maximum distance between their cumulative distribution functions (CDFs).

#### How it Works
1. **Takes Two Samples**:
   - Historical data (e.g., last month's credit scores)
   - Current data (e.g., today's credit scores)

2. **Creates CDFs**:
   - Sorts each sample
   - Calculates cumulative probabilities
   - Creates step function for each distribution

3. **Finds Maximum Distance**:
   - Calculates vertical distance between CDFs at each point
   - Returns the maximum distance as KS statistic

#### Interpretation
- **Range**: 0 to 1
  - 0: Distributions are identical
  - 1: Distributions are completely different

- **Thresholds**:
  - < 0.1: Minor drift
  - 0.1 - 0.2: Moderate drift
  - > 0.2: Severe drift

#### Example
For credit scores:
```python
# Historical: mean=700, std=50
# Current: mean=720, std=45
ks_stat = 0.15  # Moderate drift
```

#### Advantages
1. Distribution-free (works for any continuous distribution)
2. Scale-invariant
3. Easy to interpret
4. Sensitive to both location and shape differences

#### Use in Monitoring
- Calculated every 60 minutes
- Stored in Prometheus: `feature_drift{metric_type="ks_statistic"}`
- Triggers alerts when drift exceeds thresholds
- Helps identify which features are changing

### Jensen-Shannon (JS) Divergence

```python
# Calculate histograms
hist_current, bins = np.histogram(current_data[feature], bins=20, density=True)
hist_historical, _ = np.histogram(historical_data[feature], bins=bins, density=True)

# Add small epsilon to avoid zero probabilities
hist_current = hist_current + 1e-10
hist_historical = hist_historical + 1e-10

# Normalize
hist_current = hist_current / hist_current.sum()
hist_historical = hist_historical / hist_historical.sum()

# Calculate JS divergence
js_div = jensenshannon(hist_current, hist_historical)
```

#### How it Works
1. **Create Histograms**:
   - Divides data into 20 bins
   - Normalizes to get probability distributions
   - Adds small epsilon (1e-10) to avoid log(0)

2. **Calculates Divergence**:
   - Takes square root of average KL divergence
   - Symmetric (A→B same as B→A)
   - Bounded between 0 and 1

#### Interpretation
- **Range**: 0 to 1
  - 0: Distributions are identical
  - 1: Distributions are completely different

- **Thresholds**:
  - < 0.1: Minor drift
  - 0.1 - 0.2: Moderate drift
  - > 0.2: Severe drift

#### Advantages
1. Symmetric (unlike KL divergence)
2. Always finite (due to epsilon)
3. Smooth response to changes
4. Good for comparing probability distributions

### Population Stability Index (PSI)

```python
# Using normalized histograms from above
psi = np.sum((hist_current - hist_historical) * np.log(hist_current / hist_historical))
```

#### How it Works
1. **Uses Same Histograms**:
   - Reuses normalized histograms from JS calculation
   - Maintains same binning strategy

2. **Calculates PSI**:
   - Measures relative difference between distributions
   - Weighted by log ratio
   - More sensitive to large changes

#### Interpretation
- **Range**: 0 to ∞
  - 0: No distribution change
  - < 0.1: Insignificant change
  - 0.1 - 0.2: Moderate change
  - > 0.2: Significant change

#### Example
For interest rates:
```python
# Historical: 3-5%
# Current: 4-6%
psi = 0.15  # Moderate change
```

#### Advantages
1. Industry standard in credit risk
2. Very sensitive to population shifts
3. Good for detecting concept drift
4. Interpretable thresholds

### Comparison of Drift Metrics

| Metric | Range | Strengths | Best For |
|--------|--------|-----------|----------|
| KS | 0-1 | Distribution-free, Scale-invariant | Detecting any distribution changes |
| JS | 0-1 | Symmetric, Smooth | Comparing probability distributions |
| PSI | 0-∞ | Industry standard, Sensitive | Detecting significant population shifts |

### Kullback-Leibler (KL) Divergence Monitoring

The system uses KL divergence with bootstrap testing to detect both feature and prediction distribution shifts:

#### Feature Distribution Monitoring
```python
# Compute KL divergence between reference and production data
kl_div = entropy(reference_distribution, production_distribution)

# Bootstrap for statistical significance
p_value = bootstrap_test(kl_div, n_samples=500)
```

1. **Sliding Windows**
   - Window size: 50 samples
   - Updates with each new prediction
   - Maintains separate windows per feature

2. **Statistical Testing**
   - Significance level: 0.1 (more sensitive)
   - Bootstrap samples: 500
   - P-value calculation for drift significance

3. **Metrics Tracked**
   - `refinance_model_feature_kl_divergence`
   - `refinance_model_feature_drift_p_value`
   - `refinance_model_feature_drift_detected`

#### Prediction Distribution Monitoring
```python
# Monitor shifts in model prediction distributions
prediction_monitor = PredictionDistributionMonitor(
    reference_predictions=training_predictions,
    window_size=50
)
```

1. **Metrics Tracked**
   - `refinance_model_prediction_distribution_shift`
   - `refinance_model_approval_rate`

2. **Alert Thresholds**
   - KL divergence > 0.5: Warning
   - KL divergence > 1.0: Critical
   - P-value < 0.1: Significant drift

#### Advantages of KL Divergence
1. **Sensitivity**: Detects subtle changes in distributions
2. **Interpretability**: Clear statistical significance
3. **Early Warning**: Catches drift before performance degrades
4. **Feature-specific**: Identifies which features are drifting

#### Example Drift Report
```
Distribution Drift Report
=========================

Feature: interest_rate
Status: DRIFT DETECTED
KL Divergence: 0.7523
P-value: 0.0234
Window Size: 50
Timestamp: 2025-02-13 14:15:23

## Technical Monitoring Details

### Service Architecture

1. **Model Monitor Service**
   - Port: 8000
   - Metrics endpoint: `/metrics`
   - Key responsibilities:
     - Model performance tracking
     - Prediction logging
     - Error monitoring
     - Request latency tracking

2. **Distribution Monitor Service**
   - Port: 8001
   - Metrics endpoint: `/metrics`
   - Key responsibilities:
     - Feature distribution analysis
     - Drift detection
     - Statistical tests
     - Distribution visualization data

3. **Prometheus**
   - Port: 9090
   - Scrape interval: 15s
   - Retention: 15 days
   - Key metrics:
     ```promql
     # Model Performance
     refinance_model_accuracy
     prediction_error_rate
     prediction_latency_seconds

     # Feature Drift
     refinance_model_feature_kl_divergence
     feature_drift{metric_type="ks_statistic"}
     ```

4. **Grafana**
   - Port: 3000
   - Refresh rate: 1m
   - Dashboards:
     - Model Performance
     - Feature Distributions
     - Operational Metrics

### Alert Configuration

```yaml
# Critical Alerts
- alert: CriticalFeatureDrift
  expr: refinance_model_feature_kl_divergence > 3
  for: 5m
  labels:
    severity: critical

- alert: LowModelAccuracy
  expr: refinance_model_accuracy < 0.95
  for: 5m
  labels:
    severity: critical

# Warning Alerts
- alert: HighFeatureDrift
  expr: refinance_model_feature_kl_divergence > 2
  for: 10m
  labels:
    severity: warning

- alert: MultipleFeaturesDrifting
  expr: count(refinance_model_feature_kl_divergence > 2) > 3
  for: 15m
  labels:
    severity: critical
```

### Current Drift Status (as of 2025-02-13)

Feature KL Divergence Values:
- Critical Drift (KL > 3.0):
  - loan_term: 25.32 (SEVERE)
  - current_rate: 3.44 (HIGH)

- High Drift (2.0 < KL < 3.0):
  - loan_to_value_ratio: 2.27
  - loan_amount: 2.02

- Moderate Drift (1.0 < KL < 2.0):
  - rate_spread: 1.91
  - loan_balance: 1.49
  - home_value: 1.42
  - credit_score: 1.14
  - interest_rate: 1.16
  - loan_age: 1.09

- Low Drift (KL < 1.0):
  - income: 0.59
  - debt_to_income_ratio: 0.12
  - predictions: 0.00

### Monitoring Best Practices

1. **Regular Checks**
   - Review drift metrics daily
   - Monitor prediction latency
   - Check error rates hourly
   - Validate ground truth data quality

2. **Alert Response**
   - Document all critical alerts
   - Investigate accuracy drops immediately
   - Track feature drift patterns
   - Monitor system resources

3. **Maintenance**
   - Update baseline distributions monthly
   - Clean old metrics data
   - Validate monitoring thresholds
   - Check service health daily

Feature: credit_score
Status: No significant drift
KL Divergence: 0.1234
P-value: 0.3456
Window Size: 50
Timestamp: 2025-02-13 14:15:23
```

## Architecture

The monitoring system consists of several components:

1. **Model Monitor Service**
   - Collects real-time metrics from model predictions
   - Exposes Prometheus metrics endpoint at `/metrics`
   - Calculates feature drift and model performance metrics
   - Located in `app/model_monitor.py`

2. **Prometheus**
   - Scrapes metrics from the model monitor service
   - Stores time-series data
   - Accessible at port 9090

3. **Grafana**
   - Visualizes monitoring metrics
   - Provides interactive dashboards
   - Accessible at port 3000

## Monitored Metrics

### 1. Model Performance Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- Error Rate

### 2. Feature Drift Metrics
- Kolmogorov-Smirnov (KS) Statistic
- Jensen-Shannon Divergence
- Population Stability Index (PSI)
- Tracked for each feature:
  - Interest Rates
  - Credit Scores
  - Loan-to-Value Ratios
  - Debt-to-Income Ratios
  - Income
  - Home Values

### 3. Feature Importance
- Relative importance of each feature
- Helps track if important features are drifting

### 4. Prediction Distribution
- Distribution of model prediction probabilities
- Helps identify prediction bias or shifts

## Accessing the Dashboards

### Grafana Dashboard
1. Access Grafana:
   ```bash
   kubectl port-forward -n monitoring svc/monitoring-grafana 3000:80
   ```
2. Open `http://localhost:3000`
3. Login with:
   - Username: admin
   - Password: admin

### Prometheus
1. Access Prometheus:
   ```bash
   kubectl port-forward -n monitoring svc/prometheus-operated 9090:9090
   ```
2. Open `http://localhost:9090`

## Testing and Drift Simulation

### Running Drift Tests
Use the drift simulation script to test monitoring:
```bash
python test_model.py --batch-size 50 --interval 5
```

This will:
1. Generate test data with gradual drift
2. Send prediction requests
3. Monitor feature distributions
4. Show drift metrics in real-time

### Drift Parameters
The drift simulation includes:
- Interest rates trending lower
- Credit scores improving
- More conservative LTV ratios
- Lower DTI ratios
- Higher loan amounts
- Higher home values
- Higher incomes
- Newer loans

## Useful PromQL Queries

### Feature Drift
```promql
# Jensen-Shannon divergence
feature_drift{metric_type="jensen_shannon_div"}

# KS statistic
feature_drift{metric_type="ks_statistic"}

# Population Stability Index
feature_drift{metric_type="psi"}
```

### Model Performance
```promql
# Overall accuracy
refinance_model_accuracy

# Precision and recall
refinance_model_precision
refinance_model_recall

# Top 3 most important features
topk(3, feature_importance)
```

## Alert Thresholds

### Feature Drift Alerts
- Minor drift: JS divergence > 0.1
- Moderate drift: JS divergence > 0.2
- Severe drift: JS divergence > 0.3

### Performance Alerts
- Accuracy < 0.95
- F1 Score < 0.90
- Error Rate > 0.1

## Troubleshooting

### Common Issues
1. Metrics not showing up:
   - Check if model monitor pod is running
   - Verify Prometheus scraping configuration
   - Check port forwarding

2. Dashboard access issues:
   - Ensure services are running
   - Check port forwarding
   - Verify Grafana credentials

3. High drift values:
   - Check recent data distribution changes
   - Verify data preprocessing
   - Review feature engineering pipeline

## Future Improvements

1. **Additional Metrics**
   - Concept drift detection
   - Feature correlation monitoring
   - Response time tracking
   - Resource utilization metrics

2. **Enhanced Alerting**
   - Slack integration
   - Email notifications
   - Custom alert thresholds per feature

3. **Dashboard Enhancements**
   - Custom drill-down views
   - Feature correlation heatmaps
   - Prediction explanation visualizations
