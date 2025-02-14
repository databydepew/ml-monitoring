# Loan Approval Classification Model - MLOps Pipeline with KL Divergence Monitoring

## Introduction
This repository demonstrates how to integrate KL Divergence monitoring into an MLOps pipeline for a **loan approval classification model**. KL Divergence helps detect **data drift, model drift, and anomalies**, ensuring that the model remains reliable and up-to-date.

## Problem Statement
Machine learning models deployed in production can degrade over time due to:
- **Data Drift**: Changes in feature distributions (e.g., changes in applicant income or credit scores).
- **Model Drift**: Shifts in model predictions (e.g., increase in loan approval rates without a known cause).
- **Anomalies**: Fraudulent or unexpected data patterns.

KL Divergence is used to monitor these changes and trigger alerts or retraining when necessary.

## KL Divergence in MLOps
KL Divergence measures how different a probability distribution is from another. In MLOps, it is used to:
1. Detect shifts in **feature distributions** between training and production.
2. Identify **model drift** by comparing historical vs. current prediction distributions.
3. Find **anomalies** in real-time data.

## Use Cases

### 1. Detecting Data Drift (Feature Distribution Shift)
**Example:** The training data had applicants with an average income of $50,000, but in production, most applicants have incomes above $80,000.

```python
from scipy.stats import entropy
import numpy as np

# Training data distribution for income levels
P_income = np.array([0.1, 0.3, 0.4, 0.2])  # Buckets: [Low, Medium, High, Very High]

# Production data distribution for income levels
Q_income = np.array([0.05, 0.2, 0.5, 0.25])

# Compute KL divergence
kl_div_income = entropy(P_income, Q_income)
print(f"KL Divergence (Income Distribution Shift): {kl_div_income}")
```

### 2. Detecting Model Drift (Prediction Shift)
**Example:** In training, 40% of loan applications were approved. In production, 70% are being approved.

```python
# Training prediction distribution (approved/rejected)
P_pred = np.array([0.6, 0.4])  # 60% rejected, 40% approved

# Production prediction distribution
Q_pred = np.array([0.3, 0.7])  # 30% rejected, 70% approved

# Compute KL divergence
kl_div_pred = entropy(P_pred, Q_pred)
print(f"KL Divergence (Model Drift): {kl_div_pred}")
```

### 3. Anomaly Detection in Loan Applications
**Example:** A fraud attack causes a spike in high credit scores.

```python
# Previous normal distribution of credit scores
P_credit = np.array([0.05, 0.15, 0.4, 0.3, 0.1])  # Buckets: [Very Low, Low, Medium, High, Very High]

# New distribution in production (fraud detected)
Q_credit = np.array([0.01, 0.05, 0.2, 0.5, 0.24])

# Compute KL divergence
kl_div_credit = entropy(P_credit, Q_credit)
print(f"KL Divergence (Credit Score Anomaly Detection): {kl_div_credit}")
```

## Automating KL Divergence Monitoring in MLOps
### Steps:
1. Schedule a batch job to compute KL divergence daily/weekly.
2. Set an alerting threshold (e.g., KL divergence > 0.1 triggers an alert).
3. Trigger retraining or rollback if KL divergence remains high for a period.

### Tools for Integration:
- **Evidently AI**: Automated drift detection and monitoring.
- **Great Expectations**: Data validation framework.
- **MLflow / Weights & Biases**: Model tracking and monitoring.
- **Prometheus & Grafana**: Alerting on model drift.

## Summary
| Use Case        | What KL Divergence Detects | MLOps Action |
|----------------|-----------------------------|--------------|
| Data Drift (Feature Shift) | Changes in income, credit score, employment type distribution | Alert and retrain the model |
| Model Drift (Prediction Shift) | Unexpected increase in loan approvals | Investigate and retrain if necessary |
| Anomaly Detection | Fraudulent or unusual applicants appearing | Raise a fraud alert |

KL Divergence helps maintain the reliability and fairness of a loan approval model. Automating this metric in an MLOps pipeline ensures that the model adapts to real-world changes without performance degradation.

## Next Steps
- Integrate KL Divergence monitoring with real-time data pipelines.
- Set up automated alerts for drift detection.
- Implement retraining workflows when KL divergence exceeds thresholds.

------------
# Why Compare Distributions Instead of Individual Data Points in MLOps?

In production machine learning systems, particularly for loan approval models, comparing distributions rather than individual data points provides significant advantages for monitoring and maintaining model health. This document explains why distribution-based monitoring using KL divergence is superior to point-based comparisons.

## 1. Robust Detection of Data Drift

### Problem with Comparing Individual Data Points
- Individual data points vary naturally due to randomness
- Minor differences in input values do not necessarily indicate data drift

### Why Comparing Distributions is Better
- Distributions provide an aggregate view of data behavior over time
- Detects gradual shifts in input data (e.g., credit score distributions)
- Handles large datasets efficiently by summarizing data changes statistically

### Example: Monitoring Income Distribution in Loan Applications
Instead of comparing every applicant's income separately, KL divergence detects if the income distribution has shifted between training and production:

```python
from scipy.stats import entropy
import numpy as np

# Training distribution (percentages of applicants in income brackets)
P_income = np.array([0.1, 0.3, 0.4, 0.2])

# Production distribution (new incoming applications)
Q_income = np.array([0.05, 0.2, 0.5, 0.25])

kl_div_income = entropy(P_income, Q_income)
print(f"KL Divergence (Income Distribution Shift): {kl_div_income}")
```

## 2. More Meaningful Model Drift Detection

### Problem with Comparing Individual Predictions
- Individual predictions vary even when the model is stable
- A model making one wrong loan approval does not indicate drift
- Changes in input noise or minor fluctuations do not mean model failure

### Why Comparing Prediction Distributions is Better
- Detects shifts in model decision-making over time
- Captures changes in prediction confidence
- Avoids false alarms due to normal variations

### Example: Monitoring Loan Approval Trends
If a model's approval rate shifts from 40% to 70%, KL divergence detects this change:

```python
P_pred = np.array([0.6, 0.4])  # Training: [Rejected, Approved]
Q_pred = np.array([0.3, 0.7])  # Production: More approvals

kl_div_pred = entropy(P_pred, Q_pred)
print(f"KL Divergence (Model Drift): {kl_div_pred}")
```

## 3. Detects Concept Drift More Effectively

### Problem with Comparing Raw Data
- Input features may remain stable while input-output relationships change
- Policy changes affecting loan approvals may not be visible in raw data

### Why Comparing Distributions is Better
- Identifies shifts in the relationship between inputs and outputs
- Helps track regulatory changes that affect loan approvals
- Reduces false detections from outliers

### Example: Credit Scores vs. Loan Approvals
- Previously: Most approvals for credit scores > 700
- Now: More approvals for scores 600-650
- Joint distribution comparison reveals this concept drift

## 4. Efficient Computation for Large-Scale Monitoring

### Problem with Raw Data Comparison
- High-dimensional and high-volume data
- Inefficient comparison of millions of data points
- High storage and computation costs

### Why Comparing Distributions is Better
- Reduces dimensionality by summarizing key patterns
- Enables scalable, real-time monitoring
- Requires only statistical computations rather than large-scale processing

## 5. Aligns with Business KPIs and Interpretability

### Problem with Individual Data Points
- Stakeholders need high-level insights
- Raw data comparisons are hard to interpret

### Why Comparing Distributions is Better
- Aligns with business goals (e.g., "Are we approving more risky loans?")
- Provides interpretable metrics (KL divergence, Jensen-Shannon divergence)
- Helps assess fairness and bias in loan processes

## Summary: Distribution Comparison Benefits

| Benefit | Description |
|---------|-------------|
| Data Drift Detection | Summarizes feature distribution shifts |
| Model Drift Detection | Tracks overall model behavior changes |
| Concept Drift Detection | Identifies input-output relationship changes |
| Computational Efficiency | Scales well with large ML systems |
| Business Alignment | Provides actionable insights |

Using KL divergence and distribution-based monitoring ensures that machine learning models remain stable, fair, and aligned with business objectives.

## Computing Reference Distributions During Training

To effectively monitor distribution shifts in production, we need to first compute and store reference distributions during model training. Here's how to implement this:

### 1. Feature Distributions

```python
def compute_reference_distributions(training_data: pd.DataFrame, n_bins: int = 50):
    """Compute reference distributions for each feature"""
    distributions = {}
    
    for column in training_data.columns:
        # Compute histogram for the feature
        hist, bin_edges = np.histogram(
            training_data[column],
            bins=n_bins,
            density=True
        )
        
        # Store normalized distribution and bins
        distributions[column] = {
            'distribution': hist / hist.sum(),  # Normalize
            'bins': bin_edges
        }
    
    return distributions

# Example usage during training
reference_distributions = compute_reference_distributions(train_df)
```

### 2. Prediction Distributions

```python
def compute_prediction_distribution(model_predictions: np.ndarray, n_bins: int = 50):
    """Compute reference distribution for model predictions"""
    # For binary classification, use binary bins
    if len(np.unique(model_predictions)) == 2:
        hist = np.bincount(model_predictions.astype(int))
        return hist / hist.sum()
    
    # For continuous predictions, use histogram
    hist, _ = np.histogram(model_predictions, bins=n_bins, density=True)
    return hist / hist.sum()

# Example usage during training
val_predictions = model.predict(val_df)
reference_pred_dist = compute_prediction_distribution(val_predictions)
```

### 3. Store Reference Data

Reference distributions can be stored in various ways:

1. **BigQuery (Recommended for Production)**:
```python
from google.cloud import bigquery

def store_reference_distributions(project_id: str, dataset_id: str):
    client = bigquery.Client(project=project_id)
    
    # Create reference distributions table
    schema = [
        bigquery.SchemaField("feature_name", "STRING"),
        bigquery.SchemaField("distribution", "FLOAT64", mode="REPEATED"),
        bigquery.SchemaField("bins", "FLOAT64", mode="REPEATED"),
        bigquery.SchemaField("timestamp", "TIMESTAMP")
    ]
    
    table_id = f"{project_id}.{dataset_id}.reference_distributions"
    table = bigquery.Table(table_id, schema=schema)
    table = client.create_table(table, exists_ok=True)
```

2. **Local Storage (Development)**:
```python
import joblib

# Save distributions
joblib.dump(reference_distributions, 'reference_distributions.joblib')
```

### 4. Configuration for Monitoring

When setting up monitoring, provide the reference distributions:

```python
from app.helpers.distribution_monitor import DistributionMonitor

# Initialize monitor with reference data
monitor = DistributionMonitor(
    reference_data=reference_df,
    window_size=1000,
    significance_level=0.05,
    bigquery_config={
        'project_id': 'your-project',
        'dataset_id': 'your-dataset',
        'table_id': 'reference_distributions'
    }
)
```

This setup enables continuous comparison between production data and the reference distributions computed during training.

## Handling Categorical Features

Categorical features require special handling when computing distributions. Here's how to handle different types of categorical data:

### 1. Nominal Categorical Features

```python
def compute_categorical_distribution(data: pd.Series):
    """Compute distribution for nominal categorical features"""
    # Get value counts and normalize
    counts = data.value_counts(normalize=True)
    
    # Convert to fixed-size vector using one-hot encoding for new categories
    def get_distribution(new_data: pd.Series) -> np.ndarray:
        new_counts = new_data.value_counts(normalize=True)
        # Handle new categories by adding them with zero frequency
        all_categories = set(counts.index) | set(new_counts.index)
        distribution = np.zeros(len(all_categories))
        for i, cat in enumerate(sorted(all_categories)):
            distribution[i] = counts.get(cat, 0)
        return distribution
    
    return get_distribution

# Example usage
categorical_features = ['loan_type', 'property_type', 'occupation']
cat_distributions = {}

for feature in categorical_features:
    cat_distributions[feature] = compute_categorical_distribution(train_df[feature])
```

### 2. Ordinal Categorical Features

```python
def compute_ordinal_distribution(data: pd.Series, ordering: List[str]):
    """Compute distribution for ordinal categorical features"""
    # Map categories to ordered integers
    order_map = {cat: i for i, cat in enumerate(ordering)}
    numeric_data = data.map(order_map)
    
    # Compute histogram with fixed bins based on ordering
    hist, _ = np.histogram(
        numeric_data,
        bins=len(ordering),
        range=(-0.5, len(ordering) - 0.5),
        density=True
    )
    
    return hist / hist.sum()

# Example usage
credit_rating_order = ['Poor', 'Fair', 'Good', 'Excellent']
dist = compute_ordinal_distribution(train_df['credit_rating'], credit_rating_order)
```

### 3. High-Cardinality Categorical Features

```python
def compute_high_cardinality_distribution(data: pd.Series, max_categories: int = 50):
    """Handle high-cardinality categorical features"""
    # Get top categories and group others
    top_categories = data.value_counts(normalize=True)
    major_categories = top_categories.nlargest(max_categories - 1)
    other_freq = 1 - major_categories.sum()
    
    # Create distribution with 'Other' category
    distribution = np.zeros(max_categories)
    for i, (cat, freq) in enumerate(major_categories.items()):
        distribution[i] = freq
    distribution[-1] = other_freq  # 'Other' category
    
    return distribution

# Example usage for ZIP codes
zip_dist = compute_high_cardinality_distribution(train_df['zip_code'])
```

### 4. Monitoring Categorical Shifts

```python
def monitor_categorical_drift(reference_dist: np.ndarray, 
                             production_dist: np.ndarray,
                             feature_name: str,
                             threshold: float = 0.1):
    """Monitor distribution shifts in categorical features"""
    kl_div = entropy(reference_dist, production_dist)
    
    # Compute Jensen-Shannon divergence (symmetric version of KL)
    m = 0.5 * (reference_dist + production_dist)
    js_div = 0.5 * (entropy(reference_dist, m) + entropy(production_dist, m))
    
    return {
        'feature': feature_name,
        'kl_divergence': kl_div,
        'js_divergence': js_div,
        'drift_detected': js_div > threshold
    }
```

These methods ensure proper handling of categorical features in distribution monitoring, accounting for different types of categorical data and their specific characteristics.

## Monitoring Model Accuracy Drift with KL Divergence

While traditional accuracy metrics are useful, KL divergence can provide early warnings of model accuracy degradation by monitoring the distribution of correct vs. incorrect predictions over time.

### 1. Computing Accuracy Distribution

```python
def compute_accuracy_distribution(y_true: np.ndarray,
                                y_pred: np.ndarray,
                                window_size: int = 1000) -> np.ndarray:
    """Compute distribution of correct/incorrect predictions"""
    # Create sliding window of accuracy values
    correct_predictions = (y_true == y_pred).astype(int)
    
    # Compute proportion of correct/incorrect predictions
    n_correct = np.sum(correct_predictions[-window_size:])
    n_total = len(correct_predictions[-window_size:])
    
    # Return [incorrect_rate, correct_rate]
    return np.array([(n_total - n_correct) / n_total, n_correct / n_total])

# Example: Reference accuracy distribution from validation set
ref_accuracy_dist = compute_accuracy_distribution(val_y_true, val_y_pred)
```

### 2. Monitoring Accuracy Drift

```python
def monitor_accuracy_drift(reference_dist: np.ndarray,
                          production_dist: np.ndarray,
                          threshold: float = 0.1) -> dict:
    """Monitor shifts in accuracy distribution"""
    # Compute KL divergence
    kl_div = entropy(reference_dist, production_dist)
    
    # Compute actual accuracy difference
    acc_diff = abs(reference_dist[1] - production_dist[1])
    
    return {
        'kl_divergence': kl_div,
        'accuracy_difference': acc_diff,
        'drift_detected': kl_div > threshold
    }

# Example usage in production
production_accuracy_dist = compute_accuracy_distribution(prod_y_true, prod_y_pred)
results = monitor_accuracy_drift(ref_accuracy_dist, production_accuracy_dist)
```

### 3. Advantages of Distribution-Based Accuracy Monitoring

1. **Early Warning System**:
   - Detects subtle shifts in accuracy patterns before they become significant
   - More sensitive than simple accuracy thresholds

2. **Window-Based Analysis**:
   - Captures temporal patterns in model performance
   - Reduces noise through aggregation

3. **Multiple Metrics**:
   ```python
def compute_detailed_accuracy_distribution(y_true: np.ndarray,
                                         y_pred: np.ndarray,
                                         y_prob: np.ndarray,
                                         bins: int = 10) -> dict:
    """Compute detailed accuracy distribution metrics"""
    # Confidence distribution
    confidence_hist, _ = np.histogram(y_prob, bins=bins, density=True)
    
    # Accuracy by confidence level
    correct = (y_true == y_pred)
    acc_by_conf = []
    for i in range(bins):
        mask = (y_prob >= i/bins) & (y_prob < (i+1)/bins)
        if np.any(mask):
            acc_by_conf.append(np.mean(correct[mask]))
        else:
            acc_by_conf.append(0)
    
    return {
        'confidence_distribution': confidence_hist,
        'accuracy_by_confidence': np.array(acc_by_conf),
        'overall_distribution': compute_accuracy_distribution(y_true, y_pred)
    }
```

4. **Integration with Prometheus**:
```python
# Prometheus metrics for accuracy monitoring
accuracy_drift_gauge = Gauge(
    'model_accuracy_drift',
    'KL divergence between reference and current accuracy distributions'
)

accuracy_by_confidence_gauge = Gauge(
    'model_accuracy_by_confidence',
    'Accuracy for different confidence levels',
    ['confidence_bin']
)
```

This approach provides a more nuanced view of model accuracy degradation, allowing for early detection of performance issues and better understanding of where and how the model's accuracy is changing.

-------
