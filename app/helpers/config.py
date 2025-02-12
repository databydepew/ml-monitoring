"""Configuration for the model monitoring system."""

# BigQuery Configuration
BIGQUERY_CONFIG = {
    'project_id': 'mdepew-assets',
    'dataset_id': 'synthetic',
    'table_id': 'synthetic_mortgage_data',
    'target_column': 'refinance',  # Binary classification target
    'feature_columns': [
        'interest_rate',
        'loan_amount',
        'loan_balance',
        'loan_to_value_ratio',
        'credit_score',
        'debt_to_income_ratio',
        'income',
        'loan_term',
        'loan_age',
        'home_value',
        'current_rate',
        'rate_spread'
    ]
}

# Monitoring Configuration
MONITORING_CONFIG = {
    'window_size': 1000,  # Size of sliding window for production data
    'significance_level': 0.05,  # P-value threshold for drift detection
    'n_bootstrap': 1000,  # Number of bootstrap samples for significance testing
    'drift_thresholds': {
        'low': 0.1,    # Minimal drift
        'medium': 0.3, # Moderate drift
        'high': 0.5    # Severe drift requiring immediate attention
    }
}

# Feature Groups for Monitoring
FEATURE_GROUPS = {
    'loan_characteristics': [
        'loan_amount',
        'loan_balance',
        'loan_term',
        'loan_age'
    ],
    'risk_metrics': [
        'credit_score',
        'debt_to_income_ratio',
        'loan_to_value_ratio'
    ],
    'financial_indicators': [
        'interest_rate',
        'current_rate',
        'rate_spread',
        'income',
        'home_value'
    ]
}
