"""Configuration for the refinance classification model monitoring system.

This configuration file defines the settings and parameters for monitoring a mortgage
refinance classification model. The system tracks various aspects of refinance applications
including:

1. Borrower Characteristics:
   - Credit scores
   - Debt-to-income ratios
   - Income levels

2. Loan Characteristics:
   - Current loan balance and terms
   - Loan-to-value ratios
   - Loan age

3. Rate Information:
   - Current interest rates
   - Offered refinance rates
   - Rate spread (potential savings)

The monitoring system uses these configurations to detect data drift and ensure
the model maintains its performance in production.
"""

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

# Refinance Model Monitoring Configuration
MONITORING_CONFIG = {
    'window_size': 1000,          # Number of recent refinance applications to analyze
    'significance_level': 0.05,    # Statistical significance for drift detection
    'n_bootstrap': 1000,          # Bootstrap samples for significance testing
    'drift_thresholds': {
        'low': 0.1,    # Minor changes in refinance patterns
        'medium': 0.3,  # Significant shifts in refinance behavior
        'high': 0.5     # Major market changes affecting refinance patterns
    },
    'critical_metrics': {         # Metrics requiring immediate attention if drifting
        'rate_spread': 0.25,      # Significant change in rate advantages
        'ltv_ratio': 0.15,        # Sharp shifts in property values or loan balances
        'credit_score': 30         # Notable changes in applicant credit profiles
    }
}

# Feature Groups for Refinance Monitoring
FEATURE_GROUPS = {
    'loan_characteristics': [
        'loan_amount',      # Current loan amount to refinance
        'loan_balance',     # Remaining balance on existing loan
        'loan_term',        # Requested new loan term (months)
        'loan_age'          # Age of existing loan (months)
    ],
    'borrower_risk_metrics': [
        'credit_score',           # Borrower's current credit score
        'debt_to_income_ratio',   # Monthly debt payments / monthly income
        'loan_to_value_ratio'     # Loan balance / home value
    ],
    'rate_metrics': [
        'current_rate',     # Current loan interest rate
        'interest_rate',     # Offered refinance rate
        'rate_spread',       # Potential interest rate savings
        'home_value'         # Current property value
    ],
    'income_metrics': [
        'income'            # Annual income for debt service calculation
    ]
}
