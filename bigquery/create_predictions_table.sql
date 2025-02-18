CREATE TABLE IF NOT EXISTS `mdepew-assets.ml_monitoring.model_predictions`
(
  timestamp TIMESTAMP,
  prediction INT64,
  confidence FLOAT64,
  interest_rate FLOAT64,
  loan_amount FLOAT64,
  loan_balance FLOAT64,
  loan_to_value_ratio FLOAT64,
  credit_score INT64,
  debt_to_income_ratio FLOAT64,
  income FLOAT64,
  loan_term INT64,
  loan_age INT64,
  home_value FLOAT64,
  current_rate FLOAT64,
  rate_spread FLOAT64,
  drift_metrics STRING
)
PARTITION BY DATE(timestamp);
