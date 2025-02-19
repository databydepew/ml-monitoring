CREATE OR REPLACE TABLE `mdepew-assets.synthetic.model_predictions` (
  prediction FLOAT64,  -- Changed from INT64 to FLOAT64
  confidence FLOAT64,
  interest_rate FLOAT64,
  loan_amount FLOAT64,
  loan_balance FLOAT64,
  loan_to_value_ratio FLOAT64,
  credit_score FLOAT64,  -- Changed from INT64 to FLOAT64
  debt_to_income_ratio FLOAT64,
  income FLOAT64,
  loan_term FLOAT64,  -- Changed from INT64 to FLOAT64
  loan_age FLOAT64,  -- Changed from INT64 to FLOAT64
  home_value FLOAT64,
  current_rate FLOAT64,
  rate_spread FLOAT64,
  drift_metrics STRING,  -- Kept as STRING (could be JSON if needed)
  actual FLOAT64  -- Changed from INT64 to FLOAT64
) OPTIONS (
  description="Table with looser types and nullable fields"
);
