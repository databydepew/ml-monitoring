Low Refinance Probability Case (8% probability)
High credit score (800)
Small rate spread (0.3%)
Low debt-to-income ratio (0.25)
Result: Correctly predicted NO refinance
High Refinance Probability Case (83% probability)
Lower credit score (680)
High rate spread (3.0%)
Higher debt-to-income ratio (0.45)
Result: Correctly predicted WILL refinance
15-year Term Case (81% probability)
Good credit score (760)
Moderate rate spread (1.7%)
Low loan-to-value ratio (0.65)
Result: Predicted WILL refinance despite shorter term
Key Monitoring Insights:

Feature Drift Levels:
Critical (KL > 3.0):
loan_term: 25.32 (extremely high)
current_rate: 3.44
High (2.0 < KL < 3.0):
loan_to_value_ratio: 2.27
loan_amount: 2.01
Moderate (1.0 < KL < 2.0):
rate_spread: 1.91
loan_balance: 1.49
home_value: 1.41
credit_score: 1.13
interest_rate: 1.15
Low (KL < 1.0):
income: 0.58
debt_to_income_ratio: 0.11
Model Performance:
Accuracy remains at 100%
No prediction errors detected
Predictions distribution shows no significant drift
Notable Patterns:
The model is sensitive to rate spread differences
Credit score has less impact than rate spread on refinance decisions
The 15-year term case still predicted refinance, showing that rate spread may be more important than term length


============
Predictions Made:
CopyInsert
1. Prediction: WILL refinance (prob: 81%)
2. Prediction: WILL refinance (prob: 83%)
3. Prediction: WON'T refinance (prob: 8%)
Feature Drift Analysis (KL divergence values, ordered by severity):
Critical Drift (KL > 3.0):
loan_term: 25.32 (SEVERE)
current_rate: 3.44 (HIGH)
High Drift (2.0 < KL < 3.0):
loan_to_value_ratio: 2.27
loan_amount: 2.02
Moderate Drift (1.0 < KL < 2.0):
rate_spread: 1.91
loan_balance: 1.49
home_value: 1.42
credit_score: 1.14
interest_rate: 1.16
loan_age: 1.09
Low Drift (KL < 1.0):
income: 0.59
debt_to_income_ratio: 0.12
predictions: 0.00
Key Insights:
The model shows extreme drift in loan_term (25.32), suggesting the current loan term distribution is very different from training data
current_rate drift (3.44) indicates significant changes in interest rate environment
Multiple features show high drift (>2.0), suggesting widespread distribution shifts
The model maintains high confidence (81-83%) for positive predictions despite significant feature drift
Potential Issues:
The extreme drift in loan_term could indicate a systematic shift in loan products being offered
High current_rate drift suggests the model may not be well-calibrated for current market conditions
Multiple features with high drift indicate potential model decay
Recommendations:
Investigate why loan_term shows such extreme drift
Consider retraining the model with more recent data to account for current market conditions
Add more monitoring for loan term and current rate distributions
Implement confidence calibration to adjust probability scores