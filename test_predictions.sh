curl -X POST http://localhost:5001/predict -H "Content-Type: application/json" -d '{
  "interest_rate": 4.8,
  "loan_amount": 200000,
  "loan_balance": 180000,
  "loan_to_value_ratio": 0.65,
  "credit_score": 760,
  "debt_to_income_ratio": 0.3,
  "income": 130000,
  "loan_term": 15,
  "loan_age": 4,
  "home_value": 310000,
  "current_rate": 6.5,
  "rate_spread": 1.7
}'

curl -X POST http://localhost:5001/predict -H "Content-Type: application/json" -d '{
  "interest_rate": 4.2,
  "loan_amount": 400000,
  "loan_balance": 385000,
  "loan_to_value_ratio": 0.85,
  "credit_score": 680,
  "debt_to_income_ratio": 0.45,
  "income": 95000,
  "loan_term": 30,
  "loan_age": 3,
  "home_value": 470000,
  "current_rate": 7.2,
  "rate_spread": 3.0
}'


curl -X POST http://localhost:5001/predict -H "Content-Type: application/json" -d '{
  "interest_rate": 5.5,
  "loan_amount": 250000,
  "loan_balance": 240000,
  "loan_to_value_ratio": 0.7,
  "credit_score": 800,
  "debt_to_income_ratio": 0.25,
  "income": 150000,
  "loan_term": 30,
  "loan_age": 2,
  "home_value": 350000,
  "current_rate": 5.8,
  "rate_spread": 0.3
}'


echo ""
echo "=== Prediction ==="

echo "=== Feature KL Divergence ===" && \
curl -s 'http://localhost:9090/api/v1/query?query=refinance_model_feature_kl_divergence' | jq .