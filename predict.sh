for i in {1..5}; do
  curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "income": 75000,
    "loan_amount": 25000,
    "loan_term": 36,
    "credit_score": 720,
    "employment_status": 1,
    "loan_purpose": 1
  }'
  echo ""
  sleep 1
done