from google.cloud import bigquery
client = bigquery.Client(project='mdepew-assets')
table_id = 'mdepew-assets.synthetic.model_predictions'
rows = [{
    'timestamp': datetime.now(),
    'prediction': 1,
    'confidence': 0.81,
    'interest_rate': 4.5,
    'loan_amount': 300000.0,
    'loan_balance': 290000.0,
    'loan_to_value_ratio': 0.8,
    'credit_score': 720,
    'debt_to_income_ratio': 0.3,
    'income': 100000.0,
    'loan_term': 30,
    'loan_age': 5,
    'home_value': 400000.0,
    'current_rate': 5.5,
    'rate_spread': 1.0,
    'drift_metrics': '{}'
}]
errors = client.insert_rows_json(table_id, rows)