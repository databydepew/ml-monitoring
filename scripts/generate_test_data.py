#!/usr/bin/env python3
from google.cloud import bigquery
import numpy as np
from datetime import datetime, timedelta
import json

# Initialize BigQuery client
client = bigquery.Client(project='mdepew-assets')

# Generate test data
def generate_test_row():
    # Convert numpy types to Python native types
    return {
        'timestamp': datetime.now().isoformat(),
        'prediction': int(np.random.randint(0, 2)),
        'confidence': np.random.uniform(0.6, 0.95),
        'interest_rate': np.random.uniform(2.5, 6.0),
        'loan_amount': np.random.uniform(100000, 500000),
        'loan_balance': np.random.uniform(90000, 450000),
        'loan_to_value_ratio': np.random.uniform(0.6, 0.95),
        'credit_score': int(np.random.randint(580, 850)),
        'debt_to_income_ratio': np.random.uniform(0.2, 0.5),
        'income': np.random.uniform(50000, 200000),
        'loan_term': int(np.random.choice([15, 30])),
        'loan_age': int(np.random.randint(1, 10)),
        'home_value': np.random.uniform(150000, 750000),
        'current_rate': np.random.uniform(3.0, 7.0),
        'rate_spread': np.random.uniform(0.5, 2.0),
        'drift_metrics': json.dumps({
            'kl_divergence': np.random.uniform(0.1, 0.5),
            'ks_statistic': np.random.uniform(0.05, 0.2)
        })
    }

# Generate 100 test records
rows = [generate_test_row() for _ in range(100)]

# Insert into BigQuery
table_id = 'mdepew-assets.synthetic.model_predictions'
errors = client.insert_rows_json(table_id, rows)

if errors:
    print(f"Encountered errors while inserting rows: {errors}")
else:
    print(f"Successfully inserted {len(rows)} rows into {table_id}")
