#!/usr/bin/env python3
import requests
import time
import random
from datetime import datetime

def send_feedback(prediction, actual):
    """Send feedback to the model monitoring service."""
    url = "http://localhost:5000/feedback"
    payload = {
        "prediction": prediction,
        "actual": actual
    }
    
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        metrics = response.json()["metrics"]
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Feedback sent - "
              f"Prediction: {prediction}, Actual: {actual}")
        print(f"Current Metrics: Accuracy={metrics['accuracy']:.3f}, "
              f"Precision={metrics['precision']:.3f}, "
              f"Recall={metrics['recall']:.3f}, "
              f"F1={metrics['f1_score']:.3f}")
    else:
        print(f"Error sending feedback: {response.text}")

def send_sample_feedback():
    """Send sample feedback to the model monitoring service."""
    sample_data = {
        'actual_outcome': '1',
        'prediction': '1'
    }
    
    response = requests.post('http://localhost:5000/feedback', json=sample_data)
    print('Response:', response.json())

def main():
    # Scenario 1: Initially good predictions
    print("\n=== Scenario 1: Good Predictions ===")
    for _ in range(5):
        actual = random.choice([0, 1])
        send_feedback(actual, actual)  # Perfect predictions
        time.sleep(1)
    
    # Scenario 2: Some mistakes
    print("\n=== Scenario 2: Mixed Performance ===")
    for _ in range(5):
        actual = random.choice([0, 1])
        # 80% chance of correct prediction
        prediction = actual if random.random() < 0.8 else 1 - actual
        send_feedback(prediction, actual)
        time.sleep(1)
    
    # Scenario 3: Poor predictions
    print("\n=== Scenario 3: Poor Performance ===")
    for _ in range(5):
        actual = random.choice([0, 1])
        # 40% chance of correct prediction
        prediction = actual if random.random() < 0.4 else 1 - actual
        send_feedback(prediction, actual)
        time.sleep(1)
    
    # Scenario 4: Recovery
    print("\n=== Scenario 4: Recovery ===")
    for _ in range(5):
        actual = random.choice([0, 1])
        # 90% chance of correct prediction
        prediction = actual if random.random() < 0.9 else 1 - actual
        send_feedback(prediction, actual)
        time.sleep(1)
    
    # Send sample feedback
    print("\n=== Sending Sample Feedback ===")
    send_sample_feedback()


def write_to_bigquery(data):
    from google.cloud import bigquery
    import logging
    from datetime import datetime

    # Initialize the BigQuery client
    bq_client = bigquery.Client(project='mdepew-assets')

    table_id = 'mdepew-assets.synthetic.model_prediction_feedback'  # Update with your actual table ID

    # Prepare the row to insert
    row = {
        'timestamp': datetime.now().isoformat(),
        'prediction': data['prediction'],
        'actual': data['actual'],
        'confidence': data.get('confidence', None),  # Optional field
        'drift_metrics': data.get('drift_metrics', None) , # Optional field
        'loan_term': data['loan_term'],
        'interest_rate': data['interest_rate'],
        'loan_amount': data['loan_amount'],
        'loan_balance': data['loan_balance'],
        'loan_to_value_ratio': data['loan_to_value_ratio'],
        'credit_score': data['credit_score'],
        'debt_to_income_ratio': data['debt_to_income_ratio'],
        'income': data['income'],
        'loan_age': data['loan_age'],
        'home_value': data['home_value'],
        'current_rate': data['current_rate'],
        'rate_spread': data['rate_spread'],
        
    }

    try:
        # Insert the row into BigQuery
        errors = bq_client.insert_rows_json(table_id, [row])
        if errors:
            logging.error(f"Failed to write to BigQuery: {errors}")
        else:
            logging.info("Successfully wrote feedback data to BigQuery.")
    except Exception as e:
        logging.error(f"Error writing to BigQuery: {str(e)}")

if __name__ == "__main__":
    print("Starting feedback simulation...")
    print("This will send 20 feedback requests with varying accuracy patterns")
    print("You can monitor the metrics in Prometheus (http://localhost:9090)")
    print("Query to watch: model_accuracy\n")
    
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the model service.")
        print("Make sure the service is running on http://localhost:5000")
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
