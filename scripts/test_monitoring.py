import requests
import json
import time
import random
import pandas as pd
from typing import Dict
import numpy as np

def generate_random_sample() -> Dict:
    """Generate a random sample within reasonable ranges."""
    return {
        "interest_rate": random.uniform(2.5, 6.0),
        "loan_amount": random.uniform(100000, 1000000),
        "loan_balance": random.uniform(90000, 900000),
        "loan_to_value_ratio": random.uniform(0.5, 0.95),
        "credit_score": random.randint(580, 850),
        "debt_to_income_ratio": random.uniform(0.1, 0.5),
        "income": random.uniform(50000, 300000),
        "loan_term": random.choice([15, 30]),
        "loan_age": int(round(random.uniform(0, 10) *12)),
        "home_value": random.uniform(150000, 1500000),
        "current_rate": random.uniform(2.5, 7.0),
        "rate_spread": random.uniform(0.1, 2.0)
    }

def test_prediction_endpoint(n_requests: int = 100):
    """Test the prediction endpoint with multiple requests."""
    url = "http://localhost:5000/predict"
    headers = {"Content-Type": "application/json"}
    
    predictions = []
    response_times = []
    
    print(f"Sending {n_requests} requests to the model...")
    
    for i in range(n_requests):
        data = generate_random_sample()
        start_time = time.time()
        
        try:
            response = requests.post(url, headers=headers, json=data)
            response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            response_times.append(response_time)
            
            if response.status_code == 200:
                result = response.json()
                predictions.append(result['prediction'])
                print(f"Request {i+1}: Success - Prediction: {result['prediction']}, Response Time: {response_time:.2f}ms")
            else:
                print(f"Request {i+1}: Failed with status code {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"Request {i+1}: Error - {str(e)}")
        
        # Add a small delay between requests
        time.sleep(0.1)

    return predictions, response_times

def get_metrics():
    """Fetch and display metrics from the metrics endpoint."""
    try:
        response = requests.get("http://localhost:8001/metrics")
        if response.status_code == 200:
            print("\nMetrics:")
            print(response.text)
        else:
            print(f"Failed to fetch metrics. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching metrics: {str(e)}")

def analyze_results(predictions, response_times):
    """Analyze and display test results."""
    print("\nTest Results Summary:")
    print("-" * 50)
    print(f"Total Requests: {len(predictions)}")
    print(f"Average Response Time: {np.mean(response_times):.2f}ms")
    print(f"95th Percentile Response Time: {np.percentile(response_times, 95):.2f}ms")
    print(f"Prediction Distribution:")
    print(pd.Series(predictions).value_counts().to_string())

def main():
    # Run tests
    predictions, response_times = test_prediction_endpoint(100)
    
    # Analyze results
    analyze_results(predictions, response_times)
    
    # Get metrics after load test
    get_metrics()

if __name__ == "__main__":
    main()