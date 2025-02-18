#!/usr/bin/env python3
import numpy as np
import requests
import time
from datetime import datetime
import json

# Base feature ranges for normal distribution
FEATURE_RANGES = {
    'interest_rate': (2.5, 5.0),
    'loan_amount': (200000, 800000),
    'loan_balance': (180000, 750000),
    'loan_to_value_ratio': (0.6, 0.95),
    'credit_score': (580, 850),
    'debt_to_income_ratio': (0.2, 0.5),
    'income': (60000, 300000),
    'loan_term': (15, 30),
    'loan_age': (0, 10),
    'home_value': (250000, 1000000),
    'current_rate': (3.0, 6.0),
    'rate_spread': (0.5, 2.0)
}

def generate_sample(phase, drift_magnitude=0.0):
    """Generate a sample with optional drift."""
    sample = {}
    
    for feature, (min_val, max_val) in FEATURE_RANGES.items():
        # Base distribution
        if feature in ['loan_term']:  # Categorical features
            value = int(np.random.choice([15, 30]))  # Convert to int
        else:
            mean = (max_val + min_val) / 2
            std = (max_val - min_val) / 6  # Cover most of the range within 3 std
            
            # Apply drift based on phase
            if phase == 'drift':
                # Shift mean and increase variance
                mean += drift_magnitude * (max_val - min_val)
                std *= (1 + drift_magnitude)
            
            value = float(np.random.normal(mean, std))  # Convert to float
            # Clip to valid range
            value = float(np.clip(value, min_val, max_val))  # Convert to float
        
        # Ensure integer values for certain features
        if feature in ['credit_score', 'loan_term', 'loan_age']:
            value = int(round(value))
        
        sample[feature] = value
    
    return sample

def send_prediction_request(data):
    """Send prediction request to the service."""
    url = 'http://localhost:5000/predict'
    headers = {'Content-Type': 'application/json'}
    
    try:
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            result = response.json()
            print(f"Prediction: {result['prediction']}, Confidence: {result['confidence']:.2f}")
            if 'drift_metrics' in result and result['drift_metrics']:
                print("Drift Metrics:", json.dumps(result['drift_metrics'], indent=2))
        else:
            print(f"Error: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"Request failed: {str(e)}")

def main():
    print("Starting drift simulation...")
    n_samples = 200  # Total number of samples
    drift_start = 100  # When to start introducing drift
    
    for i in range(n_samples):
        phase = 'normal' if i < drift_start else 'drift'
        # Gradually increase drift magnitude
        drift_magnitude = 0 if phase == 'normal' else (i - drift_start) / (n_samples - drift_start)
        
        print(f"\nSample {i+1}/{n_samples} - Phase: {phase}, Drift Magnitude: {drift_magnitude:.2f}")
        sample = generate_sample(phase, drift_magnitude)
        send_prediction_request(sample)
        
        # Add some delay between requests
        time.sleep(1)

if __name__ == "__main__":
    main()
