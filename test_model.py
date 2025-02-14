#!/usr/bin/env python3
import requests
import json
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
import argparse

# Model endpoint
MODEL_ENDPOINT = "http://34.70.89.220/predict"

class DriftSimulator:
    def __init__(self):
        self.drift_step = 0
        self.max_steps = 20
        
    def generate_drifted_data(self):
        """Generate sample data with drift"""
        # Calculate drift factor (0 to 1)
        drift_factor = min(self.drift_step / self.max_steps, 1.0)
        self.drift_step += 1
        
        # Base distributions
        base_data = {
            "interest_rate": np.random.uniform(2.5, 7.0),
            "loan_amount": np.random.uniform(100000, 1000000),
            "loan_balance": np.random.uniform(80000, 900000),
            "loan_to_value_ratio": np.random.uniform(0.5, 0.95),
            "credit_score": np.random.uniform(600, 850),
            "debt_to_income_ratio": np.random.uniform(0.2, 0.5),
            "income": np.random.uniform(50000, 300000),
            "loan_term": np.random.choice([15, 30]),
            "loan_age": np.random.uniform(1, 120),
            "home_value": np.random.uniform(150000, 1500000),
            "current_rate": np.random.uniform(2.5, 8.0),
            "rate_spread": np.random.uniform(-2.0, 2.0)
        }
        
        # Drift distributions (simulating market changes)
        drift_data = {
            "interest_rate": np.random.uniform(1.5, 4.0),  # Lower interest rates
            "loan_amount": np.random.uniform(200000, 1500000),  # Higher loan amounts
            "loan_balance": np.random.uniform(150000, 1200000),
            "loan_to_value_ratio": np.random.uniform(0.6, 0.85),  # More conservative LTV
            "credit_score": np.random.uniform(680, 900),  # Better credit scores
            "debt_to_income_ratio": np.random.uniform(0.15, 0.4),  # Lower DTI
            "income": np.random.uniform(80000, 400000),  # Higher incomes
            "loan_term": np.random.choice([15, 30]),  # No drift in loan term
            "loan_age": np.random.uniform(1, 90),  # Newer loans
            "home_value": np.random.uniform(250000, 2000000),  # Higher home values
            "current_rate": np.random.uniform(1.5, 5.0),  # Lower current rates
            "rate_spread": np.random.uniform(-1.0, 1.0)  # Smaller spreads
        }
        
        # Interpolate between base and drift distributions
        data = {}
        for key in base_data:
            if key == "loan_term":
                data[key] = base_data[key]  # No drift for categorical
            else:
                data[key] = (1 - drift_factor) * base_data[key] + drift_factor * drift_data[key]
                # Add some noise
                noise = np.random.normal(0, 0.02 * abs(drift_data[key] - base_data[key]))
                data[key] += noise
        
        # Convert numpy types to native Python types
        return {k: float(v) if isinstance(v, (np.float64, np.float32)) else int(v) if isinstance(v, np.integer) else v 
                for k, v in data.items()}

# Global drift simulator instance
drift_simulator = DriftSimulator()

def generate_sample_data():
    return drift_simulator.generate_drifted_data()

def send_prediction_request(data):
    try:
        response = requests.post(MODEL_ENDPOINT, json=data)
        response.raise_for_status()
        result = response.json()
        return {
            "status": "success",
            "prediction": result["prediction"],
            "probability": result["probability"],
            "data": data
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "data": data
        }

def main():
    parser = argparse.ArgumentParser(description='Test the refinance prediction model with drift simulation')
    parser.add_argument('--batch-size', type=int, default=50, help='Number of requests per batch')
    parser.add_argument('--concurrent', type=int, default=5, help='Number of concurrent requests')
    parser.add_argument('--interval', type=int, default=10, help='Seconds between batches')
    args = parser.parse_args()

    print("Starting drift simulation...")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            print(f"\nDrift step {drift_simulator.drift_step}/{drift_simulator.max_steps}")
            
            # Generate test data
            test_data = [generate_sample_data() for _ in range(args.batch_size)]
            
            # Track batch metrics
            start_time = time.time()
            success_count = 0
            error_count = 0
            refinance_count = 0
            total_probability = 0
            
            # Send requests concurrently
            with ThreadPoolExecutor(max_workers=args.concurrent) as executor:
                results = list(executor.map(send_prediction_request, test_data))
            
            # Process results
            for result in results:
                if result["status"] == "success":
                    success_count += 1
                    total_probability += result["probability"]
                    if result["prediction"] == 1:
                        refinance_count += 1
                else:
                    error_count += 1
            
            # Print batch summary
            total_time = time.time() - start_time
            avg_probability = total_probability / success_count if success_count > 0 else 0
            print(f"Batch Summary:")
            print(f"Successful requests: {success_count}/{args.batch_size}")
            print(f"Refinance predictions: {refinance_count} ({refinance_count/args.batch_size*100:.1f}%)")
            print(f"Average probability: {avg_probability:.3f}")
            print(f"Requests per second: {args.batch_size/total_time:.1f}")
            
            # Reset drift simulation if needed
            if drift_simulator.drift_step >= drift_simulator.max_steps:
                print("\nResetting drift simulation...")
                drift_simulator.drift_step = 0
            
            # Wait before next batch
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print("\nStopping drift simulation...")

if __name__ == "__main__":
    main()
