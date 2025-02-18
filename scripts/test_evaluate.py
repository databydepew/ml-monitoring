#!/usr/bin/env python3
import requests
import json
from datetime import datetime
import time

def test_evaluate(hours_back=1, min_samples=10):
    """Test the model evaluation endpoint with different parameters."""
    url = f"http://localhost:5000/evaluate?hours_back={hours_back}&min_samples={min_samples}"
    
    print(f"\n=== Testing evaluation endpoint ===")
    print(f"Parameters: hours_back={hours_back}, min_samples={min_samples}")
    print(f"URL: {url}\n")
    
    try:
        response = requests.get(url)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("\nEvaluation Results:")
            print("==================")
            
            # Print metadata
            metadata = data.get('metadata', {})
            print(f"Evaluation Time: {metadata.get('evaluation_time')}")
            print(f"Hours Evaluated: {metadata.get('hours_evaluated')}")
            print(f"Model Version: {metadata.get('model_version')}")
            
            # Print performance metrics
            report = data.get('evaluation_report', {})
            perf_metrics = report.get('performance_metrics', {})
            if perf_metrics:
                print("\nPerformance Metrics:")
                print("-----------------")
                class_report = perf_metrics.get('classification_report', {})
                print(f"Accuracy: {class_report.get('accuracy', 'N/A'):.3f}")
                if '1' in class_report:
                    print(f"Precision: {class_report['1'].get('precision', 'N/A'):.3f}")
                    print(f"Recall: {class_report['1'].get('recall', 'N/A'):.3f}")
                    print(f"F1-Score: {class_report['1'].get('f1-score', 'N/A'):.3f}")
            
            # Print sample sizes
            samples = report.get('sample_sizes', {})
            if samples:
                print("\nSample Sizes:")
                print("-------------")
                print(f"Ground Truth: {samples.get('ground_truth', 'N/A')}")
                print(f"Predictions: {samples.get('predictions', 'N/A')}")
            
            # Print drift analysis
            drift = report.get('drift_analysis', {})
            if drift:
                print("\nDrift Analysis:")
                print("--------------")
                for feature, metrics in drift.items():
                    print(f"\n{feature}:")
                    print(f"  KS Statistic: {metrics.get('ks_statistic', 'N/A'):.3f}")
                    print(f"  P-Value: {metrics.get('p_value', 'N/A'):.3e}")
        else:
            print("Error Response:")
            print(json.dumps(response.json(), indent=2))
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the model service.")
        print("Make sure the service is running on http://localhost:5000")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

def main():
    print("Starting evaluation tests...")
    
    # Test 1: Default parameters
    test_evaluate()
    
    # Test 2: Longer time window
    time.sleep(1)
    test_evaluate(hours_back=24)
    
    # Test 3: Higher minimum samples
    time.sleep(1)
    test_evaluate(min_samples=50)
    
    # Test 4: Short window with high samples (should fail)
    time.sleep(1)
    test_evaluate(hours_back=1, min_samples=1000)

if __name__ == "__main__":
    main()
