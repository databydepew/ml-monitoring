import requests
import json
import time
import concurrent.futures

def send_prediction_request():
    url = "http://localhost:5000/predict"
    data = {
        "age": 35,
        "income": 75000,
        "loan_amount": 25000,
        "loan_term": 36,
        "credit_score": 720,
        "employment_status": 1,
        "loan_purpose": 1
    }
    try:
        response = requests.post(url, json=data)
        return response.status_code == 200
    except Exception as e:
        print(f"Error sending request: {e}")
        return False

def trigger_alert():
    print("Sending 8 requests in quick succession...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(send_prediction_request) for _ in range(8)]
        results = [f.result() for f in futures]
    
    successful = sum(results)
    print(f"Successfully sent {successful} out of 8 requests")
    print("Check http://localhost:9090/alerts to see if the TestQuickAlert was triggered")

if __name__ == "__main__":
    print("Starting alert test...")
    trigger_alert()
