import requests
import json

import time

def send_prediction_requests(n=20, batch_size=5, delay=0.3):
    url = "http://localhost:5000/predict"
    headers = {"Content-Type": "application/json"}
    data = {
        "age": 30,
        "income": 50000,
        "loan_amount": 10000,
        "loan_term": 12,
        "credit_score": 700,
        "employment_status": "EMPLOYED",
        "loan_purpose": "PERSONAL"
    }
    
    responses = []
    for batch in range(0, n, batch_size):
        # Send batch_size requests quickly
        batch_responses = []
        for _ in range(min(batch_size, n - batch)):
            response = requests.post(url, headers=headers, data=json.dumps(data))
            if response.status_code == 200:
                batch_responses.append(response.json())
            else:
                batch_responses.append({"error": response.status_code, "message": response.text})
        
        responses.extend(batch_responses)
        print(f"Batch {batch//batch_size + 1}: Sent {len(batch_responses)} requests")
        
        # Wait before next batch to maintain a steady rate
        if batch + batch_size < n:
            time.sleep(delay)
    
    print(f"\nTotal requests sent: {len(responses)}")
    print("Sample responses:")
    for i, res in enumerate(responses[:3]):
        print(f"Response {i+1}:", res)

if __name__ == "__main__":
    print("Sending requests to trigger high prediction volume alert...")
    send_prediction_requests(20, batch_size=5, delay=0.1)
