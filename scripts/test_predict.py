import requests
import json

def send_test_prediction():
    """Send a test prediction request."""
    data = {
        'interest_rate': 3.5,
        'loan_amount': 250000,
        'loan_balance': 245000,
        'loan_to_value_ratio': 0.8, 
        'credit_score': 720,
        'debt_to_income_ratio': 0.28,
        'income': 120000,
        'loan_term': 30,
        'loan_age': 2,
        'home_value': 350000,
        'current_rate': 3.75,
        'rate_spread': 0.25,
        'loan_purpose': 'REFINANCE'
    }
    try:
        response = requests.post("http://localhost:5000/predict", json=data)
        print(f"\nPrediction Response (Status {response.status_code}):")
        print(response.json() if response.ok else response.text)
    except Exception as e:
        print(f"Error sending prediction: {str(e)}")

if __name__ == "__main__":
    send_test_prediction()