import requests
import random
import time
from typing import Dict, Any
import numpy as np

class LoanModelTester:
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.predictions = []  # Store predictions for feedback

    def generate_loan_application(self) -> Dict[str, Any]:
        """Generate a random loan application with features matching BigQuery schema."""
        return {
            "interest_rate": round(random.uniform(2.0, 8.0), 2),
            "loan_amount": random.randint(50000, 1000000),
            "loan_balance": random.randint(40000, 900000),
            "loan_to_value_ratio": round(random.uniform(0.5, 0.95), 2),
            "credit_score": random.randint(580, 850),
            "debt_to_income_ratio": round(random.uniform(0.2, 0.6), 2),
            "income": random.randint(50000, 300000),
            "loan_term": random.choice([180, 360]),  # 15 or 30 years in months
            "loan_age": random.randint(0, 120),  # months
            "home_value": random.randint(100000, 2000000),
            "current_rate": round(random.uniform(2.5, 8.5), 2),
            "rate_spread": round(random.uniform(-2.0, 2.0), 2)
        }

    def make_prediction(self, application: Dict[str, Any]) -> Dict[str, Any]:
        """Make a prediction request to the model."""
        try:
            response = requests.post(f"{self.base_url}/predict", json=application)
            response.raise_for_status()
            prediction = response.json()
            self.predictions.append(prediction)
            return prediction
        except requests.exceptions.RequestException as e:
            print(f"Error making prediction: {e}")
            return None

    def provide_feedback(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Provide feedback based on prediction."""
        # Simulate feedback based on a simple rule
        # Higher credit scores and income are more likely to be approved
        application = prediction["features_received"]
        credit_score = application["credit_score"]
        income = application["income"]
        loan_amount = application["loan_amount"]
        
        # Calculate probability of actual refinance based on features
        prob_refinance = (
            0.3 * (credit_score - 580) / (850 - 580) +  # Credit score weight
            0.2 * (income - 50000) / (250000) +  # Income weight
            0.2 * (1 - debt_to_income_ratio / 0.6) +  # DTI weight (inverse)
            0.3 * (rate_spread / 2.0)  # Rate spread weight (higher spread = higher refinance probability)
        )
        
        actual_outcome = 1 if random.random() < prob_refinance else 0
        
        try:
            response = requests.post(
                f"{self.base_url}/feedback",
                json={
                    "prediction_id": prediction["prediction_id"],
                    "actual_outcome": actual_outcome,
                    "predicted_outcome": prediction["approval_status"]
                }
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error providing feedback: {e}")
            return None

    def run_test_batch(self, num_applications: int = 10, delay: float = 0.5):
        """Run a batch of test applications with feedback."""
        print(f"Running {num_applications} test applications...")
        
        for i in range(num_applications):
            # Generate and submit application
            application = self.generate_loan_application()
            print(f"\nApplication {i+1}/{num_applications}:")
            print(f"Application details: {application}")
            
            # Get prediction
            prediction = self.make_prediction(application)
            if prediction:
                print(f"Prediction: {'Refinance' if prediction['refinance_status'] == 1 else 'No Refinance'}")
                
                # Provide feedback
                feedback = self.provide_feedback(prediction)
                if feedback:
                    print(f"Feedback provided - Current accuracy: {feedback.get('current_accuracy', 'N/A')}")
            
            # Add delay between requests
            time.sleep(delay)

def main():
    tester = LoanModelTester()
    
    # Test different scenarios
    print("Starting loan model testing...")
    
    # Run a batch of random applications
    tester.run_test_batch(num_applications=20, delay=1.0)
    
    print("\nTesting completed!")

if __name__ == "__main__":
    main()
