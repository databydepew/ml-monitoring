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
        """Generate a random loan application."""
        return {
            "age": random.randint(21, 65),
            "income": random.randint(30000, 150000),
            "loan_amount": random.randint(5000, 50000),
            "loan_term": random.choice([12, 24, 36, 48, 60]),
            "credit_score": random.randint(580, 850),
            "employment_status": random.randint(0, 1),  # 0: unemployed, 1: employed
            "loan_purpose": random.randint(0, 2)  # 0: personal, 1: business, 2: education
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
        
        # Calculate probability of actual approval based on features
        prob_approval = (
            0.4 * (credit_score - 580) / (850 - 580) +  # Credit score weight
            0.4 * (income - 30000) / (150000 - 30000) +  # Income weight
            0.2 * (1 - loan_amount / 50000)  # Loan amount weight (inverse)
        )
        
        actual_outcome = 1 if random.random() < prob_approval else 0
        
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
                print(f"Prediction: {'Approved' if prediction['approval_status'] == 1 else 'Denied'}")
                
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
