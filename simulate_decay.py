import requests
import numpy as np
import pandas as pd
from datetime import datetime
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json

class ModelDecaySimulator:
    def __init__(self, base_url: str = "http://localhost:5001"):
        self.base_url = base_url
        self.predictions: List[Dict] = []
        self.accuracies: List[float] = []
        self.timestamps: List[datetime] = []
        
    def generate_loan_data(self, time_step: int) -> Dict:
        """Generate loan data with increasing market divergence over time."""
        # Base interest rate increases over time
        base_rate = 4.5 + (time_step * 0.2)  # Gradual increase
        
        # Generate loan data with explicit conversion to Python native types
        loan_data = {
            "interest_rate": float(round(base_rate, 2)),
            "loan_amount": int(np.random.randint(200000, 500000)),
            "loan_balance": 0,  # Will be calculated
            "loan_to_value_ratio": float(round(np.random.uniform(0.6, 0.9), 2)),
            "credit_score": int(np.random.randint(620, 850)),
            "debt_to_income_ratio": float(round(np.random.uniform(0.2, 0.5), 2)),
            "income": int(np.random.randint(80000, 200000)),
            "loan_term": int(np.random.choice([15, 30])),
            "loan_age": int(np.random.randint(1, 10)),
            "home_value": 0,  # Will be calculated
            "current_rate": 0,  # Will be calculated
            "rate_spread": 0,  # Will be calculated
        }
        
        # Calculate dependent values
        loan_data["home_value"] = int(loan_data["loan_amount"] / loan_data["loan_to_value_ratio"])
        loan_data["loan_balance"] = int(loan_data["loan_amount"] * 0.95)  # Assume some principal paid
        
        # Current rate increases more dramatically to simulate market divergence
        market_volatility = float(np.random.uniform(-0.5, 1.5) * (time_step / 10))
        loan_data["current_rate"] = float(round(base_rate + 2 + market_volatility, 2))
        loan_data["rate_spread"] = float(round(loan_data["current_rate"] - loan_data["interest_rate"], 2))
        
        return loan_data

    def synthetic_ground_truth(self, loan_data: Dict, time_step: int) -> bool:
        """
        Generate synthetic ground truth with increasing complexity over time.
        As time_step increases, the relationship between features and refinancing becomes more complex.
        """
        base_probability = 0.0
        
        # Basic factors (dominant in early time steps)
        if loan_data["rate_spread"] > 1.0:
            base_probability += 0.3
        if loan_data["credit_score"] > 720:
            base_probability += 0.2
        if loan_data["loan_to_value_ratio"] < 0.8:
            base_probability += 0.2
            
        # Complex factors (become more important over time)
        time_weight = min(time_step / 20, 1.0)  # Gradually increase importance
        
        complex_factors = (
            # Market sentiment (decreases refinance probability as rates rise)
            -0.3 * (loan_data["current_rate"] > 7.0) * time_weight +
            # Economic conditions (higher DTI reduces refinance probability)
            -0.2 * (loan_data["debt_to_income_ratio"] > 0.4) * time_weight +
            # Term preference (preference for shorter terms increases)
            0.2 * (loan_data["loan_term"] == 15) * time_weight
        )
        
        final_probability = base_probability + complex_factors
        return np.random.random() < final_probability

    def make_prediction(self, loan_data: Dict) -> Dict:
        """Make a prediction using the model API."""
        try:
            response = requests.post(
                f"{self.base_url}/predict",
                json=loan_data,
                headers={"Content-Type": "application/json"}
            )
            return response.json()
        except Exception as e:
            print(f"Error making prediction: {e}")
            return {"prediction": 0, "probability": 0.0}

    def simulate_decay(self, n_steps: int = 50, delay: float = 0.5) -> Tuple[pd.DataFrame, Dict]:
        """
        Simulate model decay over time by making predictions and comparing with synthetic ground truth.
        """
        results = []
        
        for step in range(n_steps):
            # Generate loan data with increasing market divergence
            loan_data = self.generate_loan_data(step)
            
            # Get model prediction
            prediction = self.make_prediction(loan_data)
            
            # Generate synthetic ground truth
            ground_truth = self.synthetic_ground_truth(loan_data, step)
            
            # Record results
            result = {
                "timestamp": datetime.now(),
                "step": step,
                "prediction": prediction["prediction"],
                "probability": prediction["probability"],
                "ground_truth": int(ground_truth),
                "rate_spread": loan_data["rate_spread"],
                "current_rate": loan_data["current_rate"],
                "credit_score": loan_data["credit_score"]
            }
            results.append(result)
            
            # Calculate running accuracy
            correct_predictions = sum(
                r["prediction"] == r["ground_truth"] 
                for r in results[-20:]  # Use rolling window of 20 predictions
            )
            accuracy = correct_predictions / min(20, len(results))
            
            self.accuracies.append(accuracy)
            self.timestamps.append(result["timestamp"])
            
            print(f"Step {step}: Accuracy = {accuracy:.2f}, Rate Spread = {loan_data['rate_spread']:.2f}")
            
            time.sleep(delay)
        
        # Convert results to DataFrame
        df_results = pd.DataFrame(results)
        
        # Calculate metrics
        metrics = {
            "final_accuracy": accuracy,
            "avg_probability": df_results["probability"].mean(),
            "avg_rate_spread": df_results["rate_spread"].mean(),
            "prediction_bias": (df_results["prediction"] - df_results["ground_truth"]).mean()
        }
        
        return df_results, metrics

    def plot_results(self, df_results: pd.DataFrame):
        """Plot the simulation results."""
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Accuracy over time
        plt.subplot(2, 2, 1)
        plt.plot(df_results["step"], 
                df_results["prediction"] == df_results["ground_truth"])
        plt.title("Prediction Accuracy Over Time")
        plt.xlabel("Step")
        plt.ylabel("Accurate (1) / Inaccurate (0)")
        
        # Plot 2: Rate Spread vs Accuracy
        plt.subplot(2, 2, 2)
        plt.scatter(df_results["rate_spread"], 
                   df_results["prediction"] == df_results["ground_truth"])
        plt.title("Rate Spread vs Accuracy")
        plt.xlabel("Rate Spread")
        plt.ylabel("Accurate (1) / Inaccurate (0)")
        
        # Plot 3: Running Accuracy
        plt.subplot(2, 2, 3)
        plt.plot(self.timestamps, self.accuracies)
        plt.title("Running Accuracy Over Time")
        plt.xlabel("Time")
        plt.ylabel("Accuracy (20-prediction window)")
        
        # Plot 4: Prediction Probability Distribution
        plt.subplot(2, 2, 4)
        plt.hist(df_results["probability"], bins=20)
        plt.title("Prediction Probability Distribution")
        plt.xlabel("Probability")
        plt.ylabel("Count")
        
        plt.tight_layout()
        plt.savefig('model_decay_simulation.png')
        plt.close()

def main():
    simulator = ModelDecaySimulator()
    print("Starting model decay simulation...")
    
    # Run simulation
    results, metrics = simulator.simulate_decay(n_steps=50, delay=0.5)
    
    # Plot results
    simulator.plot_results(results)
    
    # Save results
    results.to_csv('decay_simulation_results.csv', index=False)
    with open('decay_simulation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\nSimulation complete!")
    print("\nMetrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}")
    print("\nResults saved to:")
    print("- decay_simulation_results.csv")
    print("- decay_simulation_metrics.json")
    print("- model_decay_simulation.png")

if __name__ == "__main__":
    main()
