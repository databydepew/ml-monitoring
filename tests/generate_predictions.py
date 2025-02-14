import requests
import random
import time
import json
import numpy as np
import pandas as pd

def generate_refinance_data(batch_num, total_batches):
    # Simulate market changes over time
    progress = batch_num / total_batches
    
    # Simulate rising interest rates scenario
    base_current_rate = 4.5 + (progress * 2.0)  # Rates increase by 2% over time
    current_rate = round(random.uniform(base_current_rate, base_current_rate + 1.0), 2)
    
    # New offered rate follows market trend but with smaller increase
    base_new_rate = 2.5 + (progress * 1.5)  # New rates increase slower
    new_rate = round(random.uniform(base_new_rate, base_new_rate + 1.0), 2)
    
    # Rate spread shrinks over time
    rate_spread = round(current_rate - new_rate, 2)
    
    # Simulate housing market changes
    # Home values appreciate over time
    appreciation_factor = 1.0 + (progress * 0.15)  # 15% appreciation over time
    loan_balance = random.randint(100000, 900000)
    home_value = int(random.randint(int(loan_balance * 1.1), int(loan_balance * 1.5)) * appreciation_factor)
    
    # LTV ratios trend higher as home values appreciate slower than inflation
    base_ltv = 0.6 + (progress * 0.1)  # LTV ratios trend up
    loan_to_value_ratio = round(min(0.95, random.uniform(base_ltv, base_ltv + 0.1)), 2)
    
    # Credit scores trend lower as economic conditions tighten
    base_credit_score = 750 - int(progress * 50)  # Average credit scores decline
    credit_score = min(850, max(620, int(np.random.normal(base_credit_score, 30))))
    
    # DTI ratios increase as interest rates rise
    base_dti = 0.25 + (progress * 0.1)  # DTI ratios trend up
    dti_ratio = round(min(0.43, random.uniform(base_dti, base_dti + 0.05)), 2)
    
    return {
        "interest_rate": new_rate,
        "loan_amount": loan_balance,  # Refinance amount typically equals current balance
        "loan_balance": loan_balance,
        "loan_to_value_ratio": loan_to_value_ratio,
        "credit_score": random.randint(620, 850),  # Higher credit scores more likely for refinance
        "debt_to_income_ratio": round(random.uniform(0.15, 0.43), 2),  # Standard DTI requirements
        "income": random.randint(60000, 250000),
        "loan_term": random.choice([180, 360]),  # 15 or 30 years in months
        "loan_age": random.randint(12, 84),  # 1-7 years, typical refinance window
        "home_value": home_value,
        "current_rate": current_rate,
        "rate_spread": rate_spread
    }

def send_prediction(data):
    try:
        response = requests.post('http://localhost:5000/predict', json=data)
        result = response.json()
        
        # Store prediction data for analysis
        prediction_data = data.copy()
        prediction_data['prediction'] = result.get('prediction')
        prediction_data['probability'] = result.get('probability')
        all_predictions.append(prediction_data)
        
        print(f"Request: Rate Spread = {data['rate_spread']}, LTV = {data['loan_to_value_ratio']}, Credit Score = {data['credit_score']}")
        print(f"Response: {result}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

# Generate reference data that represents historical patterns
def generate_reference_data(n_samples=1000):
    reference_data = []
    for _ in range(n_samples):
        # Generate data with historical patterns (pre-market changes)
        current_rate = round(random.uniform(4.0, 5.0), 2)  # Historical rates
        new_rate = round(random.uniform(2.0, 3.0), 2)     # Historical new rates
        rate_spread = round(current_rate - new_rate, 2)
        
        loan_balance = random.randint(100000, 900000)
        home_value = random.randint(int(loan_balance * 1.1), int(loan_balance * 1.5))
        ltv = round(loan_balance / home_value, 2)
        
        data = {
            'interest_rate': new_rate,
            'loan_amount': loan_balance,
            'loan_balance': loan_balance,
            'loan_to_value_ratio': ltv,
            'credit_score': random.randint(680, 850),  # Historical better credit scores
            'debt_to_income_ratio': round(random.uniform(0.15, 0.35), 2),  # Historical better DTI
            'income': random.randint(70000, 250000),
            'loan_term': random.choice([180, 360]),
            'loan_age': random.randint(12, 84),
            'home_value': home_value,
            'current_rate': current_rate,
            'rate_spread': rate_spread,
            'refinance': 1 if random.random() < 0.7 else 0  # Historical higher approval rate
        }
        reference_data.append(data)
    return pd.DataFrame(reference_data)

# Generate reference data
reference_data = generate_reference_data(1000)
print("\nReference Data Statistics (Historical Patterns):")
for col in reference_data.columns:
    if col != 'refinance':
        stats = reference_data[col].describe()
        print(f"\n{col}:")
        print(f"  Mean: {stats['mean']:.2f}")
        print(f"  Std: {stats['std']:.2f}")
        print(f"  Min: {stats['min']:.2f}")
        print(f"  Max: {stats['max']:.2f}")

# Generate predictions in batches to simulate time periods
num_batches = 5
predictions_per_batch = 20
total_successful = 0

# Store predictions for comparison
all_predictions = []

print(f"Sending {num_batches} batches of {predictions_per_batch} refinance predictions...")
for batch in range(num_batches):
    print(f"\nBatch {batch + 1}/{num_batches} - Simulating market conditions at time period {batch + 1}")
    successful = 0
    
    for i in range(predictions_per_batch):
        data = generate_refinance_data(batch, num_batches)
        if send_prediction(data):
            successful += 1
        time.sleep(1)  # Longer delay to see drift patterns
    
    total_successful += successful
    print(f"Batch {batch + 1} complete: {successful}/{predictions_per_batch} successful")
    if batch < num_batches - 1:
        print("Waiting 5 seconds before next batch...")
        time.sleep(5)  # Delay between batches

print(f"\nCompleted {num_batches} batches with {total_successful}/{num_batches * predictions_per_batch} successful predictions")

# Convert predictions to DataFrame for analysis
predictions_df = pd.DataFrame(all_predictions)

print("\nPrediction Statistics vs Reference:")
for col in predictions_df.columns:
    if col not in ['prediction', 'probability'] and col in reference_data.columns:
        pred_stats = predictions_df[col].describe()
        ref_stats = reference_data[col].describe()
        print(f"\n{col}:")
        print(f"  Prediction Mean: {pred_stats['mean']:.2f} (Ref: {ref_stats['mean']:.2f})")
        print(f"  Prediction Std: {pred_stats['std']:.2f} (Ref: {ref_stats['std']:.2f})")
        print(f"  Prediction Range: [{pred_stats['min']:.2f}, {pred_stats['max']:.2f}]")
        print(f"  Reference Range: [{ref_stats['min']:.2f}, {ref_stats['max']:.2f}]")

# Calculate approval rates
pred_approval_rate = (predictions_df['prediction'] == 1).mean()
ref_approval_rate = (reference_data['refinance'] == 1).mean()
print(f"\nApproval Rates:")
print(f"  Predictions: {pred_approval_rate:.2%}")
print(f"  Reference: {ref_approval_rate:.2%}")
