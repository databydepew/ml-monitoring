from test_app import LoanModelTester
import time

def run_test_scenarios():
    tester = LoanModelTester()
    
    # Scenario 1: High likelihood of refinance
    print("\n=== Scenario 1: High Refinance Likelihood ===")
    high_refinance = {
        "interest_rate": 7.5,
        "loan_amount": 300000,
        "loan_balance": 280000,
        "loan_to_value_ratio": 0.75,
        "credit_score": 780,
        "debt_to_income_ratio": 0.3,
        "income": 150000,
        "loan_term": 360,
        "loan_age": 24,
        "home_value": 400000,
        "current_rate": 7.5,
        "rate_spread": 1.5  # Large positive spread (current rate much higher than offered)
    }
    print("\nApplication Details:")
    print(f"  Rate Spread: {high_refinance['rate_spread']:.2f}")
    print(f"  Credit Score: {high_refinance['credit_score']}")
    print(f"  DTI Ratio: {high_refinance['debt_to_income_ratio']:.2f}")
    prediction = tester.make_prediction(high_refinance)
    if prediction:
        feedback = tester.provide_feedback(prediction)
        time.sleep(1)

    # Scenario 2: Low likelihood of refinance
    print("\n=== Scenario 2: Low Refinance Likelihood ===")
    low_refinance = {
        "interest_rate": 4.5,
        "loan_amount": 200000,
        "loan_balance": 190000,
        "loan_to_value_ratio": 0.85,
        "credit_score": 650,
        "debt_to_income_ratio": 0.45,
        "income": 70000,
        "loan_term": 360,
        "loan_age": 36,
        "home_value": 250000,
        "current_rate": 4.5,
        "rate_spread": -0.5  # Negative spread (current rate lower than offered)
    }
    print("\nApplication Details:")
    print(f"  Rate Spread: {low_refinance['rate_spread']:.2f}")
    print(f"  Credit Score: {low_refinance['credit_score']}")
    print(f"  DTI Ratio: {low_refinance['debt_to_income_ratio']:.2f}")
    prediction = tester.make_prediction(low_refinance)
    if prediction:
        feedback = tester.provide_feedback(prediction)
        time.sleep(1)

    # Scenario 3: Borderline case
    print("\n=== Scenario 3: Borderline Case ===")
    borderline = {
        "interest_rate": 6.0,
        "loan_amount": 400000,
        "loan_balance": 380000,
        "loan_to_value_ratio": 0.8,
        "credit_score": 720,
        "debt_to_income_ratio": 0.38,
        "income": 120000,
        "loan_term": 360,
        "loan_age": 48,
        "home_value": 500000,
        "current_rate": 6.0,
        "rate_spread": 0.75  # Moderate spread
    }
    print("\nApplication Details:")
    print(f"  Rate Spread: {borderline['rate_spread']:.2f}")
    print(f"  Credit Score: {borderline['credit_score']}")
    print(f"  DTI Ratio: {borderline['debt_to_income_ratio']:.2f}")
    prediction = tester.make_prediction(borderline)
    if prediction:
        feedback = tester.provide_feedback(prediction)
        time.sleep(1)

    # Scenario 4: Good rate spread but poor qualification
    print("\n=== Scenario 4: Good Rate Spread but Poor Qualification ===")
    poor_qualification = {
        "interest_rate": 7.0,
        "loan_amount": 500000,
        "loan_balance": 490000,
        "loan_to_value_ratio": 0.9,
        "credit_score": 620,
        "debt_to_income_ratio": 0.55,
        "income": 85000,
        "loan_term": 360,
        "loan_age": 12,
        "home_value": 550000,
        "current_rate": 7.0,
        "rate_spread": 1.2  # Good spread but other factors are poor
    }
    print("\nApplication Details:")
    print(f"  Rate Spread: {poor_qualification['rate_spread']:.2f}")
    print(f"  Credit Score: {poor_qualification['credit_score']}")
    print(f"  DTI Ratio: {poor_qualification['debt_to_income_ratio']:.2f}")
    prediction = tester.make_prediction(poor_qualification)
    if prediction:
        feedback = tester.provide_feedback(prediction)

if __name__ == "__main__":
    run_test_scenarios()
