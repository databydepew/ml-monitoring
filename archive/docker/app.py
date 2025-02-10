from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Histogram, Counter

app = Flask(__name__)
metrics = PrometheusMetrics(app)

# Define Prometheus metrics
PREDICTION_LATENCY = Histogram(
    'model_prediction_latency_seconds',
    'Time spent processing prediction',
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0]
)

PREDICTION_VALUES = Histogram(
    'model_prediction_values',
    'Distribution of prediction values',
    buckets=[0.0, 0.25, 0.5, 0.75, 1.0]
)

REQUEST_COUNTER = Counter(
    'model_requests_total',
    'Total number of requests made'
)

FEATURE_DISTRIBUTIONS = {
    'age': Histogram(
        'model_feature_age',
        'Distribution of age feature',
        buckets=[20, 30, 40, 50, 60, 70]
    ),
    'income': Histogram(
        'model_feature_income',
        'Distribution of income feature',
        buckets=[20000, 40000, 60000, 80000, 100000]
    ),
    'loan_amount': Histogram(
        'model_feature_loan_amount',
        'Distribution of loan amount feature',
        buckets=[5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000]
    ),
    'credit_score': Histogram(
        'model_feature_credit_score',
        'Distribution of credit score feature',
        buckets=[300, 400, 500, 600, 700, 800]
    ),
}

#


# Load the model
try:
    model = joblib.load('model.pkl')
except:
    print("Warning: model.pkl not found. Please train the model first.")
    model = None

@app.route('/health')
def health():
    if model is not None:
        return jsonify({"status": "healthy", "model_loaded": True})
    return jsonify({"status": "unhealthy", "model_loaded": False}), 503

@app.route('/predict', methods=['POST'])
def predict():
    REQUEST_COUNTER.inc()
    
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    try:
        data = request.get_json()
        
        # Validate required features
        required_features = [
            "age", "income", "loan_amount", "loan_term",
            "credit_score", "employment_status", "loan_purpose"
        ]
        
        if not all(feature in data for feature in required_features):
            missing_features = [f for f in required_features if f not in data]
            return jsonify({
                "error": f"Missing required features: {missing_features}"
            }), 400

        # Create DataFrame with single row
        input_data = pd.DataFrame([data])
        
        # Convert categorical variables
        employment_status_map = {
            'EMPLOYED': 1,
            'SELF_EMPLOYED': 2,
            'UNEMPLOYED': 0
        }
        
        loan_purpose_map = {
            'PERSONAL': 0,
            'BUSINESS': 1,
            'EDUCATION': 2,
            'HOME': 3,
            'CAR': 4
        }
        
        try:
            input_data['employment_status'] = input_data['employment_status'].map(employment_status_map)
            input_data['loan_purpose'] = input_data['loan_purpose'].map(loan_purpose_map)
        except KeyError as e:
            return jsonify({"error": f"Invalid value for {str(e)}. Valid values for employment_status: {list(employment_status_map.keys())}, loan_purpose: {list(loan_purpose_map.keys())}"}), 400
            
        # Record feature distributions
        for feature in ['age', 'income', 'loan_amount', 'credit_score']:
            FEATURE_DISTRIBUTIONS[feature].observe(float(input_data[feature].iloc[0]))

        # Make prediction with latency tracking
        with PREDICTION_LATENCY.time():
            prediction = model.predict_proba(input_data[required_features])[0]
            
        # Record prediction value
        PREDICTION_VALUES.observe(prediction[1])  # Probability of approval

        response = {
            "prediction": int(prediction[1] > 0.5),  # 1 if approved, 0 if rejected
            "approval_probability": float(prediction[1])
        }
        
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
