from flask import Flask, request, jsonify
import joblib
import numpy as np
from prometheus_client import start_http_server, Counter, Histogram, Gauge
import os

app = Flask(__name__)

# Load the model
model = joblib.load("model.pkl")

# Define feature names
feature_names = ["age", "income", "loan_amount", "loan_term", "credit_score", "employment_status", "loan_purpose"]

# Prometheus metrics
METRICS_PORT = int(os.environ.get("PROMETHEUS_METRICS_PORT", 8001))
model_requests = Counter('model_requests_total', 'Total number of prediction requests')
model_errors = Counter('model_errors_total', 'Total number of prediction errors')
model_success = Counter('model_success_total', 'Total number of successful predictions')
prediction_latency = Histogram('model_prediction_latency_seconds', 'Time spent processing prediction request')
prediction_values = Histogram('model_prediction_values', 'Distribution of model predictions')
feature_values = {
    feature: Gauge(f'model_feature_{feature}', f'Current value of {feature}')
    for feature in feature_names
}
predictions_by_class = Counter('model_predictions_by_class_total', 'Total predictions by class',
                             labelnames=['prediction'])
model_accuracy = Gauge('model_accuracy', 'Current accuracy of the model')



@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        model_requests.inc()
        data = request.json

        # Validate input features
        if not all(feature in data for feature in feature_names):
            missing_features = [f for f in feature_names if f not in data]
            model_errors.inc()
            return jsonify({
                "error": f"Missing features: {missing_features}",
                "required_features": feature_names
            }), 400

        # Extract features in correct order
        features = [float(data[feature]) for feature in feature_names]
        
        # Update feature value metrics
        for feature, value in zip(feature_names, features):
            feature_values[feature].set(value)

        # Make prediction with latency tracking
        with prediction_latency.time():
            prediction = model.predict([features])[0]

        # Update prediction metrics
        prediction_values.observe(prediction)
        predictions_by_class.labels(prediction=str(prediction)).inc()
        model_success.inc()

        return jsonify({
            "approval_status": int(prediction),
            "features_received": dict(zip(feature_names, features))
        })

        
    except Exception as e:
        model_errors.inc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Start Prometheus metrics server
    start_http_server(METRICS_PORT)
    # Start Flask app
    app.run(host='0.0.0.0', port=5000)
