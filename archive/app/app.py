from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from prometheus_client.exposition import generate_latest
from collections import deque
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from app.helpers import (
    bigquery_reference,
    distribution_monitor
)

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Define Prometheus metrics
PREDICTION_COUNTER = Counter('refinance_model_predictions_total', 'Total number of refinance predictions made')
REFINANCE_APPROVED = Counter('refinance_model_approved_total', 'Number of refinance approvals')
PREDICTION_LATENCY = Histogram('refinance_model_prediction_latency_seconds', 'Time spent processing refinance prediction')

# Accuracy metrics
MODEL_ACCURACY = Gauge('refinance_model_accuracy', 'Current model accuracy based on feedback')
MODEL_ACCURACY_WINDOW = Gauge('refinance_model_accuracy_window', 'Model accuracy over recent predictions window')
CORRECT_PREDICTIONS = Counter('refinance_model_correct_predictions_total', 'Total number of correct refinance predictions')
INCORRECT_PREDICTIONS = Counter('refinance_model_incorrect_predictions_total', 'Total number of incorrect refinance predictions')

# Drift metrics
DATA_DRIFT_SCORE = Gauge('refinance_model_data_drift_score', 'Current data drift score')

# Feature-specific metrics
RATE_SPREAD_GAUGE = Gauge('refinance_model_rate_spread', 'Current rate spread (current rate - new rate)')
LTV_RATIO_GAUGE = Gauge('refinance_model_ltv_ratio', 'Current loan-to-value ratio')
CREDIT_SCORE_GAUGE = Gauge('refinance_model_credit_score', 'Current credit score')

# Add metrics endpoint
@app.route('/metrics')
def metrics():
    return generate_latest(), 200, {'Content-Type': 'text/plain; version=0.0.4'}

# BigQuery Configuration
BIGQUERY_CONFIG = {
    'project_id': 'mdepew-assets',
    'dataset_id': 'synthetic',
    'table_id': 'synthetic_mortgage_data',
    'target_column': 'refinance',
    'feature_columns': [
        'interest_rate',
        'loan_amount',
        'loan_balance',
        'loan_to_value_ratio',
        'credit_score',
        'debt_to_income_ratio',
        'income',
        'loan_term',
        'loan_age',
        'home_value',
        'current_rate',
        'rate_spread'
    ]
}

# Load the model and initialize BigQuery reference
model = joblib.load("model.pkl")
bq_reference = bigquery_reference.BigQueryReference(**BIGQUERY_CONFIG)

# Initialize prediction tracking
PREDICTION_WINDOW = 100  # Track last 100 predictions
prediction_history = deque(maxlen=PREDICTION_WINDOW)
total_correct = 0
total_predictions = 0
current_prediction_id = 0

# Initialize distribution monitors
feature_monitor = None
prediction_monitor = None

# Production data collection for monitoring
production_data = []
production_predictions = []

# Define feature names from BigQuery config
feature_names = BIGQUERY_CONFIG['feature_columns']

# Prometheus metrics
METRICS_PORT = int(os.environ.get("PROMETHEUS_METRICS_PORT", 8001))
model_requests = Counter('model_requests_total', 'Total number of prediction requests')
model_errors = Counter('model_errors_total', 'Total number of prediction errors')
model_success = Counter('model_success_total', 'Total number of successful predictions')
# Prometheus metrics
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
        PREDICTION_COUNTER.inc()
        global feature_monitor, prediction_monitor, current_prediction_id
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

        # Store data for distribution monitoring
        production_data.append(features)
        production_predictions.append(prediction)

        # Initialize monitors if needed
        if len(production_data) >= PREDICTION_WINDOW and feature_monitor is None:
            # Initialize monitors with BigQuery reference data
            feature_monitor, prediction_monitor = setup_distribution_monitoring(
                bigquery_config=BIGQUERY_CONFIG
            )
            logger.info("Distribution monitors initialized with BigQuery reference data")

        # Check for distribution drift
        if feature_monitor is not None and len(production_data) >= 100:
            # Create DataFrame for latest batch
            latest_data = pd.DataFrame([features], columns=feature_names)
            
            # Monitor feature distributions
            drift_results = feature_monitor.add_production_data(latest_data)
            
            # Log any significant drift
            for feature, stats in drift_results.items():
                if stats.is_significant:
                    logger.warning(f"Distribution drift detected in feature: {feature}")
                    logger.warning(f"KL divergence: {stats.kl_divergence:.4f}, p-value: {stats.p_value:.4f}")

            # Monitor prediction distribution
            pred_drift = prediction_monitor.add_production_data(
                pd.DataFrame({'predictions': [prediction]})
            )
            if any(stats.is_significant for stats in pred_drift.values()):
                logger.warning("Distribution drift detected in predictions")

        # Update feature-specific metrics
        RATE_SPREAD_GAUGE.set(float(data['rate_spread']))
        LTV_RATIO_GAUGE.set(float(data['loan_to_value_ratio']))
        CREDIT_SCORE_GAUGE.set(float(data['credit_score']))

        # Increment refinance approved counter if prediction is 1
        if prediction == 1:
            REFINANCE_APPROVED.inc()

        response_data = {
            "prediction_id": current_prediction_id,
            "refinance_recommended": bool(prediction),
            "features_received": dict(zip(feature_names, features)),
            "metrics": {
                "rate_spread": float(data['rate_spread']),
                "ltv_ratio": float(data['loan_to_value_ratio']),
                "credit_score": float(data['credit_score'])
            }
        }
        current_prediction_id += 1
        return jsonify(response_data)

        
    except Exception as e:
        model_errors.inc()
        return jsonify({"error": str(e)}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        global total_correct, total_predictions
        data = request.json
        if 'prediction_id' not in data or 'actual_outcome' not in data:
            return jsonify({"error": "Missing prediction_id or actual_outcome"}), 400

        prediction_id = data['prediction_id']
        actual_outcome = int(data['actual_outcome'])
        predicted_outcome = int(data.get('predicted_outcome', -1))

        if actual_outcome not in [0, 1]:
            return jsonify({"error": "actual_outcome must be 0 or 1"}), 400

        # Track if prediction was correct
        is_correct = actual_outcome == predicted_outcome
        prediction_history.append(is_correct)

        # Update counters
        if is_correct:
            CORRECT_PREDICTIONS.inc()
        else:
            INCORRECT_PREDICTIONS.inc()

        # Update total counts
        total_predictions += 1
        total_correct = sum(prediction_history)
        
        # Calculate and update accuracy metrics
        if total_predictions > 0:
            # Overall accuracy
            overall_accuracy = total_correct / total_predictions
            MODEL_ACCURACY.set(overall_accuracy)
            
            # Window accuracy (last N predictions)
            window_accuracy = sum(prediction_history) / len(prediction_history)
            MODEL_ACCURACY_WINDOW.set(window_accuracy)
            
            logger.info(f"Current model accuracy: {overall_accuracy:.2%} (window: {window_accuracy:.2%})")
            
            return jsonify({
                "success": True,
                "metrics": {
                    "overall_accuracy": overall_accuracy,
                    "window_accuracy": window_accuracy,
                    "total_predictions": total_predictions,
                    "correct_predictions": total_correct
                }
            })
        return jsonify({
            "success": True,
            "current_accuracy": current_accuracy
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Start Prometheus metrics server
    metrics_port = int(os.getenv('PROMETHEUS_METRICS_PORT', '8001'))
    start_http_server(metrics_port)
    logger.info(f"Started Prometheus metrics server on port {metrics_port}")
    
    # Setup distribution monitoring
    # Get reference data from BigQuery
    reference_data = bq_reference.get_reference_sample(1000)
    
    # Make predictions on reference data
    reference_predictions = model.predict(reference_data[BIGQUERY_CONFIG['feature_columns']])
    
    feature_monitor, prediction_monitor = distribution_monitor.setup_distribution_monitoring(
        reference_data=reference_data,
        reference_predictions=reference_predictions
    )
    
    # Start Flask app
    app.run(host='0.0.0.0', port=5000)
