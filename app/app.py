from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from prometheus_client.exposition import generate_latest
import joblib
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)


metrics_port = int(os.getenv('PROMETHEUS_METRICS_PORT', '8001'))
start_http_server(metrics_port)
logger.info(f"Started Prometheus metrics server on port {metrics_port}")

# Model configuration
MODEL_CONFIG = {
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
    ],
    'model_path': os.getenv('MODEL_PATH', 'model.pkl'),
    'prediction_threshold': float(os.getenv('PREDICTION_THRESHOLD', '0.5')),
    'monitoring_window': int(os.getenv('MONITORING_WINDOW', '100'))
}

# Load the model
model = joblib.load(MODEL_CONFIG['model_path'])
logger.info(f"Model loaded from {MODEL_CONFIG['model_path']}")

# Example function for predictions
def predict(input_data):
    if model is None:
        return "Model not loaded"

    # Ensure input_data is a DataFrame with correct feature names
    try:
        feature_names = model.feature_names_in_  # Get feature names from trained model
        input_df = pd.DataFrame([input_data], columns=feature_names)  # Enforce column names
    except AttributeError:
        logger.warning("Model does not have feature names; using raw input")
        input_df = pd.DataFrame([input_data])  # Fallback


# Initialize metrics
class ModelMetrics:
    def __init__(self):
        # Request metrics
        self.prediction_requests = Counter(
            'model_requests_total', 
            'Total number of prediction requests'
        )
        self.prediction_latency = Histogram(
            'model_prediction_latency_seconds', 
            'Time spent processing prediction',
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0)
        )
        
        # Classification metrics
        self.predictions_by_class = Counter(
            'model_predictions_by_class_total', 
            'Predictions by class',
            labelnames=['prediction_class']
        )
        self.prediction_confidence = Histogram(
            'model_prediction_confidence', 
            'Prediction confidence scores',
            buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
        )
        
        # Performance metrics
        self.true_positives = Counter('model_true_positives_total', 'True positives count')
        self.true_negatives = Counter('model_true_negatives_total', 'True negatives count')
        self.false_positives = Counter('model_false_positives_total', 'False positives count')
        self.false_negatives = Counter('model_false_negatives_total', 'False negatives count')
        
        self.accuracy = Gauge('model_accuracy', 'Model accuracy')
        self.precision = Gauge('model_precision', 'Model precision')
        self.recall = Gauge('model_recall', 'Model recall')
        self.f1_score = Gauge('model_f1_score', 'Model F1 score')
        
        # Feature metrics
        self.feature_values = {
            feature: Histogram(
                f'model_feature_{feature}_distribution',
                f'Distribution of {feature} values',
                buckets=tuple(np.linspace(0, 1, 11))
            )
            for feature in MODEL_CONFIG['feature_columns']
        }

metrics = ModelMetrics()

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})

@app.route('/metrics')
def get_metrics():
    """Prometheus metrics endpoint."""
    return generate_latest()

@app.route('/predict', methods=['POST'])
def predict():
    """Make a prediction and record metrics."""
    start_time = datetime.now()
    metrics.prediction_requests.inc()
    
    try:
        # Validate request
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
            
        data = request.json
        
        # Validate input features
        missing_features = [
            f for f in MODEL_CONFIG['feature_columns'] 
            if f not in data
        ]
        if missing_features:
            return jsonify({
                "error": f"Missing features: {missing_features}",
                "required_features": MODEL_CONFIG['feature_columns']
            }), 400
        
        # Extract features in correct order
        features = [
            float(data[f]) 
            for f in MODEL_CONFIG['feature_columns']
        ]
        
        # Record feature distributions
        for feature, value in zip(MODEL_CONFIG['feature_columns'], features):
            metrics.feature_values[feature].observe(value)
        
        # Make prediction with latency tracking
        with metrics.prediction_latency.time():
            probabilities = model.predict_proba([features])[0]
            confidence = probabilities[1]
            prediction = int(confidence > MODEL_CONFIG['prediction_threshold'])
        
        # Record prediction metrics
        metrics.predictions_by_class.labels(prediction_class=str(prediction)).inc()
        metrics.prediction_confidence.observe(confidence)
        
        response = {
            "prediction": prediction,
            "confidence": float(confidence),
            "timestamp": start_time.isoformat(),
            "features_received": dict(zip(MODEL_CONFIG['feature_columns'], features))
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    """Record actual outcomes and update performance metrics."""
    try:
        data = request.json
        if not all(k in data for k in ['prediction', 'actual']):
            return jsonify({"error": "Missing required fields"}), 400
            
        prediction = int(data['prediction'])
        actual = int(data['actual'])
        
        if prediction not in (0, 1) or actual not in (0, 1):
            return jsonify({"error": "Values must be 0 or 1"}), 400
        
        # Update confusion matrix metrics
        if actual == 1 and prediction == 1:
            metrics.true_positives.inc()
        elif actual == 0 and prediction == 0:
            metrics.true_negatives.inc()
        elif actual == 0 and prediction == 1:
            metrics.false_positives.inc()
        else:  # actual == 1 and prediction == 0
            metrics.false_negatives.inc()
        
        # Calculate and update performance metrics
        tp = metrics.true_positives._value.get()
        tn = metrics.true_negatives._value.get()
        fp = metrics.false_positives._value.get()
        fn = metrics.false_negatives._value.get()
        
        total = tp + tn + fp + fn
        if total > 0:
            # Accuracy
            accuracy = (tp + tn) / total
            metrics.accuracy.set(accuracy)
            
            # Precision
            if tp + fp > 0:
                precision = tp / (tp + fp)
                metrics.precision.set(precision)
            
            # Recall
            if tp + fn > 0:
                recall = tp / (tp + fn)
                metrics.recall.set(recall)
            
            # F1 Score
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
                metrics.f1_score.set(f1)
        
        return jsonify({
            "success": True,
            "metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }
        })
        
    except Exception as e:
        logger.error(f"Feedback error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Start Prometheus metrics server
    try:
        start_http_server(8001)
        logger.info("Started Prometheus metrics server on port 8001")
    except Exception as e:
        logger.error(f"Failed to start Prometheus server: {str(e)}")
        
    # Start Flask app
    app.run(host='0.0.0.0', port=5000)
    logger.info("Started Flask application on port 5000")