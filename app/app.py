from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from scipy import stats
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from prometheus_client.exposition import generate_latest
from google.cloud import bigquery
from model_evaluator import ModelEvaluator
import joblib
import json
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
    'monitoring_window': int(os.getenv('MONITORING_WINDOW', '100')),

}
project_id = "mdepew-assets"

# Initialize BigQuery client and model evaluator
try:
    bq_client = bigquery.Client(project='mdepew-assets')
    # Test the connection
    bq_client.get_dataset('synthetic')
    logger.info("Successfully connected to BigQuery")
except Exception as e:
    logger.error(f"Failed to initialize BigQuery client: {str(e)}")
    raise

evaluator = ModelEvaluator(
    project_id="mdepew-assets",
    dataset_id="synthetic",
    table_id="model_predictions"
)

def store_prediction_in_bigquery(features, prediction, confidence, drift_metrics, actual=None):
    """Store prediction details in BigQuery for later comparison.
    
    Args:
        features: Dict of feature values
        prediction: Model prediction (0 or 1)
        confidence: Prediction confidence
        drift_metrics: Dict of drift metrics
    """
    try:
        # Format the row data
        row = {
            'timestamp': datetime.now().isoformat(),
            'prediction': int(prediction),  # Ensure integer
            'actual': int(actual),  # Ensure integer
            'confidence': float(confidence),  # Ensure float
            **{k: float(v) if isinstance(v, (int, float)) else v for k, v in features.items()},  # Convert numeric types
            'drift_metrics': json.dumps(drift_metrics)
        }
        logger.info(f"Prepared row for BigQuery: {row}")
        
        # Construct table ID
        table_id = f"{project_id}.synthetic.model_predictions"
        
        # Attempt to insert row
        errors = bq_client.insert_rows_json(
            table_id,
            [row],
        )
        
        if errors:
            logger.error(f"BigQuery insert errors: {errors}")
            raise Exception(f"Failed to insert rows: {errors}")
        else:
            logger.info("Successfully wrote prediction to BigQuery")
            
    except Exception as e:
        logger.error(f"Error writing to BigQuery: {str(e)}")
        logger.error(f"Row data: {row}")
        logger.error(f"Table ID: {table_id}")
        raise
    
    if errors:
        logger.error(f"Failed to insert prediction into BigQuery: {errors}")


metrics_port = int(os.getenv('PROMETHEUS_METRICS_PORT', '8001'))
start_http_server(metrics_port)
logger.info(f"Started Prometheus metrics server on port {metrics_port}")


# Load the model
try:
    model = joblib.load(MODEL_CONFIG['model_path'])
except FileNotFoundError:
    logger.warning("Model file not found. Some endpoints may not work.")
    model = None
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


# Feature distribution tracking
@dataclass
class FeatureDistribution:
    values: List[float] = None
    bins: np.ndarray = None
    hist: np.ndarray = None
    
    def update(self, value: float) -> None:
        if self.values is None:
            self.values = []
        self.values.append(value)
        if len(self.values) > MODEL_CONFIG['monitoring_window']:
            self.values.pop(0)
    
    def compute_distribution(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.values:
            return np.array([]), np.array([])
        hist, bins = np.histogram(self.values, bins=10, density=True)
        self.bins = bins
        self.hist = hist
        return hist, bins

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
        
        # Distribution tracking
        self.feature_distributions = {
            feature: FeatureDistribution()
            for feature in MODEL_CONFIG['feature_columns']
        }
        
        # KL divergence metrics
        self.feature_kl_divergence = {
            feature: Gauge(
                f'model_feature_{feature}_kl_divergence',
                f'KL divergence for {feature} distribution'
            )
            for feature in MODEL_CONFIG['feature_columns']
        }
        
        # Drift monitoring metrics
        self.drift_detection_window = Gauge(
            'model_drift_detection_window',
            'Current size of drift detection window'
        )
        
        self.drift_alerts = Counter(
            'model_drift_alerts_total',
            'Number of drift alerts triggered',
            labelnames=['feature', 'severity']
        )
        
        self.max_kl_divergence = Gauge(
            'model_max_kl_divergence',
            'Maximum KL divergence across all features',
            labelnames=['feature']
        )
        
        self.drift_threshold_breaches = Counter(
            'model_drift_threshold_breaches_total',
            'Number of times drift threshold was breached',
            labelnames=['feature']
        )
        
        # KS test metrics
        self.feature_ks_statistic = {
            feature: Gauge(
                f'model_feature_{feature}_ks_statistic',
                f'KS statistic for {feature} distribution'
            )
            for feature in MODEL_CONFIG['feature_columns']
        }
        
        self.feature_ks_pvalue = {
            feature: Gauge(
                f'model_feature_{feature}_ks_pvalue',
                f'KS test p-value for {feature} distribution'
            )
            for feature in MODEL_CONFIG['feature_columns']
        }
        
        # Configurable drift thresholds
        self.kl_warning_threshold = 0.5
        self.kl_critical_threshold = 1.0
        self.ks_warning_threshold = 0.1  # KS statistic threshold for warning
        self.ks_critical_threshold = 0.2  # KS statistic threshold for critical
        
    def compute_kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Compute KL divergence between two distributions.
        
        Args:
            p: Current distribution
            q: Reference distribution
            
        Returns:
            float: KL divergence value
        """
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        p = p + epsilon
        q = q + epsilon
        
        # Normalize
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        return np.sum(p * np.log(p / q))
    
    def compute_ks_test(self, current_values: np.ndarray, reference_values: np.ndarray) -> tuple:
        """Compute Kolmogorov-Smirnov test between two samples.
        
        Args:
            current_values: Current distribution sample
            reference_values: Reference distribution sample
            
        Returns:
            tuple: (KS statistic, p-value)
        """
        return stats.ks_2samp(current_values, reference_values)

metrics = ModelMetrics()

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})

@app.route('/metrics')
def get_metrics():
    """Prometheus metrics endpoint."""
    return generate_latest()

@app.route('/evaluate')
def evaluate_model():
    """Evaluate model performance against ground truth data."""
    try:
        hours_back = int(request.args.get('hours_back', 1))
        min_samples = int(request.args.get('min_samples', 10))

        # Call the evaluation report with the updated parameter
        report = evaluator.generate_evaluation_report(hours_back=hours_back)
        
        # Handle the report as needed...
        
    except Exception as e:
        logger.error(f"Evaluation error: {str(e)}")
        return jsonify({"error": "Internal server error during evaluation"}), 500

        
    if not report or not report.get('performance_metrics'):
        return jsonify({
            'error': 'Not enough data for evaluation',
            'required_samples': min_samples,
            'available_samples': report.get('sample_sizes', {}).get('ground_truth', 0) if report else 0
        }), 400
        
    if report['sample_sizes']['ground_truth'] < min_samples:
        return jsonify({
            'error': 'Insufficient ground truth samples',
            'available_samples': report['sample_sizes']['ground_truth'],
            'required_samples': min_samples
        }), 400
        
    # Add evaluation metrics to Prometheus
    if report['performance_metrics']:
        metrics.accuracy.set(report['performance_metrics']['classification_report']['accuracy'])
        metrics.precision.set(report['performance_metrics']['classification_report']['1']['precision'])
        metrics.recall.set(report['performance_metrics']['classification_report']['1']['recall'])
        metrics.f1_score.set(report['performance_metrics']['classification_report']['1']['f1-score'])
    
    return jsonify({
        'status': 'success',
        'evaluation_report': report,
        'metadata': {
            'evaluation_time': datetime.now().isoformat(),
            'hours_evaluated': hours_back,
            'model_version': os.getenv('MODEL_VERSION', 'unknown')
        }
    })
    
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
        
        # Record feature distributions and compute KL divergence
        drift_metrics = {}
        for feature, value in zip(MODEL_CONFIG['feature_columns'], features):
            # Update feature histogram
            metrics.feature_values[feature].observe(value)
            
            # Update distribution tracking
            metrics.feature_distributions[feature].update(value)
            
            # Compute current distribution
            current_hist, _ = metrics.feature_distributions[feature].compute_distribution()
            
            # If we have enough data points, compute drift metrics
            window_size = len(metrics.feature_distributions[feature].values)
            metrics.drift_detection_window.set(window_size)
            
            if window_size >= MODEL_CONFIG['monitoring_window']:
                # Get reference distribution (uniform distribution as baseline)
                reference_dist = np.ones_like(current_hist) / len(current_hist)
                
                # Compute KL divergence
                kl_div = metrics.compute_kl_divergence(current_hist, reference_dist)
                metrics.feature_kl_divergence[feature].set(kl_div)
                metrics.max_kl_divergence.labels(feature=feature).set(kl_div)
                
                # Generate reference sample from uniform distribution
                reference_sample = np.random.uniform(0, 1, size=window_size)
                current_values = np.array(metrics.feature_distributions[feature].values)
                
                # Compute KS test
                ks_statistic, p_value = metrics.compute_ks_test(current_values, reference_sample)
                metrics.feature_ks_statistic[feature].set(ks_statistic)
                metrics.feature_ks_pvalue[feature].set(p_value)
                
                # Store drift metrics
                drift_metrics[feature] = {
                    'kl_divergence': float(kl_div),
                    'ks_statistic': float(ks_statistic),
                    'ks_pvalue': float(p_value)
                }
                
                # Check for drift threshold breaches
                if kl_div > metrics.kl_critical_threshold or ks_statistic > metrics.ks_critical_threshold:
                    metrics.drift_alerts.labels(feature=feature, severity='critical').inc()
                    metrics.drift_threshold_breaches.labels(feature=feature).inc()
                elif kl_div > metrics.kl_warning_threshold or ks_statistic > metrics.ks_warning_threshold:
                    metrics.drift_alerts.labels(feature=feature, severity='warning').inc()
        
        # Make prediction with latency tracking
        with metrics.prediction_latency.time():
            probabilities = model.predict_proba([features])[0]
            confidence = probabilities[1]
            prediction = int(confidence > MODEL_CONFIG['prediction_threshold'])
        
        # Record prediction metrics
        metrics.predictions_by_class.labels(prediction_class=str(prediction)).inc()
        metrics.prediction_confidence.observe(confidence)
        
        # Store prediction in BigQuery
        features_dict = dict(zip(MODEL_CONFIG['feature_columns'], features))

        store_prediction_in_bigquery(features_dict, prediction, confidence, drift_metrics)
        
        response = {
            "prediction": prediction,
            "confidence": float(confidence),
            "timestamp": start_time.isoformat(),
            "features_received": features_dict,
            "drift_metrics": drift_metrics
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
        print(data)

            
        prediction = int(data['prediction'])
        actual = int(data['actual'])
        

        
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