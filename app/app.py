from flask import Flask, request, jsonify
import joblib
import numpy as np
from prometheus_client import start_http_server, Counter, Histogram, Gauge
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize Prometheus metrics
PREDICTION_REQUESTS = Counter('refinance_prediction_requests_total', 'Total prediction requests')
PREDICTION_LATENCY = Histogram('refinance_prediction_latency_seconds', 'Time spent processing prediction')
PREDICTION_PROBABILITY = Histogram('refinance_prediction_probability', 'Distribution of refinance probabilities')
PREDICTION_RESULT = Counter('refinance_prediction_result', 'Prediction results', ['result'])
MODEL_VERSION = Gauge('model_version', 'Model version information')

try:
    # Load the model
    model = joblib.load('model.pkl')
    MODEL_VERSION.set(1)  # Set initial model version
    logger.info("Successfully loaded model.pkl")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

@app.route('/predict', methods=['POST'])
def predict():
    PREDICTION_REQUESTS.inc()
    start_time = time.time()
    
    try:
        data = request.json
        features = np.array([
            data['interest_rate'],
            data['loan_amount'],
            data['loan_balance'],
            data['loan_to_value_ratio'],
            data['credit_score'],
            data['debt_to_income_ratio'],
            data['income'],
            data['loan_term'],
            data['loan_age'],
            data['home_value'],
            data['current_rate'],
            data['rate_spread']
        ]).reshape(1, -1)
        
        # Make prediction
        probability = model.predict_proba(features)[0][1]
        prediction = int(probability > 0.5)  # Convert to 0 or 1
        
        # Record metrics
        PREDICTION_LATENCY.observe(time.time() - start_time)
        PREDICTION_PROBABILITY.observe(probability)
        PREDICTION_RESULT.labels(result=str(prediction)).inc()
        
        logger.info(f"Prediction made: {prediction}, probability: {probability:.3f}")
        
        return jsonify({
            'prediction': prediction,  # Returns 0 or 1
            'probability': float(probability)
        })
    
    except KeyError as e:
        logger.error(f"Missing required feature: {str(e)}")
        return jsonify({
            'error': f"Missing required feature: {str(e)}"
        }), 400
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'error': f"Prediction failed: {str(e)}"
        }), 500

@app.route('/metrics')
def metrics():
    from prometheus_client import generate_latest
    return generate_latest()

if __name__ == '__main__':
    # Start up the server to expose metrics
    start_http_server(8000)
    logger.info("Started Prometheus metrics server on port 8000")
    
    # Start Flask app
    app.run(host='0.0.0.0', port=5001)
    logger.info("Started Flask application on port 5001")
