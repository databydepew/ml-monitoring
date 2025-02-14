"""
Distribution Monitor Service

Runs the distribution monitors as a standalone service that:
1. Fetches data from BigQuery
2. Monitors feature and prediction distributions
3. Exposes metrics via Prometheus
"""

import os
import time
import logging
from prometheus_client import start_http_server
from drift_monitor import setup_monitoring
from google.cloud import bigquery
import pandas as pd
import numpy as np
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# BigQuery Configuration
BIGQUERY_CONFIG = {
    'project_id': 'mdepew-assets',
    'dataset_id': 'synthetic',
    'table_id': 'synthetic_mortgage_data',
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
    'target_column': 'refinance'
}

def fetch_reference_data(client: bigquery.Client, limit: int = 1000) -> pd.DataFrame:
    """Fetch reference data from BigQuery"""
    query = f"""
    SELECT
        {', '.join(BIGQUERY_CONFIG['feature_columns'])},
        {BIGQUERY_CONFIG['target_column']}
    FROM
        `{BIGQUERY_CONFIG['project_id']}.{BIGQUERY_CONFIG['dataset_id']}.{BIGQUERY_CONFIG['table_id']}`
    ORDER BY RAND()
    LIMIT {limit}
    """
    
    logger.info("Fetching reference data from BigQuery...")
    return client.query(query).to_dataframe()

def fetch_production_data(client: bigquery.Client) -> pd.DataFrame:
    """Fetch recent production data from BigQuery"""
    query = f"""
    SELECT
        {', '.join(BIGQUERY_CONFIG['feature_columns'])},
        {BIGQUERY_CONFIG['target_column']}
    FROM
        `{BIGQUERY_CONFIG['project_id']}.{BIGQUERY_CONFIG['dataset_id']}.{BIGQUERY_CONFIG['table_id']}`
    WHERE
        DATE(CURRENT_TIMESTAMP()) = CURRENT_DATE()
    """
    
    logger.info("Fetching production data from BigQuery...")
    return client.query(query).to_dataframe()

def main():
    # Start Prometheus metrics server with the correct content type
    start_http_server(8001)
    logger.info("Started Prometheus metrics server on port 8001")
    
    # Initialize BigQuery client
    client = bigquery.Client()
    
    # Get reference data and model
    reference_data = fetch_reference_data(client)
    model = joblib.load('model.pkl')
    reference_predictions = model.predict_proba(reference_data[BIGQUERY_CONFIG['feature_columns']])[:, 1]
    
    # Setup monitoring
    feature_monitor, prediction_monitor = setup_monitoring(
        reference_data=reference_data[BIGQUERY_CONFIG['feature_columns']],
        reference_predictions=reference_predictions,
        window_size=50,
        significance_level=0.1
    )
    
    # Initialize BigQuery client
    client = bigquery.Client()
    
    # Load model for predictions
    model = joblib.load('model.pkl')
    
    # Get configuration from environment
    monitoring_interval = int(os.getenv('MONITORING_INTERVAL_MINUTES', '60'))
    window_size = int(os.getenv('WINDOW_SIZE', '50'))
    significance_level = float(os.getenv('SIGNIFICANCE_LEVEL', '0.1'))
    n_bootstrap = int(os.getenv('N_BOOTSTRAP', '500'))
    
    # Fetch reference data and compute reference predictions
    reference_data = fetch_reference_data(client)
    reference_predictions = model.predict_proba(
        reference_data[BIGQUERY_CONFIG['feature_columns']]
    )[:, 1]
    
    # Initialize drift monitors
    feature_monitor, prediction_monitor = setup_monitoring(
        reference_data=reference_data[BIGQUERY_CONFIG['feature_columns']],
        reference_predictions=reference_predictions,
        window_size=window_size,
        significance_level=significance_level
    )
    
    logger.info("Distribution monitoring initialized successfully")
    
    while True:
        try:
            # Fetch recent production data
            production_data = fetch_production_data(client)
            
            if len(production_data) > 0:
                # Get production predictions
                production_predictions = model.predict_proba(
                    production_data[BIGQUERY_CONFIG['feature_columns']]
                )[:, 1]
                
                # Monitor feature distributions
                feature_drift = feature_monitor.check_drift(production_data[BIGQUERY_CONFIG['feature_columns']])
                logger.info("Feature Drift Report:\n" + 
                          feature_monitor.get_drift_report(feature_drift))
                
                # Monitor prediction distributions
                prediction_drift = prediction_monitor.check_prediction_drift(production_predictions)
                if prediction_drift:
                    logger.info(f"Prediction Drift Report:\n" +
                              f"KL Divergence: {prediction_drift.kl_divergence:.4f}\n" +
                              f"P-value: {prediction_drift.p_value:.4f}\n" +
                              f"Drift Detected: {prediction_drift.is_drift_detected}")
            
            # Sleep for monitoring interval
            time.sleep(monitoring_interval * 60)
            
        except Exception as e:
            logger.error(f"Error in monitoring loop: {str(e)}")
            time.sleep(60)  # Wait a minute before retrying

if __name__ == '__main__':
    main()
