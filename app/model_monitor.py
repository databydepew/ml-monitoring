from google.cloud import bigquery
import pandas as pd
import numpy as np
from prometheus_client import start_http_server, Gauge, Counter, Histogram
import time
import joblib
import logging
from datetime import datetime, timedelta
from drift_monitor import setup_monitoring

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Define Prometheus metrics for model monitoring
MODEL_ACCURACY = Gauge('refinance_model_accuracy', 'Model accuracy on recent data')
MODEL_PRECISION = Gauge('refinance_model_precision', 'Model precision on recent data')
MODEL_RECALL = Gauge('refinance_model_recall', 'Model recall on recent data')
MODEL_F1 = Gauge('refinance_model_f1', 'Model F1 score on recent data')
GROUND_TRUTH_REFINANCE_RATE = Gauge('ground_truth_refinance_rate', 'Actual refinance rate from BigQuery')
PREDICTION_ERROR_RATE = Gauge('prediction_error_rate', 'Rate of incorrect predictions')
FEATURE_DRIFT = Gauge('feature_drift', 'Feature drift score', ['feature', 'metric_type'])
PREDICTION_DISTRIBUTION = Gauge('prediction_distribution', 'Distribution of prediction probabilities', ['bucket'])
FEATURE_IMPORTANCE = Gauge('feature_importance', 'Feature importance scores', ['feature'])



from google.cloud import bigquery

def test_bq_connection():
    try:
        # Create a BigQuery client
        client = bigquery.Client()
        # get the column names
        columns = ', '.join(BIGQUERY_CONFIG['feature_columns'])
        # Perform a simple query to test the connection
        query_job = client.query(f"SELECT {columns}")
        result = query_job.result()  # Wait for the job to complete

        for row in result:
            print(f"Query result: {row}")

        print("BigQuery connection successful!")
    except Exception as e:
        print(f"Error connecting to BigQuery: {e}")




class ModelMonitor:
    def __init__(self, model_path='model.pkl', monitoring_interval_minutes=60):
        self.client = bigquery.Client()
        self.model = joblib.load(model_path)
        self.monitoring_interval = monitoring_interval_minutes
        
        # Initialize drift monitors
        reference_data = self.fetch_recent_data()
        reference_predictions = self.model.predict_proba(reference_data[BIGQUERY_CONFIG['feature_columns']])[:, 1]
        self.feature_monitor, self.prediction_monitor = setup_monitoring(
            reference_data=reference_data[BIGQUERY_CONFIG['feature_columns']],
            reference_predictions=reference_predictions,
            window_size=50,
            significance_level=0.1
        )
        
    def fetch_recent_data(self):
        """Fetch recent data from BigQuery for monitoring"""
        query = f"""
        SELECT
            {', '.join(BIGQUERY_CONFIG['feature_columns'])},
            {BIGQUERY_CONFIG['target_column']}
        FROM
            `{BIGQUERY_CONFIG['project_id']}.{BIGQUERY_CONFIG['dataset_id']}.{BIGQUERY_CONFIG['table_id']}`
        WHERE
            DATE(CURRENT_TIMESTAMP()) = CURRENT_DATE()
        """
        
        logger.info("Fetching recent data from BigQuery...")
        df = self.client.query(query).to_dataframe()
        logger.info(f"Fetched {len(df)} records")
        return df
    
    def calculate_feature_drift(self, current_data):
        """Calculate feature drift using multiple metrics"""
        if len(current_data) == 0:
            logger.warning("No current data available for feature drift calculation")
            return
            
        query = f"""
        SELECT
            {', '.join(BIGQUERY_CONFIG['feature_columns'])}
        FROM
            `{BIGQUERY_CONFIG['project_id']}.{BIGQUERY_CONFIG['dataset_id']}.{BIGQUERY_CONFIG['table_id']}`
        WHERE
            DATE(CURRENT_TIMESTAMP()) != CURRENT_DATE()
        LIMIT 10000
        """
        historical_data = self.client.query(query).to_dataframe()
        historical_data = historical_data.reindex(columns=BIGQUERY_CONFIG['feature_columns'], axis=1)
        
        if len(historical_data) == 0:
            logger.warning("No historical data available for feature drift calculation")
            return

        from scipy.stats import ks_2samp
        from scipy.spatial.distance import jensenshannon
        import numpy as np
            
        for feature in BIGQUERY_CONFIG['feature_columns']:
            try:
                # KS test
                ks_stat, _ = ks_2samp(historical_data[feature], current_data[feature])
                FEATURE_DRIFT.labels(feature=feature, metric_type='ks_statistic').set(ks_stat)
                
                # Calculate histograms for distribution metrics
                hist_current, bins = np.histogram(current_data[feature], bins=20, density=True)
                hist_historical, _ = np.histogram(historical_data[feature], bins=bins, density=True)
                
                # Add small epsilon to avoid zero probabilities
                eps = 1e-10
                hist_current = hist_current + eps
                hist_historical = hist_historical + eps
                
                # Normalize
                hist_current = hist_current / hist_current.sum()
                hist_historical = hist_historical / hist_historical.sum()
                
                # Jensen-Shannon divergence
                js_div = jensenshannon(hist_current, hist_historical)
                FEATURE_DRIFT.labels(feature=feature, metric_type='jensen_shannon_div').set(js_div)
                
                # Population Stability Index (PSI)
                psi = np.sum((hist_current - hist_historical) * np.log(hist_current / hist_historical))
                FEATURE_DRIFT.labels(feature=feature, metric_type='psi').set(psi)
                
            except Exception as e:
                logger.error(f"Error calculating drift for feature {feature}: {str(e)}")
                
        # Calculate and update feature importance
        try:
            importances = self.model.feature_importances_
            for feature, importance in zip(BIGQUERY_CONFIG['feature_columns'], importances):
                FEATURE_IMPORTANCE.labels(feature=feature).set(importance)
        except:
            logger.warning("Could not calculate feature importances")

        # Track prediction probability distribution
        y_prob = self.model.predict_proba(current_data[BIGQUERY_CONFIG['feature_columns']])[:, 1]
        hist, bins = np.histogram(y_prob, bins=10, range=(0, 1))
        for i, (count, bin_start) in enumerate(zip(hist, bins[:-1])):
            bucket = f"{bin_start:.1f}-{bins[i+1]:.1f}"
            PREDICTION_DISTRIBUTION.labels(bucket=bucket).set(count)
    
    def calculate_metrics(self, y_true, y_pred, y_prob):
        """Calculate and export monitoring metrics"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        MODEL_ACCURACY.set(accuracy)
        MODEL_PRECISION.set(precision)
        MODEL_RECALL.set(recall)
        MODEL_F1.set(f1)
        
        refinance_rate = np.mean(y_true)
        GROUND_TRUTH_REFINANCE_RATE.set(refinance_rate)
        
        error_rate = 1 - accuracy
        PREDICTION_ERROR_RATE.set(error_rate)
        
        logger.info(f"Metrics - Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, "
                   f"Recall: {recall:.3f}, F1: {f1:.3f}")
    
    def monitor(self):
        """Main monitoring loop"""
        while True:
            try:
                # Fetch recent data
                df = self.fetch_recent_data()
                logger.info(f"Fetched {len(df)} records")
                feature_columns = ['interest_rate', 'loan_amount', 'loan_balance', 'loan_to_value_ratio',
                                    'credit_score', 'debt_to_income_ratio', 'income', 'loan_term', 'loan_age',
                                    'home_value', 'current_rate', 'rate_spread']
                
                if len(df) > 0:
                    # Split features and target
                    X = df[feature_columns]
                    y_true = df[BIGQUERY_CONFIG['target_column']]
                    
                    # Make predictions
                    y_pred = self.model.predict(X)
                    y_prob = self.model.predict_proba(X)[:, 1]
                    
                    # Calculate and export metrics
                    self.calculate_metrics(y_true, y_pred, y_prob)
                    
                    # Check for drift
                    feature_drift = self.feature_monitor.check_drift(X)
                    prediction_drift = self.prediction_monitor.check_prediction_drift(y_prob)
                    
                    # Log drift reports
                    logger.info("Feature Drift Report:\n" + 
                              self.feature_monitor.get_drift_report(feature_drift))
                    if prediction_drift:
                        logger.info(f"Prediction Drift Report:\n" +
                                  f"KL Divergence: {prediction_drift.kl_divergence:.4f}\n" +
                                  f"P-value: {prediction_drift.p_value:.4f}\n" +
                                  f"Drift Detected: {prediction_drift.is_drift_detected}")
                    
                    # Sleep for monitoring interval
                    time.sleep(self.monitoring_interval * 60)
                    
                    # Log drift reports
                    logger.info("Feature Drift Report:\n" + 
                              self.feature_monitor.get_drift_report(feature_drift))
                    logger.info("Prediction Drift Report:\n" + 
                              self.prediction_monitor.get_drift_report(prediction_drift))
                    
                    # Calculate feature drift
                    self.calculate_feature_drift(df[BIGQUERY_CONFIG['feature_columns']])
                    
                    logger.info("Successfully updated monitoring metrics")
                else:
                    logger.warning("No recent data found for monitoring")
            
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
            
            # Wait for next monitoring interval
            time.sleep(self.monitoring_interval * 60)

if __name__ == '__main__':
    # Start Prometheus metrics server with the correct content type
    start_http_server(8000)
    logger.info("Started Prometheus metrics server on port 8001")

    # # Create and start the model monitor
    monitor = ModelMonitor()
    monitor.monitor()
