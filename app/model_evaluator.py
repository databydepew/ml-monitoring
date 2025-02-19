from google.cloud import bigquery
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import logging
from datetime import datetime, timedelta
import scipy.stats as stats

class ModelEvaluator:
    def __init__(self, project_id, dataset_id, table_id):
        """Initialize the ModelEvaluator.
        
        Args:
            project_id: GCP project ID
            dataset_id: BigQuery dataset ID
            table_id: BigQuery table ID containing ground truth data
        """
        self.client = bigquery.Client(project=project_id)
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = table_id
        self.logger = logging.getLogger(__name__)

    def get_ground_truth_data(self, start_date=None, end_date=None):
        """Query ground truth data from BigQuery."""
        where_clauses = []
        
        if start_date:
            where_clauses.append(f"timestamp >= '{start_date}'")
        if end_date:
            where_clauses.append(f"timestamp <= '{end_date}'")
        
        # Add condition for refinance not being null
        where_clauses.append("refinance IS NOT NULL")
        
        where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        query = f"""
        SELECT 
            interest_rate,
            loan_amount,
            loan_balance,
            loan_to_value_ratio,
            credit_score,
            debt_to_income_ratio,
            income,
            loan_term,
            loan_age,
            home_value,
            current_rate,
            rate_spread,
            refinance AS actual_outcome,
            DATETIME(timestamp) as prediction_time
        FROM `{self.project_id}.{self.dataset_id}.{self.table_id}`
        {where_clause}
        """
        
        df = self.client.query(query).to_dataframe()
        if df.isnull().values.any():
            self.logger.warning("Ground truth data contains NaN values.")
            df = df.dropna()  # or handle appropriately
        
        return df
    
    def compare_predictions(self, model_predictions, ground_truth):
        """Compare model predictions with ground truth data.
        
        Args:
            model_predictions: DataFrame with model predictions
            ground_truth: DataFrame with ground truth data
            
        Returns:
            dict containing evaluation metrics
        """
        merged_data = pd.merge(
            model_predictions,
            ground_truth,
            on=['interest_rate', 'loan_amount', 'loan_balance', 
                'loan_to_value_ratio', 'credit_score', 'debt_to_income_ratio',
                'income', 'loan_term', 'loan_age', 'home_value', 'current_rate',
                'rate_spread'],
            how='inner'
        )
        
        if merged_data.empty:
            self.logger.warning("No matching records found between predictions and ground truth")
            return None
            
        y_true = merged_data['actual_outcome']
        y_pred = merged_data['prediction']
        
        # Calculate metrics
        conf_matrix = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # Calculate average prediction confidence by actual outcome
        avg_conf_by_outcome = merged_data.groupby('actual_outcome')['confidence'].mean()
        
        metrics = {
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': report,
            'sample_size': len(merged_data),
            'time_range': {
                'start': merged_data['prediction_time'].min().isoformat(),
                'end': merged_data['prediction_time'].max().isoformat()
            },
            'avg_confidence': {
                'overall': float(merged_data['confidence'].mean()),
                'by_outcome': avg_conf_by_outcome.to_dict()
            }
        }
        
        return metrics

    def analyze_feature_drift(self, model_predictions, ground_truth):
        """Analyze feature drift between predictions and ground truth.
        
        Args:
            model_predictions: DataFrame with recent predictions
            ground_truth: DataFrame with historical ground truth
            
        Returns:
            dict containing drift analysis results
        """
        feature_columns = [
            'interest_rate', 'loan_amount', 'loan_balance', 
            'loan_to_value_ratio', 'credit_score', 'debt_to_income_ratio',
            'income', 'loan_term', 'loan_age', 'home_value', 'current_rate',
            'rate_spread'
        ]
        
        drift_analysis = {}
        
        for feature in feature_columns:
            # Calculate basic statistics
            pred_stats = model_predictions[feature].describe()
            truth_stats = ground_truth[feature].describe()
            
            # Perform Kolmogorov-Smirnov test
            ks_statistic, p_value = stats.ks_2samp(
                model_predictions[feature],
                ground_truth[feature]
            )
            
            drift_analysis[feature] = {
                'statistics': {
                    'predictions': pred_stats.to_dict(),
                    'ground_truth': truth_stats.to_dict()
                },
                'drift_metrics': {
                    'ks_statistic': float(ks_statistic),
                    'p_value': float(p_value)
                }
            }
        
        return drift_analysis

    def generate_evaluation_report(self, hours_back=1):
        """Generate a comprehensive evaluation report.
        
        Args:
            hours_back: Number of hours to look back for comparison (default: 1)
        
        Returns:
            dict containing the evaluation report
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=hours_back)
        
        # Get recent predictions and ground truth
        ground_truth = self.get_ground_truth_data(start_date, end_date)
        
        if ground_truth.empty:
            self.logger.warning(f"No ground truth data found for the last {hours_back} hours")
            return None
        
        # Get stored predictions from your prediction tracking table
        predictions_query = f"""
        SELECT *
        FROM `{self.project_id}.{self.dataset_id}.model_predictions`
        WHERE DATETIME(timestamp) BETWEEN 
            DATETIME('{start_date.isoformat()}') AND DATETIME('{end_date.isoformat()}')
        """
        model_predictions = self.client.query(predictions_query).to_dataframe()
        
        if model_predictions.empty:
            self.logger.warning("No model predictions found for the specified time range")
            return None
        
        # Generate comprehensive report
        report = {
            'time_range': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'sample_sizes': {
                'ground_truth': len(ground_truth),
                'predictions': len(model_predictions)
            },
            'performance_metrics': self.compare_predictions(model_predictions, ground_truth),
            'drift_analysis': self.analyze_feature_drift(model_predictions, ground_truth)
        }
        
        return report
