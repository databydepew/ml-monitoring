"""
Distribution Monitoring Module

This module implements KL divergence testing to monitor for:
1. Feature distribution shifts (data drift)
2. Prediction distribution shifts (concept drift)
3. Continuous monitoring with sliding windows
"""

import numpy as np
from scipy.stats import entropy
from scipy.special import softmax
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from prometheus_client import Gauge, Histogram
import pandas as pd
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DistributionStats:
    feature_name: str
    kl_divergence: float
    p_value: float
    timestamp: datetime
    window_size: int
    is_significant: bool

from .bigquery_reference import BigQueryReference

class DistributionMonitor:
    """Monitors distribution shifts using KL divergence"""
    
    # Class-level Prometheus metrics
    # Basic metrics
    kl_divergence_gauge = Gauge(
        'refinance_model_feature_kl_divergence',
        'KL divergence between training and production distributions for refinance features',
        ['feature']
    )
    p_value_gauge = Gauge(
        'refinance_model_feature_drift_p_value',
        'P-value for refinance feature distribution drift test',
        ['feature']
    )
    drift_detected_gauge = Gauge(
        'refinance_model_feature_drift_detected',
        'Whether significant drift was detected (1) or not (0)',
        ['feature']
    )
    
    # Distribution histograms
    reference_hist = Histogram(
        'reference_distribution',
        'Distribution of reference data',
        ['feature']
    )
    production_hist = Histogram(
        'production_distribution',
        'Distribution of production data',
        ['feature']
    )
    
    def __init__(
        self,
        reference_data: Optional[pd.DataFrame] = None,
        window_size: int = 1000,
        significance_level: float = 0.05,
        n_bootstrap: int = 1000,
        bigquery_config: Optional[dict] = None
    ):
        """
        Initialize the distribution monitor
        
        Args:
            reference_data: Training data used as reference
            window_size: Size of sliding window for production data
            significance_level: Threshold for statistical significance
            n_bootstrap: Number of bootstrap samples for p-value calculation
        """
        self.window_size = window_size
        
        # Initialize BigQuery reference if config provided
        self.bq_reference = None
        if bigquery_config:
            self.bq_reference = BigQueryReference(
                project_id=bigquery_config['project_id'],
                dataset_id=bigquery_config['dataset_id'],
                table_id=bigquery_config['table_id'],
                feature_columns=bigquery_config['feature_columns'],
                target_column=bigquery_config['target_column']
            )
            # Get a small sample as reference data
            self.reference_data = self.bq_reference.get_reference_sample(1000)
        else:
            self.reference_data = reference_data
            
        if self.reference_data is None:
            raise ValueError("Either reference_data or bigquery_config must be provided")
        self.significance_level = significance_level
        self.n_bootstrap = n_bootstrap
        
        # Store production data in sliding windows
        self.production_windows: Dict[str, deque] = {}
        
        # Initialize windows for each feature
        for col in self.reference_data.columns:
            # Skip target column if bq_reference is available
            if self.bq_reference is None or col != self.bq_reference.target_column:
                self.production_windows[col] = deque(maxlen=window_size)
                # Initialize metrics with 0
                self.kl_divergence_gauge.labels(feature=col).set(0)
                self.p_value_gauge.labels(feature=col).set(1)
                self.drift_detected_gauge.labels(feature=col).set(0)
        
        # Compute reference distributions
        self.reference_distributions = self._compute_distributions(self.reference_data)

    def _compute_distributions(
        self,
        data: pd.DataFrame,
        n_bins: int = 50
    ) -> Dict[str, np.ndarray]:
        """Compute normalized histograms for each feature"""
        distributions = {}
        
        for column in data.columns:
            # Skip target column
            if self.bq_reference and column == self.bq_reference.target_column:
                continue
                
            # Compute histogram
            hist, _ = np.histogram(
                data[column].dropna(),
                bins=n_bins,
                density=True
            )
            # Add small constant to avoid zero probabilities
            hist = hist + 1e-10
            # Normalize
            hist = hist / hist.sum()
            distributions[column] = hist
            
            # Update reference histograms in Prometheus
            if isinstance(data, pd.DataFrame):
                for value in data[column].dropna():
                    try:
                        self.reference_hist.labels(feature=column).observe(float(value))
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Could not record histogram for {column}: {e}")
                    
        return distributions

    def _bootstrap_kl_divergence(
        self,
        p: np.ndarray,
        q: np.ndarray,
        n_samples: int = 1000,
        sample_size: int = None
    ) -> List[float]:
        """
        Compute bootstrap samples of KL divergence
        
        Args:
            p: First distribution
            q: Second distribution
            n_samples: Number of bootstrap samples
            sample_size: Size of each bootstrap sample
            
        Returns:
            List of KL divergence values from bootstrap samples
        """
        if sample_size is None:
            sample_size = len(p)
            
        kl_samples = []
        for _ in range(n_samples):
            # Sample with replacement
            idx = np.random.choice(len(p), size=sample_size)
            p_sample = p[idx]
            q_sample = q[idx]
            
            # Normalize
            p_sample = p_sample / p_sample.sum()
            q_sample = q_sample / q_sample.sum()
            
            # Compute KL divergence
            kl_samples.append(entropy(p_sample, q_sample))
            
        return kl_samples

    def _compute_p_value(
        self,
        observed_kl: float,
        bootstrap_kls: List[float]
    ) -> float:
        """
        Compute p-value from bootstrap samples
        
        Args:
            observed_kl: Observed KL divergence
            bootstrap_kls: List of KL divergences from bootstrap
            
        Returns:
            p-value
        """
        return np.mean([kl >= observed_kl for kl in bootstrap_kls])

    def add_production_data(
        self,
        new_data: pd.DataFrame
    ) -> Dict[str, DistributionStats]:
        """
        Add new production data and check for distribution shifts
        
        Args:
            new_data: New production data to monitor
            
        Returns:
            Dictionary of distribution statistics for each feature
        """
        results = {}
        
        for column in new_data.columns:
            # Add new data to sliding window
            self.production_windows[column].extend(new_data[column].dropna())
            
            # Skip if we don't have enough data
            if len(self.production_windows[column]) < self.window_size // 2:
                continue
                
            # Compute production distribution
            prod_hist, _ = np.histogram(
                list(self.production_windows[column]),
                bins=50,
                density=True
            )
            prod_hist = prod_hist + 1e-10
            prod_hist = prod_hist / prod_hist.sum()
            
            # Update production histograms in Prometheus
            for value in new_data[column].dropna():
                self.production_hist.labels(feature=column).observe(value)
            
            # Compute KL divergence
            kl_div = entropy(self.reference_distributions[column], prod_hist)
            
            # Bootstrap for p-value
            bootstrap_kls = self._bootstrap_kl_divergence(
                self.reference_distributions[column],
                prod_hist,
                n_samples=self.n_bootstrap
            )
            p_value = self._compute_p_value(kl_div, bootstrap_kls)
            
            # Update Prometheus metrics
            self.kl_divergence_gauge.labels(feature=column).set(kl_div)
            self.p_value_gauge.labels(feature=column).set(p_value)
            self.drift_detected_gauge.labels(feature=column).set(
                1 if p_value < self.significance_level else 0
            )
            
            # Store results
            results[column] = DistributionStats(
                feature_name=column,
                kl_divergence=kl_div,
                p_value=p_value,
                timestamp=datetime.now(),
                window_size=len(self.production_windows[column]),
                is_significant=p_value < self.significance_level
            )
            
        return results

    def get_drift_report(
        self,
        results: Dict[str, DistributionStats]
    ) -> str:
        """Generate a human-readable drift report"""
        report = ["Distribution Drift Report", "=" * 25, ""]
        
        for feature, stats in results.items():
            status = "DRIFT DETECTED" if stats.is_significant else "No significant drift"
            report.extend([
                f"Feature: {feature}",
                f"Status: {status}",
                f"KL Divergence: {stats.kl_divergence:.4f}",
                f"P-value: {stats.p_value:.4f}",
                f"Window Size: {stats.window_size}",
                f"Timestamp: {stats.timestamp}",
                ""
            ])
            
        return "\n".join(report)

class PredictionDistributionMonitor(DistributionMonitor):
    """Monitors shifts in prediction distributions"""
    
    # Class-level Prometheus metrics
    prediction_shift_gauge = Gauge(
        'refinance_model_prediction_distribution_shift',
        'KL divergence in refinance prediction distributions'
    )
    refinance_rate_gauge = Gauge(
        'refinance_model_approval_rate',
        'Rate of refinance approvals in recent predictions'
    )
    
    def __init__(
        self,
        reference_predictions: np.ndarray,
        window_size: int = 1000,
        significance_level: float = 0.05,
        n_bootstrap: int = 1000
    ):
        """
        Initialize prediction distribution monitor
        
        Args:
            reference_predictions: Predictions from training/validation set
            window_size: Size of sliding window for production predictions
            significance_level: Threshold for statistical significance
            n_bootstrap: Number of bootstrap samples for p-value calculation
        """
        # Convert predictions to DataFrame for parent class
        ref_df = pd.DataFrame({'predictions': reference_predictions})
        super().__init__(
            ref_df,
            window_size,
            significance_level,
            n_bootstrap
        )

def setup_distribution_monitoring(
    reference_data: pd.DataFrame,
    reference_predictions: np.ndarray
) -> Tuple[DistributionMonitor, PredictionDistributionMonitor]:
    # More sensitive monitoring parameters
    window_size = 50  # Smaller window to detect changes faster
    significance_level = 0.1  # More sensitive threshold (was 0.05)
    n_bootstrap = 500  # Fewer bootstraps for faster updates
    """Setup both feature and prediction distribution monitoring"""
    
    feature_monitor = DistributionMonitor(
        reference_data=reference_data,
        window_size=window_size,
        significance_level=significance_level,
        n_bootstrap=n_bootstrap
    )
    
    prediction_monitor = PredictionDistributionMonitor(
        reference_predictions=reference_predictions,
        window_size=window_size,
        significance_level=significance_level,
        n_bootstrap=n_bootstrap
    )
    
    logger.info("Distribution monitoring initialized successfully")
    return feature_monitor, prediction_monitor
