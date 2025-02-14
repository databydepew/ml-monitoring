"""
Drift Monitor Module

Monitors feature and prediction drift using KL divergence and statistical testing.
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy
from scipy.special import softmax
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
from prometheus_client import Gauge, Histogram
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
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


# Prometheus metrics
KL_DIVERGENCE = Gauge(
    'refinance_model_feature_kl_divergence',
    'KL divergence between reference and production distributions',
    ['feature']
)

P_VALUE = Gauge(
    'refinance_model_feature_drift_p_value',
    'P-value for distribution drift test',
    ['feature']
)

DRIFT_DETECTED = Gauge(
    'refinance_model_feature_drift_detected',
    'Whether drift was detected (1) or not (0)',
    ['feature']
)

PREDICTION_SHIFT = Gauge(
    'refinance_model_prediction_distribution_shift',
    'KL divergence in prediction distributions'
)

APPROVAL_RATE = Gauge(
    'refinance_model_approval_rate',
    'Rate of refinance approvals'
)

@dataclass
class DriftResult:
    """Results from a drift detection check"""
    feature_name: str
    kl_divergence: float
    p_value: float
    is_drift_detected: bool
    timestamp: datetime

class DriftMonitor:
    """Monitor distribution drift for features and predictions"""
    
    def __init__(
        self,
        reference_data: pd.DataFrame,
        window_size: int = 50,
        significance_level: float = 0.1
    ):
        """
        Initialize drift monitor
        
        Args:
            reference_data: Reference data to compare against
            window_size: Size of sliding window for production data
            significance_level: P-value threshold for drift detection
        """
        self.reference_data = reference_data
        self.window_size = window_size
        self.significance_level = significance_level
        
        # Initialize sliding windows for each feature
        self.windows = {
            col: deque(maxlen=window_size)
            for col in reference_data.columns
        }
        
        # Initialize metrics
        for col in reference_data.columns:
            KL_DIVERGENCE.labels(feature=col).set(0)
            P_VALUE.labels(feature=col).set(1)
            DRIFT_DETECTED.labels(feature=col).set(0)
    
    def _compute_kl_divergence(self, p: np.ndarray, q: np.ndarray, bins: int = 20) -> float:
        """Compute KL divergence between two samples"""
        # Compute histograms
        p_hist, _ = np.histogram(p, bins=bins, density=True)
        q_hist, _ = np.histogram(q, bins=bins, density=True)
        
        # Add small constant to avoid division by zero
        p_hist = p_hist + 1e-10
        q_hist = q_hist + 1e-10
        
        # Normalize
        p_hist = p_hist / p_hist.sum()
        q_hist = q_hist / q_hist.sum()
        
        return entropy(p_hist, q_hist)
    
    def _compute_p_value(self, reference: np.ndarray, production: np.ndarray, n_bootstrap: int = 500) -> float:
        """Compute p-value using bootstrap sampling"""
        observed_kl = self._compute_kl_divergence(reference, production)
        
        # Bootstrap samples
        bootstrap_kls = []
        n = len(reference)
        m = len(production)
        
        for _ in range(n_bootstrap):
            # Sample with replacement
            ref_sample = np.random.choice(reference, size=n, replace=True)
            prod_sample = np.random.choice(production, size=m, replace=True)
            bootstrap_kls.append(self._compute_kl_divergence(ref_sample, prod_sample))
        
        # Compute p-value
        return np.mean(np.array(bootstrap_kls) >= observed_kl)
    
    def check_drift(self, production_data: pd.DataFrame) -> Dict[str, DriftResult]:
        """
        Check for drift in new production data
        
        Args:
            production_data: New production data to check
            
        Returns:
            Dictionary of drift results for each feature
        """
        results = {}
        
        for col in self.reference_data.columns:
            # Update sliding window
            self.windows[col].extend(production_data[col].values)
            
            if len(self.windows[col]) < self.window_size:
                continue
                
            # Get window data
            window_data = np.array(list(self.windows[col]))
            
            # Compute KL divergence
            kl_div = self._compute_kl_divergence(
                self.reference_data[col].values,
                window_data
            )
            
            # Compute p-value
            p_value = self._compute_p_value(
                self.reference_data[col].values,
                window_data
            )
            
            # Check if drift detected
            is_drift = p_value < self.significance_level
            
            # Update metrics
            KL_DIVERGENCE.labels(feature=col).set(kl_div)
            P_VALUE.labels(feature=col).set(p_value)
            DRIFT_DETECTED.labels(feature=col).set(1 if is_drift else 0)
            
            # Store results
            results[col] = DriftResult(
                feature_name=col,
                kl_divergence=kl_div,
                p_value=p_value,
                is_drift_detected=is_drift,
                timestamp=datetime.now()
            )
        
        return results
    
    def get_drift_report(self, results: Dict[str, DriftResult]) -> str:
        """Generate human-readable drift report"""
        report = []
        report.append("Distribution Drift Report")
        report.append("=" * 25)
        report.append("")
        
        for feature, result in results.items():
            report.append(f"Feature: {feature}")
            report.append(f"Status: {'DRIFT DETECTED' if result.is_drift_detected else 'No drift'}")
            report.append(f"KL Divergence: {result.kl_divergence:.4f}")
            report.append(f"P-value: {result.p_value:.4f}")
            report.append(f"Timestamp: {result.timestamp}")
            report.append("")
        
        return "\n".join(report)

class PredictionDriftMonitor(DriftMonitor):
    """Monitor drift in model predictions"""
    
    def check_prediction_drift(self, predictions: np.ndarray) -> DriftResult:
        """Check for drift in model predictions"""
        # Update sliding window
        self.windows['predictions'].extend(predictions)
        
        if len(self.windows['predictions']) < self.window_size:
            return None
            
        # Get window data
        window_data = np.array(list(self.windows['predictions']))
        
        # Compute KL divergence
        kl_div = self._compute_kl_divergence(
            self.reference_data['predictions'].values,
            window_data
        )
        
        # Compute p-value
        p_value = self._compute_p_value(
            self.reference_data['predictions'].values,
            window_data
        )
        
        # Check if drift detected
        is_drift = p_value < self.significance_level
        
        # Update metrics
        PREDICTION_SHIFT.set(kl_div)
        APPROVAL_RATE.set(np.mean(window_data > 0.5))
        
        return DriftResult(
            feature_name='predictions',
            kl_divergence=kl_div,
            p_value=p_value,
            is_drift_detected=is_drift,
            timestamp=datetime.now()
        )

def setup_monitoring(
    reference_data: pd.DataFrame,
    reference_predictions: np.ndarray,
    window_size: int = 50,
    significance_level: float = 0.1
) -> tuple[DriftMonitor, PredictionDriftMonitor]:
    """
    Set up feature and prediction drift monitoring
    
    Args:
        reference_data: Reference data for feature monitoring
        reference_predictions: Reference predictions for prediction monitoring
        window_size: Size of sliding window
        significance_level: P-value threshold for drift detection
        
    Returns:
        Tuple of (feature_monitor, prediction_monitor)
    """
    # Set up feature monitoring
    feature_monitor = DriftMonitor(
        reference_data=reference_data,
        window_size=window_size,
        significance_level=significance_level
    )
    
    # Set up prediction monitoring
    prediction_monitor = PredictionDriftMonitor(
        reference_data=pd.DataFrame({'predictions': reference_predictions}),
        window_size=window_size,
        significance_level=significance_level
    )
    
    logger.info("Drift monitoring initialized successfully")
    return feature_monitor, prediction_monitor
