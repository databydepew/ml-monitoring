"""Test script for refinance model distribution monitoring.

This script tests the monitoring system's ability to track and analyze
refinance-specific features including:
1. Rate spreads and interest rate distributions
2. Risk metrics (LTV, DTI, credit scores)
3. Loan characteristics
4. Income and affordability metrics

It validates both the data retrieval from BigQuery and the drift detection
mechanisms for refinance applications.
"""

from distribution_monitor import DistributionMonitor
from config import BIGQUERY_CONFIG, MONITORING_CONFIG, FEATURE_GROUPS
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_refinance_monitoring():
    """Test the refinance model monitoring system."""
    
    # Initialize monitor with BigQuery configuration
    monitor = DistributionMonitor(
        window_size=MONITORING_CONFIG['window_size'],
        bigquery_config=BIGQUERY_CONFIG
    )
    
    # Test rate metrics monitoring
    logger.info("\nTesting rate metrics monitoring...")
    rate_metrics = ['current_rate', 'interest_rate', 'rate_spread']
    for feature in rate_metrics:
        dist = monitor.get_reference_distribution(feature)
        stats = monitor.get_feature_statistics(feature)
        logger.info(f"\n{feature.replace('_', ' ').title()} Analysis:")
        logger.info(f"Distribution shape: {dist.shape}")
        logger.info(f"Mean: {stats['mean']:.2f}%")
        logger.info(f"Std: {stats['std']:.2f}%")
        if feature == 'rate_spread':
            logger.info(f"Median savings: {stats['median']:.2f}%")
    
    # Test risk metrics monitoring
    logger.info("\nTesting risk metrics monitoring...")
    risk_metrics = ['credit_score', 'debt_to_income_ratio', 'loan_to_value_ratio']
    for feature in risk_metrics:
        stats = monitor.get_feature_statistics(feature)
        logger.info(f"\n{feature.replace('_', ' ').title()} Analysis:")
        logger.info(f"Mean: {stats['mean']:.2f}")
        logger.info(f"95th percentile: {stats['95th_percentile']:.2f}")
        logger.info(f"Risk threshold: {MONITORING_CONFIG['critical_metrics'].get(feature, 'N/A')}")
    
    # Test loan characteristics monitoring
    logger.info("\nTesting loan characteristics monitoring...")
    loan_metrics = ['loan_amount', 'loan_balance', 'loan_term', 'loan_age']
    for feature in loan_metrics:
        stats = monitor.get_feature_statistics(feature)
        logger.info(f"\n{feature.replace('_', ' ').title()} Analysis:")
        logger.info(f"Mean: ${stats['mean']:,.2f}" if 'amount' in feature or 'balance' in feature
                   else f"Mean: {stats['mean']:.1f} months")
        logger.info(f"Distribution: {stats['distribution_type']}")
    
    # Test drift detection
    logger.info("\nTesting drift detection...")
    sample = monitor.bq_reference.get_reference_sample(sample_size=100)
    
    # Simulate drift in rate spread
    sample['rate_spread'] = sample['rate_spread'] * 1.5  # Simulate larger rate spreads
    drift_results = monitor.add_production_data(sample)
    
    logger.info("\nDrift Detection Results:")
    for feature, stats in drift_results.items():
        if stats.is_significant:
            logger.warning(f"{feature}: Significant drift detected")
            logger.warning(f"  KL divergence: {stats.kl_divergence:.4f}")
            logger.warning(f"  p-value: {stats.p_value:.4f}")

def test_critical_thresholds():
    """Test monitoring of critical refinance metrics."""
    logger.info("\nTesting critical threshold monitoring...")
    
    critical_metrics = MONITORING_CONFIG['critical_metrics']
    for metric, threshold in critical_metrics.items():
        logger.info(f"\nTesting {metric} threshold monitoring:")
        logger.info(f"Critical threshold: {threshold}")
        
        # Get current statistics
        monitor = DistributionMonitor(
            window_size=MONITORING_CONFIG['window_size'],
            bigquery_config=BIGQUERY_CONFIG
        )
        stats = monitor.get_feature_statistics(metric)
        
        # Check if we're near any thresholds
        if metric == 'rate_spread':
            if stats['mean'] < threshold:
                logger.warning(f"Average rate spread ({stats['mean']:.2f}) below threshold ({threshold})")
        elif metric == 'ltv_ratio':
            if stats['mean'] > (1 - threshold):
                logger.warning(f"Average LTV ratio ({stats['mean']:.2f}) approaching limit")
        elif metric == 'credit_score':
            if stats['std'] > threshold:
                logger.warning(f"Credit score volatility ({stats['std']:.2f}) above threshold")

if __name__ == "__main__":
    logger.info("Starting refinance model monitoring tests...")
    test_refinance_monitoring()
    test_critical_thresholds()
    logger.info("\nAll tests completed.")
