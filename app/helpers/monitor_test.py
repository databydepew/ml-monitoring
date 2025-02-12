"""Test script for BigQuery-based distribution monitoring."""

from distribution_monitor import DistributionMonitor
from config import BIGQUERY_CONFIG, MONITORING_CONFIG, FEATURE_GROUPS
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_bigquery_monitoring():
    """Test the BigQuery-based monitoring system."""
    
    # Initialize monitor with BigQuery configuration
    monitor = DistributionMonitor(
        window_size=MONITORING_CONFIG['window_size'],
        bigquery_config=BIGQUERY_CONFIG
    )
    
    # Test reference distribution retrieval
    logger.info("Testing reference distribution retrieval...")
    for feature in BIGQUERY_CONFIG['feature_columns'][:2]:  # Test first two features
        dist = monitor.get_reference_distribution(feature)
        logger.info(f"Retrieved distribution for {feature}: shape={dist.shape}")
        
        # Get feature statistics
        stats = monitor.get_feature_statistics(feature)
        logger.info(f"Statistics for {feature}:")
        for stat, value in stats.items():
            logger.info(f"  {stat}: {value}")
    
    # Test getting a sample of reference data
    logger.info("\nTesting reference data sampling...")
    sample = monitor.bq_reference.get_reference_sample(sample_size=100)
    logger.info(f"Retrieved sample data: shape={sample.shape}")
    
    # Test monitoring by feature groups
    logger.info("\nTesting monitoring by feature groups...")
    for group_name, features in FEATURE_GROUPS.items():
        logger.info(f"\nAnalyzing {group_name}:")
        for feature in features:
            try:
                stats = monitor.get_feature_statistics(feature)
                logger.info(f"{feature}:")
                logger.info(f"  Mean: {stats['mean']:.2f}")
                logger.info(f"  Std: {stats['std']:.2f}")
            except Exception as e:
                logger.error(f"Error analyzing {feature}: {str(e)}")

if __name__ == "__main__":
    test_bigquery_monitoring()
