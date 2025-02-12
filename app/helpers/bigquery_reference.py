"""Module for fetching and managing reference data from BigQuery."""

from google.cloud import bigquery
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

class BigQueryReference:
    def __init__(
        self,
        project_id: str,
        dataset_id: str,
        table_id: str,
        feature_columns: List[str],
        target_column: str,
    ):
        """Initialize BigQuery reference data manager.
        
        Args:
            project_id: GCP project ID
            dataset_id: BigQuery dataset ID
            table_id: BigQuery table ID
            feature_columns: List of feature column names
            target_column: Name of target/label column
        """
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = table_id
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.client = bigquery.Client(project=project_id)
        
        # Cache for reference distributions
        self._distribution_cache = {}
        
    def get_reference_distribution(
        self,
        feature_name: str,
        n_bins: int = 10,
        force_refresh: bool = False
    ) -> np.ndarray:
        """Get the reference distribution for a specific feature.
        
        Args:
            feature_name: Name of the feature
            n_bins: Number of bins for histogram
            force_refresh: Whether to force refresh the cache
            
        Returns:
            Normalized histogram counts as distribution
        """
        if not force_refresh and feature_name in self._distribution_cache:
            return self._distribution_cache[feature_name]
            
        query = f"""
        WITH FeatureStats AS (
            SELECT 
                MIN({feature_name}) as min_val,
                MAX({feature_name}) as max_val
            FROM `{self.project_id}.{self.dataset_id}.{self.table_id}`
        )
        SELECT
            ROUND(({feature_name} - min_val) / ((max_val - min_val) / {n_bins})) as bin,
            COUNT(*) as count
        FROM `{self.project_id}.{self.dataset_id}.{self.table_id}`,
            FeatureStats
        GROUP BY bin
        ORDER BY bin
        """
        
        query_job = self.client.query(query)
        results = query_job.result()
        
        # Convert to normalized distribution
        counts = np.zeros(n_bins)
        for row in results:
            bin_idx = min(int(row.bin), n_bins - 1)  # Ensure within bounds
            counts[bin_idx] = row.count
            
        # Normalize
        distribution = counts / counts.sum()
        self._distribution_cache[feature_name] = distribution
        
        return distribution
    
    def get_feature_statistics(self, feature_name: str) -> Dict[str, float]:
        """Get basic statistics for a feature from reference data.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            Dictionary with mean, std, min, max values
        """
        query = f"""
        SELECT
            AVG({feature_name}) as mean,
            STDDEV({feature_name}) as std,
            MIN({feature_name}) as min,
            MAX({feature_name}) as max
        FROM `{self.project_id}.{self.dataset_id}.{self.table_id}`
        """
        
        query_job = self.client.query(query)
        result = next(query_job.result())
        
        return {
            'mean': result.mean,
            'std': result.std,
            'min': result.min,
            'max': result.max
        }
    
    def get_reference_sample(
        self,
        sample_size: int = 1000,
        random_seed: Optional[int] = None
    ) -> pd.DataFrame:
        """Get a random sample from reference data.
        
        Args:
            sample_size: Number of rows to sample
            random_seed: Random seed for reproducibility
            
        Returns:
            DataFrame with sampled reference data
        """
        columns = ", ".join(self.feature_columns + [self.target_column])
        
        # Use a simpler random sampling approach
        query = f"""
        SELECT {columns}
        FROM `{self.project_id}.{self.dataset_id}.{self.table_id}`
        ORDER BY RAND()
        LIMIT {sample_size}
        """
        
        return self.client.query(query).to_dataframe()
    
    def compute_target_distribution(self) -> np.ndarray:
        """Compute distribution of target variable.
        
        Returns:
            Normalized distribution of target variable
        """
        query = f"""
        SELECT {self.target_column}, COUNT(*) as count
        FROM `{self.project_id}.{self.dataset_id}.{self.table_id}`
        GROUP BY {self.target_column}
        ORDER BY {self.target_column}
        """
        
        results = self.client.query(query).result()
        counts = [row.count for row in results]
        return np.array(counts) / sum(counts)
