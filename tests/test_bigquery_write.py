from google.cloud import bigquery
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_bigquery_write():
    try:
        # Initialize the BigQuery client
        client = bigquery.Client(project='mdepew-assets')
        logger.info("Successfully created BigQuery client")
        
        # Create dataset if it doesn't exist
        dataset_id = 'refinance_data'
        dataset_ref = client.dataset(dataset_id)
        
        try:
            client.get_dataset(dataset_ref)
            logger.info(f"Dataset {dataset_id} already exists")
        except Exception:
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = "US"
            dataset = client.create_dataset(dataset)
            logger.info(f"Created dataset {dataset_id}")
        
        # Create sample refinance data
        sample_data = {
            'interest_rate': np.random.uniform(2.0, 5.0, 10),
            'loan_amount': np.random.randint(100000, 900000, 10),
            'credit_score': np.random.randint(620, 850, 10),
            'approval_status': np.random.choice([0, 1], 10),
            'timestamp': pd.date_range(start='2025-01-01', periods=10)
        }
        df = pd.DataFrame(sample_data)
        
        # Define table schema
        schema = [
            bigquery.SchemaField("interest_rate", "FLOAT"),
            bigquery.SchemaField("loan_amount", "INTEGER"),
            bigquery.SchemaField("credit_score", "INTEGER"),
            bigquery.SchemaField("approval_status", "INTEGER"),
            bigquery.SchemaField("timestamp", "TIMESTAMP")
        ]
        
        # Create table reference
        table_id = 'test_refinance_applications'
        table_ref = dataset_ref.table(table_id)
        
        # Create table with schema
        table = bigquery.Table(table_ref, schema=schema)
        try:
            client.get_table(table)
            logger.info(f"Table {table_id} already exists")
        except Exception:
            table = client.create_table(table)
            logger.info(f"Created table {table_id}")
        
        # Load data into table
        job_config = bigquery.LoadJobConfig()
        job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
        job.result()
        
        logger.info(f"Successfully loaded {len(df)} rows into {dataset_id}.{table_id}")
        
        # Test reading data back
        query = f"""
        SELECT * FROM `{client.project}.{dataset_id}.{table_id}`
        LIMIT 5
        """
        query_job = client.query(query)
        results = query_job.result()
        
        logger.info("Successfully read sample data:")
        for row in results:
            logger.info(f"Interest Rate: {row.interest_rate}, Credit Score: {row.credit_score}")
            
        return True
        
    except Exception as e:
        logger.error(f"Error in BigQuery write test: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_bigquery_write()
    if success:
        logger.info("✅ BigQuery write test passed!")
    else:
        logger.error("❌ BigQuery write test failed!")
