from google.cloud import bigquery
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_bigquery_connection():
    try:
        # Initialize the BigQuery client
        client = bigquery.Client(project='mdepew-assets')
        logger.info("Successfully created BigQuery client")
        
        # Test listing datasets
        try:
            datasets = list(client.list_datasets())
            logger.info(f"Successfully listed {len(datasets)} datasets:")
            for dataset in datasets:
                logger.info(f"- {dataset.dataset_id}")
        except Exception as e:
            logger.error(f"Error listing datasets: {str(e)}")
            return False
        
        # Test simple query
        try:
            query = """
            SELECT 
                current_timestamp() as timestamp,
                session_user() as user
            """
            query_job = client.query(query)
            results = query_job.result()
            for row in results:
                logger.info(f"Current timestamp: {row.timestamp}")
                logger.info(f"Current user: {row.user}")
        except Exception as e:
            logger.error(f"Error running test query: {str(e)}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error initializing BigQuery client: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_bigquery_connection()
    if success:
        logger.info("✅ BigQuery connection test passed!")
    else:
        logger.error("❌ BigQuery connection test failed!")
