import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from google.cloud import bigquery
import logging

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

def fetch_data_from_bigquery():
    client = bigquery.Client()
    
    # Construct feature columns string
    feature_cols = ', '.join(BIGQUERY_CONFIG['feature_columns'])
    target_col = BIGQUERY_CONFIG['target_column']
    
    query = f"""
    SELECT
        {feature_cols},
        {target_col}
    FROM
        `{BIGQUERY_CONFIG['project_id']}.{BIGQUERY_CONFIG['dataset_id']}.{BIGQUERY_CONFIG['table_id']}`
    """
    
    logger.info("Fetching data from BigQuery...")
    df = client.query(query).to_dataframe()
    logger.info(f"Fetched {len(df)} rows from BigQuery")
    return df

# Fetch data from BigQuery
df = fetch_data_from_bigquery()

# Split features and target
X = df[BIGQUERY_CONFIG['feature_columns']]
y = df[BIGQUERY_CONFIG['target_column']]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
logger.info("Training model...")
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Performance:")
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': BIGQUERY_CONFIG['feature_columns'],
    'importance': model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)

# Save the trained model
logger.info("Saving model...")
joblib.dump(model, "model.pkl")
print("\nModel saved as model.pkl")
