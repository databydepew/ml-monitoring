import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Define feature names (must match `app.py`)
feature_names = ["age", "income", "loan_amount", "loan_term", "credit_score", "employment_status", "loan_purpose"]

# Generate synthetic training data
np.random.seed(42)
num_samples = 5000

data = {
    "age": np.random.randint(20, 70, num_samples),
    "income": np.random.randint(20000, 100000, num_samples),
    "loan_amount": np.random.randint(5000, 50000, num_samples),
    "loan_term": np.random.choice([12, 24, 36, 48, 60], num_samples),
    "credit_score": np.random.randint(300, 850, num_samples),
    "employment_status": np.random.choice([0, 1], num_samples),  # 0 = Unemployed, 1 = Employed
    "loan_purpose": np.random.choice([0, 1, 2, 3], num_samples),  # Example: 0=Personal, 1=Business, etc.
}

# Create target variable (1 = Approved, 0 = Rejected)
target = (data["credit_score"] > 600) & (data["income"] > 30000) & (data["loan_amount"] < 40000)
data["approval_status"] = target.astype(int)

# Convert to DataFrame
df = pd.DataFrame(data)

# Split data into train and test sets
X = df[feature_names]
y = df["approval_status"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the trained model
joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")
