#!/usr/bin/env python3
import joblib
from sklearn.ensemble import RandomForestClassifier
import os

# Create a dummy model
model = RandomForestClassifier(n_estimators=10)

# Create some dummy data
import numpy as np
X = np.random.rand(100, 12)  # 12 features
y = np.random.randint(0, 2, 100)  # Binary target

# Fit the model
model.fit(X, y)

# Create model directory if it doesn't exist
os.makedirs('/app/model', exist_ok=True)

# Save the model
joblib.dump(model, '/app/model/model.pkl')
print("Created dummy model at /app/model/model.pkl")
