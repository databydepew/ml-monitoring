# Model Feedback API Documentation

## Overview
The `/feedback` endpoint is a crucial component of the MLOps monitoring system that allows tracking and evaluating model performance in real-time. It accepts actual outcomes and compares them with predictions to calculate various performance metrics.

## Endpoint Specifications
- **URL**: `/feedback`
- **Method**: POST
- **Content-Type**: application/json

## Request Format
```json
{
    "prediction": 0 or 1,  // The model's prediction
    "actual": 0 or 1      // The actual outcome
}
```

## Input Validation
1. Required fields: Both `prediction` and `actual` must be present
2. Value constraints: Both values must be either 0 or 1
3. Returns 400 Bad Request if validation fails

## Metrics Updated
The endpoint updates several Prometheus metrics:

### 1. Confusion Matrix Metrics
- `true_positives_total`: When prediction=1 and actual=1
- `true_negatives_total`: When prediction=0 and actual=0
- `false_positives_total`: When prediction=1 and actual=0
- `false_negatives_total`: When prediction=0 and actual=1

### 2. Performance Metrics
All metrics are automatically calculated based on the confusion matrix:
- `accuracy`: (TP + TN) / (TP + TN + FP + FN)
- `precision`: TP / (TP + FP)
- `recall`: TP / (TP + FN)
- `f1_score`: 2 * (precision * recall) / (precision + recall)

## Response Format
### Success Response (200 OK)
```json
{
    "success": true,
    "metrics": {
        "accuracy": float,
        "precision": float,
        "recall": float,
        "f1_score": float
    }
}
```

### Error Responses
1. **400 Bad Request**
```json
{
    "error": "Missing required fields"
}
```
or
```json
{
    "error": "Values must be 0 or 1"
}
```

2. **500 Internal Server Error**
```json
{
    "error": "Internal server error"
}
```

## Example Usage
```bash
# Example cURL request
curl -X POST http://localhost:5000/feedback \
     -H "Content-Type: application/json" \
     -d '{"prediction": 1, "actual": 1}'
```

## Implementation Details
1. The endpoint uses atomic counters for thread-safe metric updates
2. Performance metrics are calculated only when there's at least one feedback entry
3. All errors are logged for debugging purposes
4. Metrics are persisted using Prometheus and can be visualized in dashboards

## Best Practices
1. Send feedback as soon as actual outcomes are available
2. Monitor the error logs for any validation or processing issues
3. Use the metrics endpoint to track model performance over time
4. Set up alerts for significant drops in performance metrics

## Integration with Monitoring
- All metrics are exposed via the Prometheus metrics endpoint
- Can be used to create dashboards in Grafana
- Supports setting up alerts based on performance thresholds
- Enables tracking model drift and performance degradation

## Use Cases
The feedback system is essential for:
- Continuous model monitoring
- Performance tracking
- Model drift detection
- Quality assurance
- Compliance reporting

## Testing the Endpoint
You can test the endpoint using the following Python script:

```pythona
import requests
import json

def test_feedback():
    url = 'http://localhost:5000/feedback'
    headers = {'Content-Type': 'application/json'}
    
    # Test case 1: Successful feedback
    data = {
        'prediction': 1,
        'actual': 1
    }
    response = requests.post(url, headers=headers, json=data)
    print(f"Test 1 - Success case: {response.status_code}")
    print(response.json())
    
    # Test case 2: Missing fields
    data = {
        'prediction': 1
    }
    response = requests.post(url, headers=headers, json=data)
    print(f"Test 2 - Missing fields: {response.status_code}")
    print(response.json())
    
    # Test case 3: Invalid values
    data = {
        'prediction': 2,
        'actual': 1
    }
    response = requests.post(url, headers=headers, json=data)
    print(f"Test 3 - Invalid values: {response.status_code}")
    print(response.json())

if __name__ == '__main__':
    test_feedback()
```

## Monitoring Dashboard
To visualize the metrics, you can create a Grafana dashboard with the following panels:

1. **Confusion Matrix**
   - Display the counts of TP, TN, FP, FN
   - Use a heatmap visualization

2. **Performance Metrics Over Time**
   - Line charts for accuracy, precision, recall, and F1 score
   - Set thresholds for alerting

3. **Request Volume**
   - Track the number of feedback submissions
   - Monitor for unusual patterns

4. **Error Rate**
   - Track validation and processing errors
   - Alert on high error rates


