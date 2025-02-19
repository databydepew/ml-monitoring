import unittest
import json
from app import app  # Import your Flask app

class PredictTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_predict(self):
        # Sample input data
        input_data = {
            "feature1": value1,
            "feature2": value2,
            # Add other features as needed
        }
        
        response = self.app.post('/predict', data=json.dumps(input_data), content_type='application/json')
        
        # Check that the response is 200 OK
        self.assertEqual(response.status_code, 200)
        
        # Check the response data (modify as per your expected output)
        response_data = json.loads(response.data)
        self.assertIn('prediction', response_data)

if __name__ == '__main__':
    unittest.main()
