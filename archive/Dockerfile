FROM python:3.10-slim

WORKDIR /app

# Install required packages
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application files
COPY app/ ./app/
COPY model.pkl .
COPY test_bigquery.py .
COPY test_bigquery_write.py .

# Create directory for credentials
RUN mkdir -p /root/.config/gcloud

# Expose ports for Flask and Prometheus metrics
EXPOSE 5000
EXPOSE 8001

# Set Python path and run the Flask application
ENV PYTHONPATH=/app
CMD ["python", "-m", "app.app"]
