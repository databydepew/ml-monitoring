FROM python:3.10-slim

WORKDIR /app

# Install required packages
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy model and application files
COPY app/app.py .
COPY model.pkl .

# Expose ports for Flask and Prometheus metrics
EXPOSE 5000
EXPOSE 8001

# Run the Flask application
CMD ["python", "app.py"]
