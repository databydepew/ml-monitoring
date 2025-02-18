FROM python:3.10-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create model directory and copy model file

# Copy the rest of the application
COPY . .

# Expose the application and metrics ports
EXPOSE 5000 8001 9090

# Run with Flask
CMD ["python", "app.py"]