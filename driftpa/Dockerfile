FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full driftpa package
COPY . .

# Ensure /app is in Python path so 'driftpa' package is importable
ENV PYTHONPATH=/app

# OpenEnv HTTP server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
