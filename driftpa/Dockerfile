FROM ghcr.io/meta-pytorch/openenv-base:latest

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full driftpa package
COPY . .

# OpenEnv HTTP server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
