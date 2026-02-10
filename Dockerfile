# Sensitivity-Optimized Gas Sensing
# Complete Reproducibility Environment

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV GAS_SENSING_DATA=/app/data

# Expose port for Jupyter
EXPOSE 8888

# Default command
CMD ["python", "run_scientific_pipeline.py"]
