FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy application files
COPY requirements.txt .
COPY static/ ./static/
COPY utils/ ./utils/
COPY *.py ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create directories that the application needs
RUN mkdir -p /app/api/data/prepared_data /app/api/data/metrics /app/api/data/models

EXPOSE 8000

# Run as root to ensure we have write permissions on mounted volumes
# Note: This is not ideal for production but works for development
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]