# FinRL Trading Platform Dockerfile
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# Fix: second PYTHONPATH assignment was overwriting the first - combined into one
ENV PYTHONPATH=/app:/app/src

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    sqlite3 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY setup.py .
COPY README.md .
COPY .env* ./

# Install the package in development mode
RUN pip install -e .

# Create necessary directories
# Note: added a models/ dir for saving trained RL model checkpoints locally
RUN mkdir -p data logs models

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.path.insert(0, '/app/src'); import config; print('Health check passed')" || exit 1

# Expose port for web interface
EXPOSE 8501

# Default command - use the main CLI
CMD ["python", "src/main.py", "dashboard"]
