# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies (eventlet added for production sockets)
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir eventlet

# Copy application code
COPY src/ ./src/
COPY . .

# Create necessary directories
RUN mkdir -p src/models/saved src/database

# Expose port (match your main.py port = 5001)
EXPOSE 5001

# Set environment variables
ENV FLASK_APP=src/main.py
ENV FLASK_ENV=production
ENV PYTHONPATH=/app

# Run the application with eventlet
CMD ["python", "src/main.py"]
