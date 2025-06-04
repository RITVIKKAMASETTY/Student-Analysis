# Use official Python slim image
FROM python:3.11-slim

# Install system dependencies (including Tesseract)
RUN apt-get update && apt-get install -y tesseract-ocr libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside container
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app code
COPY . .

# Expose port 5001 (or the port your app uses)
EXPOSE 5001

# Command to run your app
CMD ["python", "app.py"]
