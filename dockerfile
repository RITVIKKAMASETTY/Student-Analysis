# Use official slim Python image
FROM python:3.11-slim

# Install system packages including Tesseract
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside container
WORKDIR /app

# Copy dependency list and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app code
COPY . .

# Expose the port your Flask app will run on
EXPOSE 10000

# Start the Flask app
CMD ["python", "app.py"]
