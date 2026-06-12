# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn

# Download NLTK data
RUN python -c "import nltk; nltk.download('wordnet'); nltk.download('stopwords'); nltk.download('omw-1.4')"

# Copy application files
COPY . .

# Install the local project as a package (replaces -e . from requirements.txt)
RUN pip install .

# Environment Variables (Defaults - override these at runtime if needed)
ENV MLFLOW_TRACKING_URI="http://ec2-3-89-143-153.compute-1.amazonaws.com:5000/"
ENV MLFLOW_RUN_ID="cd2f5c44f89e44019ba5db582a764e4a"
ENV S3_BUCKET_NAME="mlflow-bucket-4545"

EXPOSE 8080

# Run with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "flask_app.main:app"]
