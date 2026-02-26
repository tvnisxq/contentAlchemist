# Use the official Python 3.10 slim image
FROM python:3.10-slim

# Install system dependencies required for OpenCV, FFmpeg, and yt-dlp
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port that the Flask app will run on (Render uses 10000 by default, 5000 is our internal)
EXPOSE 5000

# Command to run the application using Gunicorn (production WSGI server)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "600", "src.app:app"]
