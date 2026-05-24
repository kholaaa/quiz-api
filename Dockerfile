FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libzbar0 \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

# Best working version for Railway + Gunicorn
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5000} app:app"]