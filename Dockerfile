# Use Python 3.10 base image
FROM python:3.10

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Install system dependencies and prebuilt TA-Lib
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    python3-pip \
    wget \
    automake \
    autoconf \
    libta-lib-dev \
    ta-lib \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose necessary ports
EXPOSE 5002 5005

# Default command
CMD ["bash"]
