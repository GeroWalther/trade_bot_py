# Use Python 3.10 base image
FROM python:3.10

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Install system dependencies (including TA-Lib dependencies)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    wget \
    automake \
    autoconf \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib C library
RUN apt-get update && apt-get install -y ta-lib

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Ensure TA-Lib Python package is installed
RUN pip install --no-cache-dir TA-Lib

# Copy project files
COPY . .

# Expose the necessary ports
EXPOSE 5002 5005

# Default command (overridden by docker-compose)
CMD ["bash"]
