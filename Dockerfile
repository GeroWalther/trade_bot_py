# Use Python 3.10 base image
FROM python:3.10

# Install system dependencies (including TA-Lib dependencies)
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    wget \
    libatlas-base-dev \
    automake \
    autoconf \
    && rm -rf /var/lib/apt/lists/*

# Download and install TA-Lib
RUN wget https://github.com/TA-Lib/ta-lib/releases/download/v0.4.0/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && ./configure && make && \
    make install

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose the necessary ports
EXPOSE 5002 5005

# Default command (overridden by docker-compose)
CMD ["bash"]
