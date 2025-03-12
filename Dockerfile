# Use a minimal Ubuntu base image
FROM ubuntu:22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3-venv \
    build-essential libssl-dev libffi-dev python3-dev \
    wget automake autoconf \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib C library
RUN wget -q https://versaweb.dl.sourceforge.net/project/ta-lib/ta-lib/0.4.0/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib-0.4.0 \
    && ./configure --prefix=/usr \
    && make -j$(nproc) \
    && make install \
    && cd .. \
    && rm -rf ta-lib-0.4.0 ta-lib-0.4.0-src.tar.gz

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose necessary ports
EXPOSE 5002 5005

# Default command for docker-compose
CMD ["bash"]
