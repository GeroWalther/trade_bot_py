# Use Ubuntu 22.04 as the base image
FROM ubuntu:22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Install system dependencies (including TA-Lib build dependencies)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3-venv \
    build-essential libssl-dev libffi-dev python3-dev \
    wget automake autoconf \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib from source
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

# Copy dependencies first to leverage Docker caching
COPY requirements.txt .

# Create a virtual environment and install dependencies
RUN python3 -m venv /opt/venv \
    && . /opt/venv/bin/activate \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir TA-Lib

# Copy project files
COPY . .

# Use virtual environment by default
ENV PATH="/opt/venv/bin:$PATH"

# Expose necessary ports
EXPOSE 5002 5005

# Default command (overridden by docker-compose)
CMD ["bash"]
