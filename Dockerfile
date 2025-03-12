# Use Python 3.10 base image
FROM python:3.10

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    wget \
    automake \
    autoconf \
    && rm -rf /var/lib/apt/lists/*

# Download and install TA-Lib from an alternative source
RUN wget -O ta-lib-0.4.0-src.tar.gz https://github.com/TA-Lib/ta-lib/archive/refs/tags/0.4.0.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && mv ta-lib-0.4.0 ta-lib \
    && cd ta-lib \
    && ./configure --prefix=/usr \
    && make -j$(nproc) \
    && make install \
    && cd .. \
    && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Set working directory
WORKDIR /app

# Copy dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose necessary ports
EXPOSE 5002 5005

# Default command
CMD ["bash"]
