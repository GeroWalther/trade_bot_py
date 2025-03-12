# Use an official Python image
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

# Install TA-Lib from an alternative source
RUN wget -q -O ta-lib-0.4.0-src.tar.gz "https://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz?download" \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && ls -l ta-lib-0.4.0 \
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

# Default command
CMD ["bash"]
