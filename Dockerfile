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
RUN wget https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib_0.6.4_amd64.deb && \
    dpkg -i ta-lib_0.6.4_amd64.deb && \
    rm -rf ta-lib_0.6.4_amd64.deb

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
