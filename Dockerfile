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

RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
RUN tar -xzf ta-lib-0.4.0-src.tar.gz
RUN cd ta-lib/

RUN make

RUN make install

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
