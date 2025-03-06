FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y     build-essential     python3-dev     && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:/Users/gerowalther/.console-ninja/.bin:/Library/Frameworks/Python.framework/Versions/3.12/bin:/opt/homebrew/bin:/opt/homebrew/sbin:/Users/gerowalther/.nvm/versions/node/v22.8.0/bin:/Library/Frameworks/Python.framework/Versions/3.9/bin:/usr/local/bin:/System/Cryptexes/App/usr/bin:/usr/bin:/bin:/usr/sbin:/sbin:/var/run/com.apple.security.cryptexd/codex.system/bootstrap/usr/local/bin:/var/run/com.apple.security.cryptexd/codex.system/bootstrap/usr/bin:/var/run/com.apple.security.cryptexd/codex.system/bootstrap/usr/appleinternal/bin:/Library/Apple/usr/bin:/opt/anaconda3/bin:/opt/anaconda3/condabin:/Users/gerowalther/.console-ninja/.bin:/Library/Frameworks/Python.framework/Versions/3.12/bin:/Users/gerowalther/Library/Android/sdk/emulator:/Users/gerowalther/Library/Android/sdk/platform-tools:/Users/gerowalther/Library/Android/sdk/tools:/Users/gerowalther/Library/Android/sdk/tools/bin:/Users/gerowalther/.lmstudio/bin:/Users/gerowalther/.rvm/bin:/Users/gerowalther/.lmstudio/bin"

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Command to run the application
CMD ["python", "trading_server.py"]
