import os
from dotenv import load_dotenv

# Load environment variables once
load_dotenv()

OANDA_CREDS = {
    "ACCESS_TOKEN": os.getenv("OANDA_ACCESS_TOKEN"),  # Your actual OANDA API token
    "ACCOUNT_ID": os.getenv("OANDA_ACCOUNT_ID"),      # Your actual OANDA account ID
    "ENVIRONMENT": "practice"  # or "live" for real trading
} 