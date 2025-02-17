import os
from dotenv import load_dotenv

# Load environment variables once
load_dotenv()

OANDA_CREDS = {
    "ACCESS_TOKEN": os.getenv("OANDA_ACCESS_TOKEN"),
    "ACCOUNT_ID": os.getenv("OANDA_ACCOUNT_ID"),
    "ENVIRONMENT": os.getenv("OANDA_ENVIRONMENT", "practice")
} 