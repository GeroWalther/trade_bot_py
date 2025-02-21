import os
from dotenv import load_dotenv

load_dotenv()

OANDA_CREDS = {
    "ACCOUNT_ID": os.getenv('OANDA_ACCOUNT_ID'),
    "ACCESS_TOKEN": os.getenv('OANDA_ACCESS_TOKEN'),
    "ENVIRONMENT": os.getenv('OANDA_ENVIRONMENT', 'practice')  # or 'live'
} 