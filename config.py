import os
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

OANDA_CREDS = {
    'ACCESS_TOKEN': os.getenv('OANDA_ACCESS_TOKEN'),
    'ACCOUNT_ID': os.getenv('OANDA_ACCOUNT_ID'),
    'ENVIRONMENT': 'practice'  # or 'live' for real trading
}

# Validate configuration
if not OANDA_CREDS['ACCESS_TOKEN']:
    logger.error("OANDA access token not found in environment variables")
    raise ValueError("OANDA access token not found")

if not OANDA_CREDS['ACCOUNT_ID']:
    logger.error("OANDA account ID not found in environment variables")
    raise ValueError("OANDA account ID not found")

NEWS_API_KEY = os.getenv('NEWS_API_KEY')
if not NEWS_API_KEY:
    raise ValueError("NEWS_API_KEY not found in environment variables") 