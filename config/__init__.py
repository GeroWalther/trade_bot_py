# Empty file to make config a package 
from .oanda_config import OANDA_CREDS
from .api_config import validate_api_keys
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logger
logger = logging.getLogger(__name__)

# OANDA Configuration
OANDA_CREDS = {
    "ACCOUNT_ID": os.getenv('OANDA_ACCOUNT_ID'),
    "ACCESS_TOKEN": os.getenv('OANDA_ACCESS_TOKEN'),
    "ENVIRONMENT": os.getenv('OANDA_ENVIRONMENT', 'practice')  # or 'live'
}

# API Keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
MARKET_NEWS_API_KEY = os.getenv('MARKET_NEWS_API_KEY')

# Add NEWS_API_KEY here
if not NEWS_API_KEY:
    logger.error("NEWS_API_KEY not found in environment variables")
    raise ValueError("NEWS_API_KEY not found in environment variables")

__all__ = ['OANDA_CREDS', 'validate_api_keys'] 