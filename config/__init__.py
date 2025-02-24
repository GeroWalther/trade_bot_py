# Empty file to make config a package 
from .oanda_config import OANDA_CREDS
from .api_config import validate_api_keys
import os
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Add NEWS_API_KEY here
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
if not NEWS_API_KEY:
    logger.error("NEWS_API_KEY not found in environment variables")
    raise ValueError("NEWS_API_KEY not found in environment variables")

# FRED API key
FRED_API_KEY = os.getenv('FRED_API_KEY')

# Alpha Vantage API key
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')

__all__ = ['OANDA_CREDS', 'validate_api_keys'] 