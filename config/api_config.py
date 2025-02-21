import os
from typing import Dict
import logging

logger = logging.getLogger(__name__)

REQUIRED_API_KEYS = {
    'OANDA_ACCESS_TOKEN': 'OANDA trading access',
    'OANDA_ACCOUNT_ID': 'OANDA account',
    'ANTHROPIC_API_KEY': 'Claude AI analysis',
    'NEWS_API_KEY': 'Market news',
    'FRED_API_KEY': 'Economic indicators',
    'ALPHA_VANTAGE_KEY': 'Market data'
}

def validate_api_keys() -> Dict[str, bool]:
    """Validate all required API keys are present"""
    status = {}
    
    for key, description in REQUIRED_API_KEYS.items():
        value = os.getenv(key)
        if not value:
            logger.error(f"Missing {description} API key: {key}")
            status[key] = False
        else:
            logger.info(f"Found {description} API key")
            status[key] = True
            
    if not all(status.values()):
        missing_keys = [k for k, v in status.items() if not v]
        raise ValueError(f"Missing required API keys: {', '.join(missing_keys)}")
        
    return status 