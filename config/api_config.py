import os
from typing import Dict
import logging

logger = logging.getLogger(__name__)

# Required API keys for core functionality
REQUIRED_KEYS = {
    'OANDA_ACCESS_TOKEN': 'OANDA trading access',
    'OANDA_ACCOUNT_ID': 'OANDA account',
    'OPENAI_API_KEY': 'OpenAI analysis',
    'NEWS_API_KEY': 'Market news'
}

# Optional API keys for enhanced features
OPTIONAL_KEYS = {
    # Remove ALPHA_VANTAGE_KEY since we're not using it anymore
}

def validate_api_keys() -> Dict[str, bool]:
    """Validate all required API keys are present"""
    status = {}
    
    # Check required keys
    for key, description in REQUIRED_KEYS.items():
        value = os.getenv(key)
        if not value:
            logger.error(f"Missing {description} API key: {key}")
            status[key] = False
        else:
            logger.info(f"Found {description} API key")
            status[key] = True
    
    # Check optional keys (warn but don't fail)
    for key, description in OPTIONAL_KEYS.items():
        value = os.getenv(key)
        if not value:
            logger.warning(f"Optional {description} API key not found: {key}")
            status[key] = False
        else:
            logger.info(f"Found {description} API key")
            status[key] = True
            
    # Only fail if required keys are missing
    required_status = {k: status[k] for k in REQUIRED_KEYS.keys()}
    if not all(required_status.values()):
        missing_keys = [k for k, v in required_status.items() if not v]
        raise ValueError(f"Missing required API keys: {', '.join(missing_keys)}")
        
    return status 