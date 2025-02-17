from oandapyV20 import API
import oandapyV20.endpoints.accounts as accounts
from config import OANDA_CREDS
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_environment():
    logger.info("Testing environment variables...")
    missing = []
    for key in ["ACCESS_TOKEN", "ACCOUNT_ID", "ENVIRONMENT"]:
        if not OANDA_CREDS.get(key):
            missing.append(key)
    
    if missing:
        logger.error(f"Missing credentials: {', '.join(missing)}")
        return False
    return True

def test_api_connection():
    logger.info("Testing OANDA API connection...")
    try:
        api = API(access_token=OANDA_CREDS["ACCESS_TOKEN"])
        r = accounts.AccountSummary(accountID=OANDA_CREDS["ACCOUNT_ID"])
        response = api.request(r)
        
        logger.info("Connection successful!")
        logger.info(f"Account Name: {response['account']['alias']}")
        logger.info(f"Currency: {response['account']['currency']}")
        logger.info(f"Balance: {response['account']['balance']}")
        return True
    except Exception as e:
        logger.error(f"API connection failed: {e}")
        return False

def main():
    if not test_environment():
        sys.exit(1)
    if not test_api_connection():
        sys.exit(1)
    logger.info("All tests passed!")

if __name__ == "__main__":
    main() 