from oandapyV20 import API
import oandapyV20.endpoints.accounts as accounts
from dotenv import load_dotenv
import os
import sys

def test_oanda_connection():
    # Load environment variables
    load_dotenv()
    
    # Get credentials
    access_token = os.getenv("OANDA_ACCESS_TOKEN")
    account_id = os.getenv("OANDA_ACCOUNT_ID")
    
    print("\n=== OANDA API Connection Test ===")
    print(f"Environment variables loaded:")
    print(f"- Account ID: {account_id}")
    print(f"- Token (first 10 chars): {access_token[:10] if access_token else 'None'}...")
    
    if not access_token or not account_id:
        print("\n❌ ERROR: Missing credentials in .env file")
        print("Please check your .env file contains:")
        print("OANDA_ACCESS_TOKEN=your-token")
        print("OANDA_ACCOUNT_ID=your-account-id")
        print("OANDA_ENVIRONMENT=practice")
        sys.exit(1)
    
    try:
        print("\nAttempting to connect to OANDA API...")
        api = API(access_token=access_token, environment="practice")
        
        print("Attempting to get account details...")
        r = accounts.AccountSummary(accountID=account_id)
        response = api.request(r)
        
        print("\n✅ Success! Account details:")
        print(f"- Account Name: {response['account']['alias']}")
        print(f"- Currency: {response['account']['currency']}")
        print(f"- Balance: {response['account']['balance']}")
        print(f"- Open Positions: {response['account']['openPositionCount']}")
        
    except Exception as e:
        print(f"\n❌ Error connecting to OANDA: {str(e)}")
        print("\nPlease check:")
        print("1. Your internet connection")
        print("2. Your API token is correct")
        print("3. Your account ID is correct")
        print("4. Your account has proper permissions")
        sys.exit(1)

if __name__ == "__main__":
    test_oanda_connection() 