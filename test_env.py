from dotenv import load_dotenv
import os

load_dotenv()

print("Environment variables:")
print(f"ACCESS_TOKEN exists: {'OANDA_ACCESS_TOKEN' in os.environ}")
print(f"ACCOUNT_ID: {os.getenv('OANDA_ACCOUNT_ID')}")
print(f"ENVIRONMENT: {os.getenv('OANDA_ENVIRONMENT')}") 