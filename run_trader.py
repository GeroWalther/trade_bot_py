from lumibot.traders import Trader
from trading_state import initialize_strategy
from config import OANDA_CREDS
import logging

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logging.info("Initializing trading strategy...")
    strategy = initialize_strategy(OANDA_CREDS)
    
    logging.info("Setting up trader...")
    trader = Trader()
    trader.add_strategy(strategy)
    
    logging.info("Starting trading bot...")
    logging.info("Trading symbols: EUR/USD (1D timeframe), BTC/USD (1min timeframe)")
    trader.run_all()

if __name__ == "__main__":
    main() 