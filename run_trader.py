from tradingbot import MultiBBStrategy, OANDA_CREDS, OandaTrader
from lumibot.traders import Trader
import logging

def main():
    try:
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        
        # Create your strategy instance
        strategy = MultiBBStrategy(
            name='mlstrat',
            broker=OandaTrader(OANDA_CREDS),
            parameters={
                "symbols": {
                    "forex": ["EUR_USD"]  # Start with just one pair for testing
                },
                "cash_at_risk": 0.1,  # Risk 10% per trade
                "bb_length": 20,
                "bb_std": 2.0
            }
        )

        # Create and run the trader
        trader = Trader()
        trader.add_strategy(strategy)
        trader.run_all()
        
    except Exception as e:
        logging.error(f"Error running trader: {e}")
        raise

# 5. This ensures the code only runs when the file is executed directly
if __name__ == "__main__":
    main() 