from lumibot.strategies.strategy import Strategy
from finbert_utils import estimate_sentiment

class MultiBBStrategy(Strategy):
    # Move the strategy class here
    # Copy all the strategy code from tradingbot.py
    # ... (copy the entire MultiBBStrategy class here)
    pass

# Create a global instance that can be shared
strategy = None

def initialize_strategy(oanda_creds):
    global strategy
    from oanda_trader import OandaTrader  # Import here to avoid circular import
    
    if strategy is None:
        strategy = MultiBBStrategy(
            name='mlstrat',
            broker=OandaTrader(oanda_creds),
            parameters={
                "symbols": {
                    "forex": ["EUR_USD"]
                },
                "cash_at_risk": 0.1,
                "bb_length": 20,
                "bb_std": 2.0
            }
        )
    return strategy 