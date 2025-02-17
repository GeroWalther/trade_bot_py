from lumibot.strategies.strategy import Strategy
from datetime import datetime, timedelta
from finbert_utils import estimate_sentiment
import numpy as np
from trading_data import store
from oanda_trader import OandaTrader

class MultiBBStrategy(Strategy):
    # Move the entire MultiBBStrategy class implementation here
    # (All the methods: initialize, calculate_bollinger_bands, etc.)
    ...

# Create a global instance that can be shared
strategy = None

def initialize_strategy(oanda_creds):
    global strategy
    
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