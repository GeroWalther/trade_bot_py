from lumibot.strategies.strategy import Strategy
from datetime import datetime, timedelta
from finbert_utils import estimate_sentiment
import numpy as np
from trading_data import store
from oanda_trader import OandaTrader
from strategies.bollinger_bands import BollingerBandsStrategy
from config import OANDA_CREDS
import logging

logger = logging.getLogger(__name__)

class MultiBBStrategy(Strategy):
    def initialize(self):
        self.sleeptime = 1  # This means 1 minute
        self.last_trade = {}
        for symbol in self.parameters["symbols"]["forex"] + self.parameters["symbols"]["crypto"]:
            self.last_trade[symbol] = None

    def calculate_bollinger_bands(self, symbol, period=20, std=2):
        # Get historical data
        timeframe = self.parameters["timeframes"][symbol]
        bars = self.get_historical_prices(symbol, timeframe, period)
        if bars is None or len(bars) < period:
            return None, None, None
        
        closes = np.array([bar.close for bar in bars])
        sma = np.mean(closes)
        std_dev = np.std(closes)
        
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        
        return upper_band, sma, lower_band

    def is_market_open(self, symbol):
        return self.broker.is_market_open(symbol)

    def on_trading_iteration(self):
        # Get current account info
        cash, positions_value, total_value = self.broker._get_balances_at_broker()
        
        # Get current market prices with proper error handling
        eur_usd_price = self.get_last_price("EUR_USD", quote="USD")
        btc_usd_price = self.get_last_price("BTC_USD", quote="USD")
        
        # Get current time in UTC
        current_time = self.get_datetime()

        # Update store with current state
        store.update_data({
            'account': {
                'balance': cash,
                'unrealized_pl': positions_value,
                'total_value': total_value,
            },
            'market_prices': {
                'EUR_USD': float(eur_usd_price) if eur_usd_price is not None else None,
                'BTC_USD': float(btc_usd_price) if btc_usd_price is not None else None,
                'last_update': current_time.isoformat() if current_time else None
            },
            'positions': self.get_positions(),
            'market_status': {
                'is_market_open': True,  # Update based on actual market status
                'active_symbols': self.parameters["symbols"]["forex"] + self.parameters["symbols"]["crypto"]
            },
            'trading_stats': {
                'win_rate': 0,  # Update these with actual stats
                'winning_trades': 0,
                'losing_trades': 0
            }
        })
        print("Updated store with data:", store.get_data())

        for symbol in self.parameters["symbols"]["forex"] + self.parameters["symbols"]["crypto"]:
            if not self.is_market_open(symbol):
                continue

            current_price = self.get_last_price(symbol)
            if current_price is None:
                continue

            upper_band, sma, lower_band = self.calculate_bollinger_bands(
                symbol, 
                self.parameters["bb_length"], 
                self.parameters["bb_std"]
            )
            
            if upper_band is None:
                continue

            position = self.get_position(symbol)
            
            # Trading logic
            if position is None:
                if current_price < lower_band:
                    # Buy signal
                    qty = self.calculate_position_size(symbol, self.parameters["cash_at_risk"])
                    if qty > 0:
                        self.buy(symbol, qty)
                        self.last_trade[symbol] = "buy"
                elif current_price > upper_band:
                    # Sell signal
                    qty = self.calculate_position_size(symbol, self.parameters["cash_at_risk"])
                    if qty > 0:
                        self.sell(symbol, qty)
                        self.last_trade[symbol] = "sell"
            else:
                # Exit conditions
                if (self.last_trade[symbol] == "buy" and current_price > sma) or \
                   (self.last_trade[symbol] == "sell" and current_price < sma):
                    self.close_position(position)
                    self.last_trade[symbol] = None

    def calculate_position_size(self, symbol, risk_pct):
        cash = self.get_cash()
        price = self.get_last_price(symbol)
        if price is None:
            return 0
        return (cash * risk_pct) / price

# Initialize strategy with default parameters
strategy = None
broker = None

def initialize_strategy():
    global strategy, broker
    try:
        # Initialize broker if not already done
        if broker is None:
            broker = OandaTrader(OANDA_CREDS)
            
        # Initialize strategy with default parameters
        if strategy is None:
            strategy = BollingerBandsStrategy(
                broker=broker,
                parameters={
                    'bb_length': 20,
                    'bb_std': 2.0,
                    'cash_at_risk': 0.1
                }
            )
            strategy._continue = False  # Start in stopped state
            logger.info("Strategy initialized successfully")
            
        return strategy
    except Exception as e:
        logger.error(f"Error initializing strategy: {e}", exc_info=True)
        return None

# Initialize strategy on module import
initialize_strategy() 