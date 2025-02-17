from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from datetime import datetime, timedelta 
from finbert_utils import estimate_sentiment
import pandas as pd
import numpy as np
from config import OANDA_CREDS  # Import from config instead
from dotenv import load_dotenv
import os
from oandapyV20 import API
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.instruments as instruments
import logging
from oanda_trader import OandaTrader
from trading_data import store

# Load environment variables
load_dotenv()

# At the start of the file, add a debug print
print("OANDA Credentials:", OANDA_CREDS)

class OandaDataSource:
    def __init__(self, broker):
        self.broker = broker
        
    def get_datetime(self, adjust_for_delay=True):
        return datetime.now()
        
    def get_last_price(self, symbol):
        return self.broker.get_last_price(symbol)
        
    def get_yesterday_dividends(self, assets, quote=None):
        """Forex doesn't have dividends, return empty dict"""
        return {asset: 0.0 for asset in assets}  # Return 0 dividends for all assets

class OandaTrader:
    IS_BACKTESTING_BROKER = False
    
    def __init__(self, credentials):
        self.api = API(access_token=credentials["ACCESS_TOKEN"])
        self.account_id = credentials["ACCOUNT_ID"]
        self.name = "OANDA"
        self.IS_BACKTESTING_BROKER = False
        self.IS_PAPER_TRADING = credentials.get("ENVIRONMENT") == "practice"
        self._cash = 10000  # Default cash amount
        self._positions = {}
        self._subscribers = []
        self._strategy_name = None  # Add this for strategy name
        self.executor = None  # Add this
        self.market = "24/5"  # Change this from "forex" to "24/5"
        self._continue = True  # Add this flag
        self.data_source = OandaDataSource(self)  # Add this
        
    def set_strategy_name(self, name):  # Add this method
        self._strategy_name = name
        
    def _add_subscriber(self, subscriber):
        self._subscribers.append(subscriber)
        
    def _remove_subscriber(self, subscriber):
        if subscriber in self._subscribers:
            self._subscribers.remove(subscriber)
            
    def get_last_price(self, symbol):
        try:
            params = {"instruments": symbol}
            r = pricing.PricingInfo(accountID=self.account_id, params=params)
            response = self.api.request(r)
            if not response or 'prices' not in response or not response['prices']:
                logging.error(f"Invalid price response for {symbol}")
                return None
            price = response['prices'][0]
            return float(price['closeoutAsk'])
        except Exception as e:
            logging.error(f"Error getting price for {symbol}: {e}")
            return None
        
    def is_market_open(self, symbol=None):
        """Check if forex market is open"""
        current_time = self.get_time()
        current_weekday = current_time.weekday()
        
        # Forex market is closed from Friday 5 PM EST to Sunday 5 PM EST
        if current_weekday == 4 and current_time.hour >= 17:  # Friday after 5 PM
            return False
        if current_weekday == 5:  # Saturday
            return False
        if current_weekday == 6 and current_time.hour < 17:  # Sunday before 5 PM
            return False
        return True
        
    def get_next_market_open(self):
        # Return next market open time - placeholder
        return datetime.now() + timedelta(hours=1)
        
    def _get_balances_at_broker(self, strategy=None, symbols=None):
        """Get account balance from OANDA"""
        try:
            r = accounts.AccountSummary(self.account_id)
            response = self.api.request(r)
            
            cash = float(response['account']['balance'])
            positions_value = float(response['account']['unrealizedPL'])
            total_value = cash + positions_value
            
            return cash, positions_value, total_value
        except Exception as e:
            print(f"Error getting balances: {e}")
            return self._cash, 0, self._cash  # Return defaults if API fails
        
    def _set_initial_positions(self, strategy):
        # Initialize positions for the strategy
        self._positions = {}  # Store positions in broker
        strategy._cash = self._cash  # Use internal cash attribute
        
    def get_timestamp(self):
        return datetime.now()
        
    def get_account_balance(self):
        return self._cash
        
    def get_position(self, symbol):
        # Implement getting position for symbol
        return self._positions.get(symbol)
        
    def get_positions(self):
        # Implement getting all positions
        return list(self._positions.keys())
        
    def submit_order(self, order):
        """Submit order to OANDA"""
        try:
            data = {
                "order": {
                    "type": "MARKET",
                    "instrument": order.symbol,
                    "units": str(order.quantity) if order.side == "buy" else str(-order.quantity),
                    "timeInForce": "FOK",
                    "positionFill": "DEFAULT"
                }
            }
            
            # Add take profit if specified
            if hasattr(order, 'take_profit_price') and order.take_profit_price:
                data["order"]["takeProfitOnFill"] = {
                    "price": str(order.take_profit_price)
                }
            
            # Add stop loss if specified
            if hasattr(order, 'stop_loss_price') and order.stop_loss_price:
                data["order"]["stopLossOnFill"] = {
                    "price": str(order.stop_loss_price)
                }
                
            r = orders.OrderCreate(self.account_id, data=data)
            response = self.api.request(r)
            return response['orderFillTransaction']['id']
        except Exception as e:
            print(f"Error submitting order: {e}")
            return None
        
    def cancel_order(self, order_id):
        # Implement order cancellation
        pass

    def get_historical_prices(self, symbol, length, timestep="M1"):
        """Get historical candles from OANDA"""
        try:
            params = {
                "count": length,
                "granularity": timestep,
                "price": "M"  # Midpoint data
            }
            r = instruments.InstrumentsCandles(instrument=symbol, params=params)
            response = self.api.request(r)
            
            # Convert to Bar objects that lumibot expects
            bars = []
            for candle in response['candles']:
                bar = type('Bar', (), {
                    'open': float(candle['mid']['o']),
                    'high': float(candle['mid']['h']),
                    'low': float(candle['mid']['l']),
                    'close': float(candle['mid']['c']),
                    'volume': float(candle['volume']),
                    'timestamp': pd.to_datetime(candle['time'])
                })
                bars.append(bar)
            return bars
        except Exception as e:
            print(f"Error fetching historical prices: {e}")
            return None
            
    def set_executor(self, executor):  # Add this required method
        self.executor = executor
        
    def get_tradeable_markets(self):  # Add this required method
        return ["EUR_USD"]  # Return list of tradeable symbols
        
    def get_time(self):  # Add this required method
        return datetime.now()

    def is_backtesting_broker(self):
        return self.IS_BACKTESTING_BROKER

    def should_continue(self):  # Add this method
        """Determine if trading should continue"""
        return self._continue
        
    def stop(self):  # Add this method to allow stopping
        self._continue = False

    def _await_market_to_open(self, timedelta=None, strategy=None):
        """Since forex markets are 24/5, we'll implement a simple check"""
        # For forex, we mainly need to check if it's not weekend
        current_time = self.get_time()
        if current_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
            # Wait until Monday
            next_market_day = current_time + timedelta(days=(7 - current_time.weekday()))
            return next_market_day.replace(hour=0, minute=0, second=0)
        return current_time

    def get_time_to_close(self):
        """Get time until market closes"""
        current_time = self.get_time()
        if current_time.weekday() >= 5:  # Weekend
            return 0
        # For weekdays, forex market is open 24 hours
        return 24 * 60 * 60  # Return seconds until "close" (24 hours)
        
    def get_datetime_range(self, timeshift=None):
        """Get market open/close times"""
        current_time = self.get_time()
        market_open = current_time.replace(hour=0, minute=0, second=0)
        market_close = current_time.replace(hour=23, minute=59, second=59)
        return market_open, market_close

    def get_tracked_positions(self, strategy_name=None):
        """Get tracked positions for a strategy"""
        try:
            r = positions.OpenPositions(accountID=self.account_id)
            response = self.api.request(r)
            
            tracked_positions = {}
            for position in response['positions']:
                instrument = position['instrument']
                tracked_positions[instrument] = {
                    'quantity': float(position['long']['units']) if 'long' in position else float(position['short']['units']),
                    'entry_price': float(position['long']['averagePrice']) if 'long' in position else float(position['short']['averagePrice']),
                    'unrealized_pl': float(position['unrealizedPL']),
                    'value': float(position['long']['units']) * float(position['long']['averagePrice']) if 'long' in position 
                            else float(position['short']['units']) * float(position['short']['averagePrice'])
                }
            return tracked_positions
        except Exception as e:
            print(f"Error getting tracked positions: {e}")
            return {}
            
    def get_tracked_position(self, symbol, strategy_name=None):
        """Get a specific tracked position"""
        positions = self.get_tracked_positions(strategy_name)
        return positions.get(symbol)

    def market_close_time(self):
        """Get market close time for forex"""
        current_time = self.get_time()
        # For forex, market closes Friday at 5pm EST
        if current_time.weekday() == 4:  # Friday
            return current_time.replace(hour=17, minute=0, second=0)
        # For other days, market is open 24 hours
        return current_time.replace(hour=23, minute=59, second=59)
        
    def _await_market_to_close(self, timedelta=None, strategy=None):
        """Wait for market to close - for forex this is mainly for weekends"""
        current_time = self.get_time()
        if current_time.weekday() == 4 and current_time.hour >= 17:  # Friday after 5pm
            # Wait until Monday
            next_market_day = current_time + timedelta(days=3)
            return next_market_day.replace(hour=0, minute=0, second=0)
        return current_time

    def on_trading_iteration(self):
        """Required method for Lumibot"""
        pass
        
    def on_abrupt_closing(self):
        """Required method for Lumibot"""
        self.stop()

    def market_open_time(self):
        """Get market open time for forex"""
        current_time = self.get_time()
        if current_time.weekday() >= 5:  # Weekend
            # If weekend, next open is Monday
            days_to_monday = (7 - current_time.weekday())
            next_open = current_time + timedelta(days=days_to_monday)
            return next_open.replace(hour=0, minute=0, second=0)
        # For weekdays, forex market opens at midnight
        return current_time.replace(hour=0, minute=0, second=0)

class MultiBBStrategy(Strategy):
    def initialize(self, symbols={"forex": ["EUR_USD"]}, cash_at_risk=0.1,
                  bb_length=20, bb_std=2.0):
        self.symbols = symbols
        self.cash_at_risk = cash_at_risk
        self.sleeptime = "1H"
        self.last_trade = {}
        self.bb_length = bb_length
        self.bb_std = bb_std
        
        # Initialize last_trade status for each symbol
        for category in symbols:
            for symbol in symbols[category]:
                self.last_trade[symbol] = None

    def calculate_bollinger_bands(self, symbol, history_bars=30):
        # Get historical data
        history = self.get_historical_prices(symbol, history_bars)
        if history is None or len(history) < self.bb_length:
            return None, None, None
        
        closes = np.array([bar.close for bar in history])
        
        # Calculate BB
        basis = np.mean(closes[-self.bb_length:])
        std = np.std(closes[-self.bb_length:])
        upper = basis + (self.bb_std * std)
        lower = basis - (self.bb_std * std)
        
        return upper, basis, lower

    def position_sizing(self, symbol, current_price):
        cash = self.get_cash() * self.cash_at_risk
        
        # Different sizing for different asset classes
        if "USD" in symbol:  # Forex pairs
            units = cash / current_price
        else:  # Stocks/Indices
            units = (cash / current_price)
        
        return round(units, 2)

    def on_trading_iteration(self):
        """Main trading logic"""
        for category in self.symbols:
            for symbol in self.symbols[category]:
                self.process_symbol(symbol)
                
    def process_symbol(self, symbol):
        """Process individual symbol"""
        # Get current price
        current_price = self.get_last_price(symbol)
        if not current_price:
            return
            
        # Get BB values
        upper, basis, lower = self.calculate_bollinger_bands(symbol)
        if not all([upper, basis, lower]):
            return

        # Get sentiment
        probability, sentiment = self.get_sentiment(symbol)
        
        # Calculate position size
        quantity = self.position_sizing(symbol, current_price)
        
        # Trading logic
        if current_price > upper and sentiment == "positive" and probability > 0.999:
            # Long entry condition
            if self.last_trade.get(symbol) != "long":
                if self.last_trade.get(symbol) == "short":
                    self.sell_all(symbol)
                
                order = self.create_order(
                    symbol,
                    quantity,
                    "buy",
                    type="bracket",
                    take_profit_price=current_price * 1.20,
                    stop_loss_price=current_price * 0.95
                )
                self.submit_order(order)
                self.last_trade[symbol] = "long"
        
        elif current_price < lower and sentiment == "negative" and probability > 0.999:
            # Exit long positions
            if self.last_trade.get(symbol) == "long":
                self.sell_all(symbol)
                self.last_trade[symbol] = None

        # Emit status update
        self.emit_status(symbol)

    def get_sentiment(self, symbol):
        try:
            # Get relevant news based on symbol
            today = datetime.now()
            three_days_prior = today - timedelta(days=3)
            
            # Adjust symbol for news search
            search_term = {
                "EUR_USD": "EUR/USD forex",
                "GBP_USD": "GBP/USD forex",
                "SPX500_USD": "S&P 500",
                "BTC_USD": "Bitcoin"
            }.get(symbol, symbol)
            
            # Placeholder for now - implement proper news fetching
            news = ["Sample news headline"]  # Replace with actual news fetching
            
            probability, sentiment = estimate_sentiment(news)
            return probability, sentiment
        except Exception as e:
            print(f"Error getting sentiment: {e}")
            return 0, "neutral"

    def emit_status(self, symbol):
        # Get current balances
        cash, positions_value, total_value = self.broker._get_balances_at_broker(self)
        
        # Create status update
        status = {
            'cash_available': cash,
            'total_portfolio_value': total_value,
            'daily_return_pct': round((total_value - self._initial_portfolio_value) / 
                                    self._initial_portfolio_value * 100, 2),
            'positions': self.get_positions(),
            'trading_stats': {
                'win_rate': 60,  # Replace with actual stats
                'winning_trades': 6,
                'losing_trades': 4
            },
            'market_status': {
                'is_market_open': self.is_market_open(),
                'active_symbols': list(self.symbols.values())
            }
        }
        
        # Update the store
        store.update_data(status)

# Instead, add this if you want to run backtesting
if __name__ == "__main__":
    from trading_state import initialize_strategy
    
    strategy = initialize_strategy(OANDA_CREDS)
    # ... rest of your main code

# trader = Trader()
# trader.add_strategy(strategy)
# trader.run_all()
