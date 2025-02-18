from lumibot.brokers import Broker
from datetime import datetime, timedelta
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.instruments as instruments
from oandapyV20 import API
import pandas as pd
import logging
import pytz  # Add this import
from queue import Queue  # Add this import at the top
from lumibot.entities import Position  # Add this import
from collections import deque  # Add this import

class OandaDataSource:
    def __init__(self, broker):
        self.broker = broker
        
    def get_datetime(self, adjust_for_delay=True):
        # Return timezone-aware datetime
        return datetime.now(pytz.UTC)
        
    def get_last_price(self, symbol):
        return self.broker.get_last_price(symbol)
        
    def get_yesterday_dividends(self, assets, quote=None):
        """Forex doesn't have dividends, return empty dict"""
        return {asset: 0.0 for asset in assets}

class FilledPositions:
    def __init__(self):
        self._positions = []

    def append(self, position):
        self._positions.append(position)

    def get_list(self):
        return self._positions

    def __len__(self):
        return len(self._positions)

    def __getitem__(self, index):
        return self._positions[index]

    def __setitem__(self, index, value):  # Add this method for item assignment
        if isinstance(index, int) and index < len(self._positions):
            self._positions[index] = value
        else:
            self._positions.append(value)

    def __iter__(self):
        return iter(self._positions)

    def clear(self):
        self._positions.clear()

    def get_dict(self):  # Add this helper method
        return {pos.symbol: pos for pos in self._positions}

class OandaTrader:
    def __init__(self, credentials):
        self.api = API(access_token=credentials["ACCESS_TOKEN"])
        self.account_id = credentials["ACCOUNT_ID"]
        self.name = "OANDA"
        self.IS_PAPER_TRADING = credentials.get("ENVIRONMENT") == "practice"
        self._cash = 10000
        self._positions = {}
        self._filled_positions = FilledPositions()
        self._subscribers = []
        self.market = "24/5"
        self._continue = True
        self.executor = None
        self.extended_trading_minutes = 0
        self.logger = logging.getLogger(__name__)  # Add logger
        self._orders_queue = Queue()  # Add this line
        self._orders = {}  # Add this line for tracking orders
        self._held_trades = {}  # Add this line
        self._price_history = {
            'EUR_USD': deque(maxlen=100),  # Store last 100 price points
            'BTC_USD': deque(maxlen=100)
        }
        self._last_prices = {
            'EUR_USD': None,
            'BTC_USD': None
        }
        
    def set_strategy_name(self, name):
        self._strategy_name = name
        
    def _add_subscriber(self, subscriber):
        self._subscribers.append(subscriber)
        
    def _remove_subscriber(self, subscriber):
        if subscriber in self._subscribers:
            self._subscribers.remove(subscriber)
            
    def get_last_price(self, symbol, quote=None, exchange=None):
        try:
            params = {"instruments": symbol}
            r = pricing.PricingInfo(accountID=self.account_id, params=params)
            response = self.api.request(r)
            
            if not response or 'prices' not in response or not response['prices']:
                return None

            price = float(response['prices'][0]['closeoutAsk'])
            
            # Store price in history with timestamp
            if symbol in self._price_history:
                self._price_history[symbol].append({
                    'price': price,
                    'time': self.get_time().isoformat()
                })
                self._last_prices[symbol] = price

            return price
        except Exception as e:
            logging.error(f"Error getting price for {symbol}: {e}")
            return None

    def is_market_open(self, symbol=None):
        # Crypto markets are always open
        if symbol and "BTC" in symbol:
            return True
            
        # Existing forex market hours logic
        current_time = self.get_time()
        current_weekday = current_time.weekday()
        
        if current_weekday == 4 and current_time.hour >= 17:  # Friday after 5 PM
            return False
        if current_weekday == 5:  # Saturday
            return False
        if current_weekday == 6 and current_time.hour < 17:  # Sunday before 5 PM
            return False
        return True
        
    def get_next_market_open(self):
        return datetime.now() + pd.Timedelta(hours=1)
        
    def _get_balances_at_broker(self, *args, **kwargs):
        try:
            r = accounts.AccountSummary(self.account_id)
            response = self.api.request(r)
            
            cash = float(response['account']['balance'])
            positions_value = float(response['account']['unrealizedPL'])
            total_value = cash + positions_value
            
            return cash, positions_value, total_value
        except Exception as e:
            print(f"Error getting balances: {e}")
            return self._cash, 0, self._cash
            
    def get_tracked_positions(self, strategy_name=None):
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

    def get_time(self):
        # Return timezone-aware datetime
        return datetime.now(pytz.UTC)
        
    def should_continue(self):
        return self._continue
        
    def stop(self):
        self._continue = False 

    def _get_stream_object(self):
        # Not implemented for OANDA
        return None

    def _modify_order(self, order):
        # Not implemented for OANDA
        return None

    def _parse_broker_order(self, response):
        # Not implemented for OANDA
        return None

    def _pull_broker_all_orders(self):
        # Return empty list as we're not tracking orders
        return []

    def _pull_broker_order(self, order_id):
        # Not implemented for OANDA
        return None

    def _pull_position(self, symbol):
        try:
            positions = self.get_tracked_positions()
            return positions.get(symbol)
        except Exception as e:
            logging.error(f"Error pulling position for {symbol}: {e}")
            return None

    def _pull_positions(self, strategy=None, **kwargs):
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

    def _register_stream_events(self):
        # Not implemented for OANDA
        pass

    def _run_stream(self):
        # Not implemented for OANDA
        pass

    def _submit_order(self, order):
        """Required implementation of abstract method"""
        try:
            logging.info(f"Submitting order: {order}")
            
            # Create OANDA order data
            order_data = {
                "order": {
                    "type": "MARKET",
                    "instrument": order['symbol'],
                    "units": str(order['quantity']) if order['side'] == 'buy' else str(-order['quantity']),
                    "timeInForce": "FOK",
                    "positionFill": "DEFAULT"
                }
            }
            
            logging.info(f"Order data: {order_data}")
            
            # Submit order to OANDA
            r = orders.OrderCreate(accountID=self.account_id, data=order_data)
            response = self.api.request(r)
            
            # Detailed logging
            logging.info(f"Full OANDA response: {response}")
            
            if 'errorMessage' in response:
                logging.error(f"OANDA error: {response['errorMessage']}")
                return None
            
            if not response or 'orderFillTransaction' not in response:
                logging.error(f"Invalid order response structure: {response}")
                return None
            
            # Log successful order
            logging.info(f"Order filled at price: {response['orderFillTransaction']['price']}")
            
            # Create and track position
            position = Position(
                strategy=order['strategy'],
                symbol=order['symbol'],
                quantity=order['quantity'],
                entry_price=float(response['orderFillTransaction']['price'])
            )
            self._filled_positions.append(position)
            
            return response['orderCreateTransaction']['id']
        except Exception as e:
            logging.error(f"Error submitting order: {e}", exc_info=True)
            return None

    def submit_order(self, order):
        """Public method that calls the private _submit_order"""
        return self._submit_order(order)

    def cancel_order(self, order_id):
        # Not implemented for OANDA
        pass

    def get_historical_account_value(self, timeframe="1D"):
        # Return current value as we don't track historical values
        cash, positions_value, total_value = self._get_balances_at_broker()
        return [{"timestamp": self.get_time(), "value": total_value}]

    def get_tracked_position(self, strategy_name, symbol):
        """Get a specific position for a strategy"""
        try:
            positions = self.get_tracked_positions(strategy_name)
            return positions.get(symbol)
        except Exception as e:
            logging.error(f"Error getting tracked position for {symbol}: {e}")
            return None

    def market_close_time(self):
        # Return timezone-aware market close time
        close_time = self.market_hours(close=True)
        if not close_time.tzinfo:
            close_time = pytz.UTC.localize(close_time)
        return close_time

    def market_hours(self, close=False):
        """Get market hours"""
        current_time = self.get_time()
        # For 24/5 forex market
        if current_time.weekday() < 4:  # Monday to Thursday
            if close:
                close_time = current_time.replace(hour=17, minute=0, second=0, microsecond=0)
                return pytz.UTC.localize(close_time) if not close_time.tzinfo else close_time
            next_open = current_time.replace(hour=17, minute=0, second=0, microsecond=0) + timedelta(days=1)
            return pytz.UTC.localize(next_open) if not next_open.tzinfo else next_open
        elif current_time.weekday() == 4:  # Friday
            if close:
                close_time = current_time.replace(hour=17, minute=0, second=0, microsecond=0)
                return pytz.UTC.localize(close_time) if not close_time.tzinfo else close_time
            next_open = current_time.replace(hour=17, minute=0, second=0, microsecond=0) + timedelta(days=3)
            return pytz.UTC.localize(next_open) if not next_open.tzinfo else next_open
        else:  # Weekend
            next_market = current_time.replace(hour=17, minute=0, second=0, microsecond=0)
            while next_market.weekday() > 4:
                next_market += timedelta(days=1)
            return pytz.UTC.localize(next_market) if not next_market.tzinfo else next_market

    def get_time_to_close(self):
        """Get time until market closes"""
        if self.market == "24/5":
            current_time = self.get_time()
            close_time = self.market_hours(close=True)
            if current_time > close_time:
                close_time += timedelta(days=1)
            return (close_time - current_time).total_seconds()
        return float('inf')  # For 24/7 markets like crypto 

    def _process_order_queue(self):
        """Process any pending orders in the queue"""
        while not self._orders_queue.empty():
            order = self._orders_queue.get()
            self._submit_order(order)

    def submit_order(self, order):
        """Add order to queue"""
        self._orders_queue.put(order)
        self._process_order_queue() 

    # Add these required abstract methods
    def _create_stream_object(self):
        """Required abstract method"""
        return None

    def _process_stream_object(self, stream_object):
        """Required abstract method"""
        return None

    def _process_user_message(self, message):
        """Required abstract method"""
        return None

    def _run_stream(self):
        """Required abstract method"""
        pass

    def _start_stream(self):
        """Required abstract method"""
        pass

    def _stop_stream(self):
        """Required abstract method"""
        pass

    def get_historical_prices(self, symbol, length, timestep="1m"):
        """Required abstract method"""
        try:
            params = {
                "count": length,
                "granularity": timestep.upper()
            }
            r = instruments.InstrumentsCandles(instrument=symbol, params=params)
            response = self.api.request(r)
            return response.get('candles', [])
        except Exception as e:
            logging.error(f"Error getting historical prices: {e}")
            return []

    def get_position(self, symbol):
        """Required abstract method"""
        try:
            positions = self.get_tracked_positions()
            return positions.get(symbol)
        except Exception as e:
            logging.error(f"Error getting position: {e}")
            return None

    def get_price_action(self, symbol):
        """Get recent price history and movement indicators"""
        if symbol not in self._price_history or not self._price_history[symbol]:
            return None

        prices = list(self._price_history[symbol])
        current_price = self._last_prices[symbol]
        
        if not prices or current_price is None:
            return None

        # Calculate price movement
        price_change = 0
        price_direction = 'neutral'
        if len(prices) > 1:
            price_change = ((current_price - prices[0]['price']) / prices[0]['price']) * 100
            price_direction = 'up' if price_change > 0 else 'down' if price_change < 0 else 'neutral'

        return {
            'current': current_price,
            'history': prices,
            'change_percent': round(price_change, 3),
            'direction': price_direction
        } 