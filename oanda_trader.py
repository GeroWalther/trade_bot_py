from lumibot.brokers import Broker
from datetime import datetime
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.instruments as instruments
from oandapyV20 import API
import pandas as pd
import logging

class OandaDataSource:
    def __init__(self, broker):
        self.broker = broker
        
    def get_datetime(self, adjust_for_delay=True):
        return datetime.now()
        
    def get_last_price(self, symbol):
        return self.broker.get_last_price(symbol)
        
    def get_yesterday_dividends(self, assets, quote=None):
        """Forex doesn't have dividends, return empty dict"""
        return {asset: 0.0 for asset in assets}

class OandaTrader(Broker):
    IS_BACKTESTING_BROKER = False
    
    def __init__(self, credentials):
        self.api = API(access_token=credentials["ACCESS_TOKEN"])
        self.account_id = credentials["ACCOUNT_ID"]
        self.name = "OANDA"
        self.IS_PAPER_TRADING = credentials.get("ENVIRONMENT") == "practice"
        self._cash = 10000  # Default cash amount
        self._positions = {}
        self._subscribers = []
        self._strategy_name = None
        self.executor = None
        self.market = "24/5"
        self._continue = True
        self.data_source = None  # Will be set by strategy
        
    def set_strategy_name(self, name):
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
        
    def _get_balances_at_broker(self, strategy=None):
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
        return datetime.now()

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

    def _pull_positions(self):
        return self.get_tracked_positions()

    def _register_stream_events(self):
        # Not implemented for OANDA
        pass

    def _run_stream(self):
        # Not implemented for OANDA
        pass

    def _submit_order(self, order):
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
            
            r = orders.OrderCreate(self.account_id, data=data)
            response = self.api.request(r)
            return response['orderFillTransaction']['id']
        except Exception as e:
            logging.error(f"Error submitting order: {e}")
            return None

    def cancel_order(self, order_id):
        # Not implemented for OANDA
        pass

    def get_historical_account_value(self, timeframe="1D"):
        # Return current value as we don't track historical values
        cash, positions_value, total_value = self._get_balances_at_broker()
        return [{"timestamp": self.get_time(), "value": total_value}] 