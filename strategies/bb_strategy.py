from datetime import datetime
import logging
import time

logger = logging.getLogger(__name__)

class BBStrategy:
    def __init__(self, broker, parameters=None):
        self.broker = broker
        self.available_instruments = [
            'EUR_USD', 'GBP_USD', 'USD_JPY', 'AUD_USD', 
            'USD_CAD', 'BTC_USD', 'SPX500_USD', 'NAS100_USD', 
            'XAU_USD', 'BCO_USD'
        ]
        # Default parameters
        default_params = {
            'symbol': 'BTC_USD',
            'quantity': 0.1,
            'bb_length': 20,
            'bb_std': 2.0,
            'check_interval': 240,
            'continue_after_trade': True,
            'max_concurrent_trades': 1
        }
        
        if parameters:
            default_params.update(parameters)
        self.parameters = default_params
        self._continue = False
        self.status_updates = []
        self.trades_history = []
        self.performance = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit_loss': 0,
            'largest_win': 0,
            'largest_loss': 0
        }
        self.current_trade = None

    def get_status(self):
        """Get current strategy status"""
        strategy_info = {
            'name': 'Bollinger Bands Strategy',
            'period': '20 Period BB',
            'description': 'Uses Bollinger Bands (20-period SMA with 2 standard deviations) to identify overbought and oversold conditions:',
            'rules': [
                'Enter LONG when price touches lower band (oversold)',
                'Enter SHORT when price touches upper band (overbought)',
                'Exit LONG when price crosses middle band (SMA)',
                'Exit SHORT when price crosses middle band (SMA)',
                'Uses 2 standard deviation bands for trade signals'
            ]
        }
        
        return {
            'name': 'Bollinger Bands Strategy',
            'running': self._continue,
            'parameters': self.parameters,
            'performance': self.performance,
            'recent_updates': self.status_updates,
            'available_instruments': self.available_instruments,
            'current_trade': self.current_trade,
            'strategy_info': strategy_info
        }

    def should_continue(self):
        """Check if strategy should continue running"""
        return self._continue

    def start(self):
        """Start the strategy"""
        self._continue = True
        self.log_status("🟢 Strategy started")
        # Start the trading loop
        self.run()

    def run(self):
        """Main strategy loop"""
        self.log_status(
            f"🟢 Strategy activated:\n"
            f"   Symbol: {self.parameters['symbol']}\n"
            f"   Quantity: {self.parameters['quantity']}\n"
            f"   Check Interval: {self.parameters['check_interval']}s\n"
            f"   Continue After Trade: {'Yes' if self.parameters['continue_after_trade'] else 'No'}\n"
            f"   Max Concurrent Trades: {self.parameters['max_concurrent_trades']}"
        )
        
        while self._continue:
            try:
                self.check_and_trade()
                time.sleep(self.parameters['check_interval'])
            except Exception as e:
                self.log_status(f"❌ Error in strategy loop: {e}")
                time.sleep(5)

    def check_and_trade(self):
        """Main trading logic"""
        try:
            # Get current positions for this strategy
            current_positions = self.broker.get_tracked_positions()
            symbol = self.parameters.get('symbol')  # Use get() to avoid KeyError
            if not symbol:
                self.log_status("❌ No trading symbol specified")
                return

            symbol_positions = [pos for pos in current_positions.values() 
                              if pos['symbol'] == symbol]
            
            # Check if we can trade based on max concurrent trades
            if len(symbol_positions) >= self.parameters['max_concurrent_trades']:
                self.log_status(f"Maximum trades reached ({len(symbol_positions)}). Waiting for positions to close.")
                return

            # Get price action
            price_action = self.get_price_action(symbol)
            if not price_action:
                return

            # ... rest of trading logic ...

        except Exception as e:
            self.log_status(f"❌ Error in trading logic: {str(e)}")
            logger.error(f"Trading error: {e}", exc_info=True)

    def get_price_action(self, symbol=None):
        """Get price action and calculate Bollinger Bands"""
        try:
            symbol = symbol or self.parameters.get('symbol')
            if not symbol:
                logger.error("No trading symbol specified")
                return None

            current_price = self.broker.get_last_price(symbol)
            if not current_price:
                logger.error(f"Could not get current price for {symbol}")
                return None

            # ... calculate Bollinger Bands ...
            return {
                'current_price': current_price,
                'bb_upper': None,  # Add BB calculations
                'bb_lower': None,
                'bb_middle': None
            }

        except Exception as e:
            logger.error(f"Error getting price action: {e}")
            return None

    def stop(self):
        """Stop the strategy"""
        self._continue = False
        self.log_status("🔴 Strategy stopped")

    def log_status(self, message):
        """Add a timestamped status message"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = f"[{timestamp}] {message}"
        logger.info(status)
        self.status_updates.append(status)
        # Keep only last 100 updates
        if len(self.status_updates) > 100:
            self.status_updates.pop(0)

    # ... rest of the BB strategy implementation ... 