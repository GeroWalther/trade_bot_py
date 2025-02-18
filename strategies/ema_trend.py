import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import threading
import time

logger = logging.getLogger(__name__)

class EMATrendStrategy:
    def __init__(self, broker, parameters=None):
        self.broker = broker
        self.available_instruments = [
            'EUR_USD', 'GBP_USD', 'USD_JPY', 'AUD_USD', 
            'USD_CAD', 'BTC_USD', 'SPX500_USD', 'NAS100_USD', 
            'XAU_USD', 'BCO_USD'
        ]
        self.parameters = parameters or {
            'symbol': 'BTC_USD',  # Default symbol
            'quantity': 0.1,
            'ema_period': 50,
            'check_interval': 240  # 4 minutes
        }
        self._continue = False
        self.last_check_time = None
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
        self.last_analysis = None
        self._thread = None

    def get_price_action(self, symbol=None):
        """Get current price action without historical data"""
        try:
            symbol = symbol or self.parameters['symbol']
            current_price = self.broker.get_last_price(symbol)
            
            if not current_price:
                logger.error(f"Could not get current price for {symbol}")
                return None
                
            # Get the previous price (1 minute ago) for basic trend
            prev_price = self.broker.get_last_price(symbol, minutes_ago=1)
            
            if not prev_price:
                logger.warning(f"Could not get previous price for {symbol}, using current price")
                prev_price = current_price
            
            trend = "UP" if current_price > prev_price else "DOWN"
            
            return {
                'current_price': current_price,
                'prev_price': prev_price,
                'trend': trend
            }
            
        except Exception as e:
            logger.error(f"Error getting price action: {e}")
            return None

    def should_enter_trade(self, symbol=None):
        """Determine if we should enter a trade based on simple price action"""
        try:
            price_action = self.get_price_action(symbol)
            if not price_action:
                return False
                
            # Simple trend following - enter on uptrend
            return price_action['trend'] == "UP"
            
        except Exception as e:
            logger.error(f"Error in should_enter_trade: {e}")
            return False

    def execute_trade(self):
        """Execute trade based on current conditions"""
        try:
            symbol = self.parameters['symbol']
            quantity = self.parameters['quantity']
            
            # Get current price directly from broker
            current_price = self.broker.get_last_price(symbol)
            if not current_price:
                self.log_status(f"❌ Cannot execute trade - No price data for {symbol}")
                return False
            
            # Immediately execute buy order on strategy start
            order = {
                'symbol': symbol,
                'quantity': quantity,
                'side': 'buy'
            }
            
            self.log_status(f"🚀 Initiating BUY order: {quantity} {symbol} @ {current_price:.5f}")
            order_id = self.broker.submit_order(order)
            
            if order_id:
                self.current_trade = {
                    'order_id': order_id,
                    'symbol': symbol,
                    'quantity': quantity,
                    'entry_price': current_price,
                    'entry_time': self.broker.get_time()
                }
                self.log_status(
                    f"✅ Position opened successfully:\n"
                    f"   Symbol: {symbol}\n"
                    f"   Quantity: {quantity}\n"
                    f"   Entry Price: {current_price:.5f}\n"
                    f"   Time: {self.broker.get_time().strftime('%H:%M:%S')}"
                )
                return True
            else:
                self.log_status("❌ Order submission failed")
            
            return False
            
        except Exception as e:
            self.log_status(f"❌ Error executing trade: {e}")
            return False

    def exit_trade(self):
        """Exit current trade"""
        try:
            if not self.current_trade:
                return False
            
            symbol = self.current_trade['symbol']
            quantity = self.current_trade['quantity']
            entry_price = self.current_trade['entry_price']
            
            current_price = self.broker.get_last_price(symbol)
            if not current_price:
                self.log_status("❌ Could not get current price for exit")
                return False
            
            profit_loss = (current_price - entry_price) * quantity
            profit_pct = ((current_price - entry_price) / entry_price) * 100
            
            self.log_status(
                f"🔄 Closing position - Entry: {entry_price:.5f}, "
                f"Exit: {current_price:.5f}, "
                f"P/L: {profit_loss:.2f} ({profit_pct:.2f}%)"
            )
            
            order = {
                'symbol': symbol,
                'quantity': quantity,
                'side': 'sell'
            }
            
            order_id = self.broker.submit_order(order)
            if order_id:
                # Record trade result
                trade_result = {
                    'symbol': symbol,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'quantity': quantity,
                    'profit_loss': profit_loss,
                    'profit_pct': profit_pct,
                    'duration': (self.broker.get_time() - self.current_trade['entry_time']).total_seconds() / 60,
                    'timestamp': self.broker.get_time()
                }
                
                self.update_performance(trade_result)
                self.current_trade = None
                
                self.log_status(
                    f"✅ Position closed - Duration: {trade_result['duration']:.1f} minutes, "
                    f"Total P/L: {self.performance['total_profit_loss']:.2f}"
                )
                return True
            
            self.log_status("❌ Failed to close position")
            return False
            
        except Exception as e:
            self.log_status(f"❌ Error closing trade: {e}")
            return False

    def run(self):
        """Main strategy loop"""
        self.log_status(
            f"🟢 Strategy activated:\n"
            f"   Symbol: {self.parameters['symbol']}\n"
            f"   Quantity: {self.parameters['quantity']}\n"
            f"   Check Interval: {self.parameters['check_interval']}s\n"
            f"   Mode: Aggressive (Enter immediately)"
        )
        
        while self._continue:
            try:
                if not self.current_trade:
                    # Check if market is open
                    if not self.broker.is_market_open(self.parameters['symbol']):
                        self.log_status(f"⏰ Market closed for {self.parameters['symbol']} - Waiting for market open")
                        time.sleep(60)
                        continue
                    
                    self.log_status("🎯 No position open - Executing immediate entry")
                    self.execute_trade()
                else:
                    # Monitor existing trade
                    current_price = self.broker.get_last_price(self.parameters['symbol'])
                    if current_price:
                        entry_price = self.current_trade['entry_price']
                        profit_pct = ((current_price - entry_price) / entry_price) * 100
                        
                        self.log_status(
                            f"📊 Active Position Status:\n"
                            f"   Current Price: {current_price:.5f}\n"
                            f"   Entry Price: {entry_price:.5f}\n"
                            f"   P/L: {profit_pct:.2f}%\n"
                            f"   Duration: {(self.broker.get_time() - self.current_trade['entry_time']).total_seconds() / 60:.1f}min"
                        )
                        
                        # Exit on downtrend
                        if current_price < entry_price:
                            self.log_status("🔻 Price below entry - Exiting position")
                            self.exit_trade()
                
                time.sleep(self.parameters['check_interval'])
                
            except Exception as e:
                self.log_status(f"❌ Error in strategy loop: {e}")
                time.sleep(5)

    def start(self):
        """Start the strategy in a background thread"""
        self._continue = True
        self.log_status(
            f"🚀 Starting Aggressive Trading Strategy:\n"
            f"   Mode: Immediate Entry\n"
            f"   Symbol: {self.parameters['symbol']}\n"
            f"   Target Quantity: {self.parameters['quantity']}"
        )
        
        def run_strategy():
            self.run()
        
        self._thread = threading.Thread(target=run_strategy, daemon=True)
        self._thread.start()
        
    def stop(self):
        """Stop the strategy"""
        self._continue = False
        if self._thread:
            self._thread.join(timeout=5)
        self.log_status("Strategy stopped")
        
    def should_continue(self):
        return self._continue

    def get_default_quantity(self, symbol):
        """Get default quantity based on instrument type"""
        if 'XAU' in symbol:
            return 1  # Gold trades in smaller units
        if 'BTC' in symbol:
            return 0.1  # Bitcoin trades in smaller units
        if 'SPX' in symbol or 'NAS' in symbol:
            return 1  # Index trades
        if 'BCO' in symbol:
            return 10  # Oil trades in barrels
        return 1000  # Default for forex

    def update_symbol(self, new_symbol):
        """Update trading symbol and adjust quantity accordingly"""
        try:
            if new_symbol not in self.available_instruments:
                self.log_status(f"Invalid symbol: {new_symbol}")
                return False
            
            # Stop strategy if running
            was_running = self._continue
            if was_running:
                self.stop()
            
            # Update symbol and quantity
            self.parameters['symbol'] = new_symbol
            self.parameters['quantity'] = self.get_default_quantity(new_symbol)
            
            # Clear any existing trade data
            self.current_trade = None
            self.last_analysis = None
            
            # Restart if it was running
            if was_running:
                self.start()
            
            self.log_status(f"Updated symbol to {new_symbol} with quantity {self.parameters['quantity']}")
            return True
            
        except Exception as e:
            self.log_status(f"Error updating symbol: {e}")
            logger.error(f"Symbol update error: {e}", exc_info=True)
            return False

    def get_status(self):
        """Enhanced status information"""
        return {
            'running': self._continue,
            'symbol': self.parameters['symbol'],
            'parameters': self.parameters,
            'available_instruments': self.available_instruments,  # Add this
            'last_check': self.last_check_time.isoformat() if self.last_check_time else None,
            'recent_updates': self.status_updates[-10:],
            'performance': self.performance,
            'current_trade': self.current_trade,
            'market_analysis': self.last_analysis,
            'trades_history': self.trades_history[-5:],
            'total_trades_today': len([t for t in self.trades_history if t['timestamp'].date() == datetime.now().date()]),
            'daily_profit_loss': sum(t['profit_loss'] for t in self.trades_history if t['timestamp'].date() == datetime.now().date())
        }

    def log_status(self, message):
        """Add a timestamped status message"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = f"[{timestamp}] {message}"
        logger.info(status)
        self.status_updates.append(status)
        # Keep only last 100 updates
        if len(self.status_updates) > 100:
            self.status_updates.pop(0)

    def calculate_drawdown(self):
        """Calculate current drawdown"""
        if not self.trades_history:
            return 0
        
        peak = 0
        drawdown = 0
        running_pl = 0
        
        for trade in self.trades_history:
            running_pl += trade['profit_loss']
            peak = max(peak, running_pl)
            drawdown = min(drawdown, running_pl - peak)
        
        return drawdown

    def update_performance(self, trade_result):
        """Update performance metrics after a trade"""
        self.performance['total_trades'] += 1
        self.performance['total_profit_loss'] += trade_result['profit_loss']
        
        if trade_result['profit_loss'] > 0:
            self.performance['winning_trades'] += 1
            self.performance['largest_win'] = max(self.performance['largest_win'], trade_result['profit_loss'])
        else:
            self.performance['losing_trades'] += 1
            self.performance['largest_loss'] = min(self.performance['largest_loss'], trade_result['profit_loss'])
            
        # Calculate drawdown
        current_drawdown = self.calculate_drawdown()
        self.performance['current_drawdown'] = current_drawdown
        self.performance['max_drawdown'] = min(self.performance['max_drawdown'], current_drawdown)
        
        # Add to trade history
        self.trades_history.append(trade_result)
        if len(self.trades_history) > 100:  # Keep last 100 trades
            self.trades_history.pop(0) 