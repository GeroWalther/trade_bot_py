import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class BollingerBandsStrategy:
    def __init__(self, broker, parameters=None):
        self.broker = broker
        default_params = {
            'bb_length': 20,
            'bb_std': 2.0,
            'cash_at_risk': 0.1,
            'trailing_stop_loss': False,  # Whether to use trailing stop loss
            'risk_level': 0.02,  # Default risk level (2%)
            'take_profit_level': 0.04  # Default take profit level (4%)
        }
        
        # Update with provided parameters if any
        self.parameters = default_params.copy()
        if parameters:
            self.parameters.update(parameters)
            
        self._continue = False
        self.symbol = 'EUR_USD'  # Default trading pair
        self.last_check_time = None
        self.status_updates = []  # Store recent status updates
        self.trades_history = []
        self.performance = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit_loss': 0,
            'largest_win': 0,
            'largest_loss': 0,
            'current_drawdown': 0,
            'max_drawdown': 0
        }
        self.current_trade = None
        self.last_analysis = None
        
    def should_continue(self):
        return self._continue
        
    def stop(self):
        self._continue = False
        logger.info("Strategy stopped")
        
    def start(self):
        self._continue = True
        logger.info("Strategy started")
        self.log_status(
            f"Strategy activated:\n"
            f"   Symbol: {self.symbol}\n"
            f"   BB Length: {self.parameters['bb_length']}\n"
            f"   BB Std: {self.parameters['bb_std']}\n"
            f"   Cash at Risk: {self.parameters['cash_at_risk'] * 100}%\n"
            f"   Trailing Stop Loss: {'Enabled' if self.parameters['trailing_stop_loss'] else 'Disabled'}\n"
            f"   Risk Level: {self.parameters['risk_level'] * 100}%\n"
            f"   Take Profit Level: {self.parameters['take_profit_level'] * 100}%"
        )
        
    def calculate_position_size(self, cash_at_risk, current_price):
        position_size = (cash_at_risk * self.broker._cash) / current_price
        return round(position_size)
        
    def calculate_indicators(self, prices):
        df = pd.DataFrame(prices, columns=['price'])
        df['SMA'] = df['price'].rolling(window=self.parameters['bb_length']).mean()
        df['STD'] = df['price'].rolling(window=self.parameters['bb_length']).std()
        df['Upper'] = df['SMA'] + (df['STD'] * self.parameters['bb_std'])
        df['Lower'] = df['SMA'] - (df['STD'] * self.parameters['bb_std'])
        return df
        
    def check_entry_signals(self, current_price, indicators):
        if current_price > indicators['Upper'].iloc[-1]:
            return 'sell'
        elif current_price < indicators['Lower'].iloc[-1]:
            return 'buy'
        return None 
    
    def run_iteration(self):
        """Execute one iteration of the strategy"""
        try:
            current_time = datetime.now()
            current_price = self.broker.get_last_price(self.symbol)
            
            if not current_price:
                self.log_status("Could not get current price")
                return
            
            # Check if we need to exit current trade
            if self.current_trade:
                if self.check_exit_conditions(current_price):
                    self.log_status("Exited trade based on exit conditions")
                    return
                
            # Get historical prices for indicator calculation
            prices = []
            for i in range(self.parameters['bb_length'] + 1):
                price = self.broker.get_last_price(self.symbol)
                if price:
                    prices.append({'price': price, 'time': current_time - timedelta(minutes=i)})
            
            if len(prices) < self.parameters['bb_length']:
                self.log_status("Not enough price data")
                return
                
            # Calculate indicators
            indicators = self.calculate_indicators(prices)
            
            # Get current position
            positions = self.broker.get_tracked_positions()
            current_position = positions.get(self.symbol)
            
            # Log current market state
            self.log_status(
                f"Price: {current_price:.5f}, "
                f"Upper: {indicators['Upper'].iloc[-1]:.5f}, "
                f"Lower: {indicators['Lower'].iloc[-1]:.5f}, "
                f"Position: {current_position['quantity'] if current_position else 'None'}"
            )
            
            # Check for trading signals
            signal = self.check_entry_signals(current_price, indicators)
            
            if signal:
                if not current_position:
                    # Calculate position size
                    cash_at_risk = self.parameters['cash_at_risk']
                    position_size = self.calculate_position_size(cash_at_risk, current_price)
                    
                    # Calculate stop loss and take profit
                    if signal == 'buy':
                        raw_stop_loss = current_price * (1 - self.parameters['risk_level'])
                        raw_take_profit = current_price * (1 + self.parameters['take_profit_level'])
                    else:  # signal == 'sell'
                        raw_stop_loss = current_price * (1 + self.parameters['risk_level'])
                        raw_take_profit = current_price * (1 - self.parameters['take_profit_level'])
                    
                    # Format prices according to instrument precision
                    stop_loss = self.format_price(self.symbol, raw_stop_loss)
                    take_profit = self.format_price(self.symbol, raw_take_profit)
                    
                    # Log potential trade
                    self.log_status(
                        f"Signal: {signal.upper()}, "
                        f"Size: {position_size}, "
                        f"Price: {current_price:.5f}, "
                        f"Stop Loss: {stop_loss:.5f}, "
                        f"Take Profit: {take_profit:.5f}"
                    )
                    
                    # Submit order
                    if position_size > 0:
                        order = {
                            'symbol': self.symbol,
                            'quantity': position_size,
                            'side': signal,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit
                        }
                        order_id = self.broker.submit_order(order)
                        if order_id:
                            self.log_status(f"Order executed: {order_id}")
                            # Store trade information
                            self.current_trade = {
                                'order_id': order_id,
                                'symbol': self.symbol,
                                'quantity': position_size,
                                'side': signal,
                                'entry_price': current_price,
                                'entry_time': current_time,
                                'stop_loss': stop_loss,
                                'take_profit': take_profit,
                                'initial_stop_loss': stop_loss  # Store initial stop loss for trailing calculation
                            }
                        else:
                            self.log_status("Order submission failed")
                else:
                    self.log_status("Already in position, skipping signal")
            
            self.last_check_time = current_time
            
        except Exception as e:
            self.log_status(f"Error in strategy: {str(e)}")
            logger.error(f"Strategy error: {e}", exc_info=True)
    
    def log_status(self, message):
        """Add a timestamped status message"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = f"[{timestamp}] {message}"
        logger.info(status)
        self.status_updates.append(status)
        # Keep only last 100 updates
        if len(self.status_updates) > 100:
            self.status_updates.pop(0)
    
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

    def analyze_market(self, current_price, indicators):
        """Analyze current market conditions"""
        sma = indicators['SMA'].iloc[-1]
        upper_band = indicators['Upper'].iloc[-1]
        lower_band = indicators['Lower'].iloc[-1]
        
        # Calculate volatility and trend strength
        volatility = (upper_band - lower_band) / sma * 100
        trend_strength = abs(current_price - sma) / (upper_band - lower_band) * 100
        
        # Determine market conditions
        if current_price > upper_band:
            market_condition = "Overbought"
        elif current_price < lower_band:
            market_condition = "Oversold"
        else:
            market_condition = "Neutral"
            
        self.last_analysis = {
            'timestamp': datetime.now(),
            'price': current_price,
            'sma': sma,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'volatility': volatility,
            'trend_strength': trend_strength,
            'market_condition': market_condition
        }
        
        return self.last_analysis

    def get_status(self):
        """Enhanced status information"""
        return {
            'running': self._continue,
            'symbol': self.symbol,
            'parameters': self.parameters,
            'last_check': self.last_check_time.isoformat() if self.last_check_time else None,
            'recent_updates': self.status_updates[-10:],
            'performance': self.performance,
            'current_trade': self.current_trade,
            'market_analysis': self.last_analysis,
            'trades_history': self.trades_history[-5:],  # Last 5 trades
            'total_trades_today': len([t for t in self.trades_history if t['timestamp'].date() == datetime.now().date()]),
            'daily_profit_loss': sum(t['profit_loss'] for t in self.trades_history if t['timestamp'].date() == datetime.now().date())
        } 

    def check_exit_conditions(self, current_price=None):
        """Check if current trade should be exited based on take profit or stop loss"""
        try:
            if not self.current_trade:
                return False
            
            symbol = self.current_trade['symbol']
            if not current_price:
                current_price = self.broker.get_last_price(symbol)
            
            if not current_price:
                logger.error(f"Could not get current price for {symbol}")
                return False
            
            entry_price = self.current_trade['entry_price']
            stop_loss = self.current_trade['stop_loss']
            take_profit = self.current_trade['take_profit']
            side = self.current_trade['side']
            
            # Log current position status
            profit_loss = (current_price - entry_price) if side == 'buy' else (entry_price - current_price)
            profit_pct = (profit_loss / entry_price) * 100
            
            self.log_status(
                f"Position Update:\n"
                f"   Entry: {entry_price:.5f}\n"
                f"   Current: {current_price:.5f}\n"
                f"   P/L: {profit_loss:.5f} ({profit_pct:.2f}%)\n"
                f"   Stop Loss: {stop_loss:.5f}\n"
                f"   Take Profit: {take_profit:.5f}"
            )
            
            # Check if trailing stop loss is enabled
            if self.parameters['trailing_stop_loss']:
                # Calculate initial stop loss distance
                if side == 'buy':
                    initial_stop_distance = entry_price - self.current_trade['initial_stop_loss']
                    # For BUY trades, if price moves up, move stop loss up
                    raw_new_stop = current_price - initial_stop_distance
                    potential_new_stop = self.format_price(symbol, raw_new_stop)
                    if potential_new_stop > stop_loss:
                        # Update stop loss to new level
                        self.current_trade['stop_loss'] = potential_new_stop
                        logger.info(f"Updated trailing stop loss to {potential_new_stop:.5f}")
                        self.log_status(f"Trailing Stop Loss updated: {potential_new_stop:.5f}")
                        
                        # If trade has an order ID, update the stop loss with the broker
                        if 'order_id' in self.current_trade:
                            try:
                                self.broker.modify_order(
                                    self.current_trade['order_id'],
                                    {'stop_loss': potential_new_stop}
                                )
                            except Exception as e:
                                logger.error(f"Error updating stop loss with broker: {e}")
                else:  # side == 'sell'
                    initial_stop_distance = self.current_trade['initial_stop_loss'] - entry_price
                    # For SELL trades, if price moves down, move stop loss down
                    raw_new_stop = current_price + initial_stop_distance
                    potential_new_stop = self.format_price(symbol, raw_new_stop)
                    if potential_new_stop < stop_loss:
                        # Update stop loss to new level
                        self.current_trade['stop_loss'] = potential_new_stop
                        logger.info(f"Updated trailing stop loss to {potential_new_stop:.5f}")
                        self.log_status(f"Trailing Stop Loss updated: {potential_new_stop:.5f}")
                        
                        # If trade has an order ID, update the stop loss with the broker
                        if 'order_id' in self.current_trade:
                            try:
                                self.broker.modify_order(
                                    self.current_trade['order_id'],
                                    {'stop_loss': potential_new_stop}
                                )
                            except Exception as e:
                                logger.error(f"Error updating stop loss with broker: {e}")
            
            # Check if price hit stop loss or take profit
            if side == 'buy':
                if current_price <= stop_loss:
                    self.log_status(f"Stop Loss triggered at {current_price:.5f}")
                    return self.exit_trade(current_price)
                elif current_price >= take_profit:
                    self.log_status(f"Take Profit triggered at {current_price:.5f}")
                    return self.exit_trade(current_price)
            else:  # side == 'sell'
                if current_price >= stop_loss:
                    self.log_status(f"Stop Loss triggered at {current_price:.5f}")
                    return self.exit_trade(current_price)
                elif current_price <= take_profit:
                    self.log_status(f"Take Profit triggered at {current_price:.5f}")
                    return self.exit_trade(current_price)
            
            return False
            
        except Exception as e:
            self.log_status(f"Error checking exit conditions: {e}")
            logger.error(f"Error in check_exit_conditions: {e}", exc_info=True)
            return False
    
    def exit_trade(self, current_price=None):
        """Exit current trade"""
        try:
            if not self.current_trade:
                return False
            
            symbol = self.current_trade['symbol']
            quantity = self.current_trade['quantity']
            entry_price = self.current_trade['entry_price']
            side = self.current_trade['side']
            
            if not current_price:
                current_price = self.broker.get_last_price(symbol)
            
            if not current_price:
                self.log_status("Could not get current price for exit")
                return False
            
            # Calculate profit/loss
            profit_loss = (current_price - entry_price) * quantity if side == 'buy' else (entry_price - current_price) * quantity
            profit_pct = ((current_price - entry_price) / entry_price) * 100 if side == 'buy' else ((entry_price - current_price) / entry_price) * 100
            
            # Create closing order (opposite side of entry)
            close_side = 'sell' if side == 'buy' else 'buy'
            
            order = {
                'symbol': symbol,
                'quantity': quantity,
                'side': close_side
            }
            
            self.log_status(
                f"Closing position - Entry: {entry_price:.5f}, "
                f"Exit: {current_price:.5f}, "
                f"P/L: {profit_loss:.2f} ({profit_pct:.2f}%)"
            )
            
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
                    'duration': (datetime.now() - self.current_trade['entry_time']).total_seconds() / 60,
                    'timestamp': datetime.now()
                }
                
                self.update_performance(trade_result)
                self.current_trade = None
                
                self.log_status(
                    f"Position closed - Duration: {trade_result['duration']:.1f} minutes, "
                    f"Total P/L: {self.performance['total_profit_loss']:.2f}"
                )
                return True
            
            self.log_status("Failed to close position")
            return False
            
        except Exception as e:
            self.log_status(f"Error closing trade: {e}")
            logger.error(f"Error in exit_trade: {e}", exc_info=True)
            return False 

    def get_price_precision(self, symbol):
        """Get the required price precision for a given instrument"""
        # Check for forex pairs first (they contain an underscore and currency codes)
        if '_' in symbol and any(curr in symbol for curr in ['EUR', 'GBP', 'USD', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD']):
            # Make sure it's not a crypto or index that happens to have USD in the name
            if not any(asset in symbol for asset in ['BTC', 'SPX', 'NAS', 'XAU']):
                return 5  # 5 decimal places for forex pairs
        
        # Then check for specific instruments
        if 'XAU' in symbol:
            return 2  # 2 decimal places for Gold
        elif 'BTC' in symbol:
            return 2  # 2 decimal places for Bitcoin
        elif 'SPX' in symbol or 'NAS' in symbol:
            return 2  # 2 decimal places for indices
        else:
            return 2  # Default to 2 decimal places for safety

    def format_price(self, symbol, price):
        """Format price according to instrument precision requirements"""
        try:
            precision = self.get_price_precision(symbol)
            
            # For decimal precision instruments
            formatted_price = round(float(price), precision)
            
            return formatted_price
        except Exception as e:
            self.log_status(f"Error formatting price: {str(e)}")
            # Return original price as fallback
            return price 

    def clear_logs(self):
        """Manually clear all status updates/logs"""
        self.status_updates = []
        self.log_status("🧹 Logs have been manually cleared")
        return True 