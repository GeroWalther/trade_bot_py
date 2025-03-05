import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import threading
import time

logger = logging.getLogger(__name__)

class EMATrendStrategy:
    def __init__(self, broker, parameters=None):
        # Add debug logging
        logger.info(f"Initializing EMA Strategy with parameters: {parameters}")
        
        self.broker = broker
        self.available_instruments = [
            'EUR_USD', 'GBP_USD', 'USD_JPY', 'AUD_USD', 
            'USD_CAD', 'BTC_USD', 'SPX500_USD', 'NAS100_USD', 
            'XAU_USD', 'BCO_USD'
        ]
        
        # Default parameters - make a deep copy to avoid reference issues
        self.parameters = {
            'symbol': 'BTC_USD',
            'quantity': 0.1,  # This will be used as a fallback/minimum
            'ema_period': 50,
            'check_interval': 240,
            'continue_after_trade': True,
            'max_concurrent_trades': 1,
            'trailing_stop_loss': False,  # Whether to use trailing stop loss
            'take_profit_level': 0.04,  # Default take profit level (4%)
            'risk_percent': 0.5,  # Default risk percent per trade (0.5%)
        }
        
        # Update with provided parameters if any
        if parameters:
            self.parameters.update(parameters)
        
        logger.info(f"Final parameters after initialization: {self.parameters}")
        
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
            'largest_loss': 0,
            'current_drawdown': 0,
            'max_drawdown': 0
        }
        self.current_trade = None
        self.last_analysis = None
        self._thread = None

    def calculate_ema(self, prices, period=50):
        """Calculate EMA for a list of prices"""
        if len(prices) < period:
            return None
        
        # Convert to numpy array for calculations
        prices = np.array(prices)
        multiplier = 2 / (period + 1)
        ema = [prices[0]]  # First value is price
        
        for price in prices[1:]:
            ema.append((price - ema[-1]) * multiplier + ema[-1])
        
        return ema[-1]  # Return the last EMA value

    def get_price_action(self, symbol=None):
        """Get price action and EMA"""
        try:
            symbol = symbol or self.parameters.get('symbol')
            if not symbol:
                logger.error("No trading symbol specified")
                return None
            
            current_price = self.broker.get_last_price(symbol)
            
            if not current_price:
                logger.error(f"Could not get current price for {symbol}")
                return None
            
            # Get historical prices for EMA calculation
            historical_prices = []
            price_history = self.broker._price_history.get(symbol, [])
            for price_data in price_history:
                if isinstance(price_data, dict) and 'price' in price_data:
                    historical_prices.append(price_data['price'])
                elif isinstance(price_data, (int, float)):
                    historical_prices.append(price_data)
            
            # Add current price to historical prices
            historical_prices.append(current_price)
            
            # Calculate EMA
            ema_value = self.calculate_ema(historical_prices, self.parameters['ema_period'])
            if not ema_value:
                logger.error(f"Could not calculate EMA for {symbol}")
                return None
            
            # Determine trend based on price vs EMA
            trend = "UP" if current_price > ema_value else "DOWN"
            
            return {
                'current_price': current_price,
                'ema': ema_value,
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
                
            # Always enter a trade regardless of trend direction
            # We'll determine if it's a LONG or SHORT in execute_trade
            return True
            
        except Exception as e:
            logger.error(f"Error in should_enter_trade: {e}")
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
        """Format price according to instrument precision"""
        try:
            # Specific handling for JPY pairs
            if symbol == 'USD_JPY' or symbol == 'EUR_JPY' or 'JPY' in symbol:
                return round(price, 3)  # JPY pairs typically use 3 decimal places
            
            # Handle other forex pairs
            elif '_' in symbol and any(curr in symbol for curr in ['EUR', 'GBP', 'USD', 'AUD', 'CAD', 'CHF', 'NZD']):
                # Make sure it's not a crypto or index that happens to have USD in the name
                if not any(asset in symbol for asset in ['BTC', 'SPX', 'NAS', 'XAU']):
                    return round(price, 5)  # 5 decimal places for forex pairs
            
            # Handle specific instruments
            elif 'XAU' in symbol:
                return round(price, 2)  # 2 decimal places for Gold
            elif 'BTC' in symbol:
                return round(price, 1)  # 1 decimal place for Bitcoin
            elif 'SPX' in symbol or 'NAS' in symbol:
                return round(price, 1)  # 1 decimal place for indices
            else:
                return round(price, 2)  # Default to 2 decimal places for safety
                
        except Exception as e:
            self.log_status(f"Error formatting price: {e}")
            return round(price, 2)  # Return rounded price if formatting fails

    def execute_trade(self):
        """Execute trade based on EMA strategy"""
        try:
            logger.info("Starting execute_trade...")
            symbol = self.parameters['symbol']
            
            # Check for existing positions first
            current_positions = self.broker.get_tracked_positions()
            symbol_positions = []
            for pos in current_positions.values():
                if isinstance(pos, dict) and 'instrument' in pos:
                    if pos['instrument'] == symbol:
                        symbol_positions.append(pos)
            
            # Enforce max concurrent trades = 1
            if len(symbol_positions) >= 1:
                self.log_status(f"⚠️ Maximum trades reached ({len(symbol_positions)}). Cannot open new position.")
                return False
            
            # Calculate position size based on risk percent
            risk_percent = self.parameters.get('risk_percent', 0.5)  # Default to 0.5% if not specified
            account_balance = self.broker._cash
            
            # Get current price and EMA
            price_action = self.get_price_action(symbol)
            logger.info(f"Price action: {price_action}")
            
            if not price_action:
                self.log_status(f"❌ Cannot execute trade - No price data for {symbol}")
                return False
            
            current_price = price_action['current_price']
            ema_value = price_action['ema']
            logger.info(f"Current price: {current_price}, EMA: {ema_value}")
            
            # Determine trade direction based on price vs EMA
            if current_price > ema_value:
                side = 'buy'
                self.log_status(f"🚀 Price ({current_price:.5f}) above EMA ({ema_value:.5f}) - Going LONG")
                
                # Use EMA as stop loss for BUY (with small buffer)
                buffer = ema_value * 0.001  # 0.1% buffer below EMA
                raw_stop_loss = ema_value - buffer
                raw_take_profit = current_price * (1 + self.parameters['take_profit_level'])
                
                # Calculate position size based on risk
                risk_amount = (account_balance * risk_percent / 100)
                stop_loss_distance = current_price - raw_stop_loss
                if stop_loss_distance <= 0:
                    self.log_status(f"❌ Invalid stop loss distance: {stop_loss_distance}")
                    return False
                quantity = risk_amount / stop_loss_distance
                
                # Format prices according to instrument precision
                stop_loss = self.format_price(symbol, raw_stop_loss)
                take_profit = self.format_price(symbol, raw_take_profit)
                
                self.log_status(f"🛡️ Stop Loss set at 50 EMA: {stop_loss} (Distance: {stop_loss_distance:.5f})")
            else:
                side = 'sell'
                self.log_status(f"🔻 Price ({current_price:.5f}) below EMA ({ema_value:.5f}) - Going SHORT")
                
                # Use EMA as stop loss for SELL (with small buffer)
                buffer = ema_value * 0.001  # 0.1% buffer above EMA
                raw_stop_loss = ema_value + buffer
                raw_take_profit = current_price * (1 - self.parameters['take_profit_level'])
                
                # Calculate position size based on risk
                risk_amount = (account_balance * risk_percent / 100)
                stop_loss_distance = raw_stop_loss - current_price
                if stop_loss_distance <= 0:
                    self.log_status(f"❌ Invalid stop loss distance: {stop_loss_distance}")
                    return False
                quantity = risk_amount / stop_loss_distance
                
                # Format prices according to instrument precision
                stop_loss = self.format_price(symbol, raw_stop_loss)
                take_profit = self.format_price(symbol, raw_take_profit)
                
                self.log_status(f"🛡️ Stop Loss set at 50 EMA: {stop_loss} (Distance: {stop_loss_distance:.5f})")
            
            # Format quantity based on instrument
            if 'JPY' in symbol:
                quantity = round(quantity, 0)  # Whole units for JPY pairs
                quantity = max(quantity, 1)  # Minimum 1 unit
            else:
                quantity = max(quantity, 0.01)  # Ensure minimum quantity
                quantity = round(quantity, 2)  # Round to 2 decimal places
            
            self.log_status(f"💰 Risk-based position size: {quantity} (Risk: {risk_percent}%, Account: {account_balance:.2f})")
            
            # Format current price
            formatted_current_price = self.format_price(symbol, current_price)
            
            logger.info(f"Submitting order: {symbol} {side} {quantity} @ {formatted_current_price}")
            logger.info(f"Stop Loss: {stop_loss}, Take Profit: {take_profit}")
            
            order = {
                'symbol': symbol,
                'quantity': quantity,
                'side': side,
                'price': formatted_current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }
            
            self.log_status(f"🔄 Initiating {side.upper()} order: {quantity} {symbol} @ {formatted_current_price}")
            self.log_status(f"📊 Risk Management: Stop Loss @ {stop_loss}, Take Profit @ {take_profit}")
            
            order_id = self.broker.submit_order(order)
            logger.info(f"Order result: {order_id}")
            
            if order_id:
                self.current_trade = {
                    'order_id': order_id,
                    'symbol': symbol,
                    'quantity': quantity,
                    'side': side,
                    'entry_price': formatted_current_price,
                    'entry_time': self.broker.get_time(),
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'initial_stop_loss': stop_loss  # Store initial stop loss for trailing calculation
                }
                
                self.log_status(f"✅ Order placed successfully: ID {order_id}")
                self.log_status(f"🔔 Position OPENED: {quantity} {symbol} @ {formatted_current_price} ({side.upper()})")
                return True
            
            self.log_status("❌ Failed to place order")
            return False
            
        except Exception as e:
            self.log_status(f"❌ Error executing trade: {e}")
            logger.error(f"Error in execute_trade: {e}", exc_info=True)
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
            f"   Check Interval: {self.parameters['check_interval']}s\n"
            f"   Continue After Trade: {'Yes' if self.parameters['continue_after_trade'] else 'No'}\n"
            f"   Trailing Stop Loss: {'Enabled' if self.parameters['trailing_stop_loss'] else 'Disabled'}\n"
            f"   Take Profit Level: {self.parameters['take_profit_level'] * 100}%\n"
            f"   Risk Per Trade: {self.parameters['risk_percent']}%\n"
            f"   Stop Loss: 50 EMA"
        )
        
        while self._continue:
            try:
                self.check_and_trade()
                time.sleep(self.parameters['check_interval'])
            except Exception as e:
                self.log_status(f"❌ Error in strategy loop: {e}")
                time.sleep(5)

    def start(self):
        """Start the strategy"""
        logger.info("Starting EMA strategy...")
        logger.info(f"Parameters at start: {self.parameters}")
        
        # Verify parameters before starting
        if not self.parameters.get('symbol'):
            error_msg = "Cannot start - No trading symbol specified"
            logger.error(error_msg)
            self.log_status(f"❌ {error_msg}")
            return
        
        self._continue = True
        
        self.log_status(
            f"🚀 Starting Aggressive Trading Strategy:\n"
            f"   Mode: Immediate Entry\n"
            f"   Symbol: {self.parameters['symbol']}\n"
            f"   Target Quantity: {self.parameters['quantity']}"
        )
        
        # Run directly
        self.run()

    def stop(self):
        """Stop the strategy"""
        self._continue = False
        self.log_status("🔴 Strategy stopped")

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
        """Get current strategy status"""
        strategy_info = {
            'name': 'EMA Trend Strategy',
            'period': '50 Period EMA',
            'description': 'Uses a 50-period Exponential Moving Average (EMA) to determine trade direction and risk management:',
            'rules': [
                'Enter LONG when price crosses above EMA',
                'Enter SHORT when price crosses below EMA',
                'Exit LONG when price crosses below EMA',
                'Exit SHORT when price crosses above EMA',
                'Stop loss is placed at the 50 EMA level (with small buffer)',
                'Position size is calculated based on risk percent per trade (user-configurable)',
                'Take profit is set at a user-defined percentage from entry price'
            ]
        }
        
        return {
            'name': 'EMA Trend Strategy',
            'running': self._continue,
            'parameters': self.parameters,
            'performance': self.performance,
            'recent_updates': self.status_updates,
            'available_instruments': self.available_instruments,
            'current_trade': self.current_trade,
            'strategy_info': strategy_info  # Add strategy info to status
        }

    def log_status(self, message):
        """Add a timestamped status message"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = f"[{timestamp}] {message}"
        logger.info(status)
        self.status_updates.append(status)
        # Keep last 500 updates instead of 100
        if len(self.status_updates) > 500:
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
        try:
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
            self.performance['max_drawdown'] = min(self.performance.get('max_drawdown', 0), current_drawdown)
            
            # Add to trade history
            self.trades_history.append(trade_result)
            if len(self.trades_history) > 100:  # Keep last 100 trades
                self.trades_history.pop(0)
            
        except Exception as e:
            logger.error(f"Error updating performance: {str(e)}")
            logger.error(f"Trade result: {trade_result}")
            logger.error(f"Current performance: {self.performance}")

    def check_exit_conditions(self):
        """Check if current trade should be exited based on take profit, stop loss, or EMA crossover"""
        try:
            if not self.current_trade:
                return False
            
            symbol = self.current_trade['symbol']
            
            # Get current price and EMA
            price_action = self.get_price_action(symbol)
            if not price_action:
                logger.error(f"Could not get price action for {symbol}")
                return False
                
            current_price = price_action['current_price']
            ema_value = price_action['ema']
            
            entry_price = self.current_trade.get('entry_price')
            stop_loss = self.current_trade.get('stop_loss')
            take_profit = self.current_trade.get('take_profit')
            side = self.current_trade.get('side')
            
            # Check for None values and handle them
            if entry_price is None or current_price is None:
                self.log_status("⚠️ Missing price data for exit calculation")
                return False
            
            # Log current position status
            try:
                profit_loss = (current_price - entry_price) if side == 'buy' else (entry_price - current_price)
                profit_pct = (profit_loss / entry_price) * 100 if entry_price != 0 else 0
                
                status_message = f"📈 Position Update:\n" \
                                f"   Entry: {entry_price}\n" \
                                f"   Current: {current_price}\n" \
                                f"   EMA: {ema_value}\n" \
                                f"   P/L: {profit_loss:.5f} ({profit_pct:.2f}%)"
                
                if stop_loss is not None:
                    status_message += f"\n   Stop Loss: {stop_loss}"
                if take_profit is not None:
                    status_message += f"\n   Take Profit: {take_profit}"
                    
                self.log_status(status_message)
            except Exception as e:
                self.log_status(f"⚠️ Error calculating position metrics: {e}")
            
            # Check if trailing stop loss is enabled
            if self.parameters['trailing_stop_loss']:
                # For trailing stop loss, we'll use the EMA as the reference
                if side == 'buy':
                    # For BUY trades, if price moves up, move stop loss up to just below EMA
                    buffer = ema_value * 0.001  # 0.1% buffer below EMA
                    potential_new_stop = self.format_price(symbol, ema_value - buffer)
                    if potential_new_stop > stop_loss:
                        # Update stop loss to new level
                        self.current_trade['stop_loss'] = potential_new_stop
                        logger.info(f"Updated trailing stop loss to {potential_new_stop:.5f} (EMA: {ema_value:.5f})")
                        self.log_status(f"🔄 Trailing Stop Loss updated to follow EMA: {potential_new_stop:.5f}")
                        
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
                    # For SELL trades, if price moves down, move stop loss down to just above EMA
                    buffer = ema_value * 0.001  # 0.1% buffer above EMA
                    potential_new_stop = self.format_price(symbol, ema_value + buffer)
                    if potential_new_stop < stop_loss:
                        # Update stop loss to new level
                        self.current_trade['stop_loss'] = potential_new_stop
                        logger.info(f"Updated trailing stop loss to {potential_new_stop:.5f} (EMA: {ema_value:.5f})")
                        self.log_status(f"🔄 Trailing Stop Loss updated to follow EMA: {potential_new_stop:.5f}")
                        
                        # If trade has an order ID, update the stop loss with the broker
                        if 'order_id' in self.current_trade:
                            try:
                                self.broker.modify_order(
                                    self.current_trade['order_id'],
                                    {'stop_loss': potential_new_stop}
                                )
                            except Exception as e:
                                logger.error(f"Error updating stop loss with broker: {e}")
            
            # Check if price crossed the EMA in the opposite direction
            if side == 'buy' and current_price < ema_value:
                self.log_status(f"🔄 EMA Crossover: Price ({current_price:.5f}) crossed below EMA ({ema_value:.5f}) - Exiting LONG position")
                return self.exit_trade()
            elif side == 'sell' and current_price > ema_value:
                self.log_status(f"🔄 EMA Crossover: Price ({current_price:.5f}) crossed above EMA ({ema_value:.5f}) - Exiting SHORT position")
                return self.exit_trade()
            
            # Check if price hit stop loss or take profit
            if side == 'buy':
                if current_price <= stop_loss:
                    self.log_status(f"🛑 Stop Loss triggered at {current_price:.5f}")
                    return self.exit_trade()
                elif current_price >= take_profit:
                    self.log_status(f"💰 Take Profit triggered at {current_price:.5f}")
                    return self.exit_trade()
            else:  # side == 'sell'
                if current_price >= stop_loss:
                    self.log_status(f"🛑 Stop Loss triggered at {current_price:.5f}")
                    return self.exit_trade()
                elif current_price <= take_profit:
                    self.log_status(f"💰 Take Profit triggered at {current_price:.5f}")
                    return self.exit_trade()
            
            return False
            
        except Exception as e:
            self.log_status(f"❌ Error checking exit conditions: {e}")
            logger.error(f"Error in check_exit_conditions: {e}", exc_info=True)
            return False

    def check_and_trade(self):
        """Main trading logic"""
        try:
            logger.info("Starting check_and_trade cycle")
            logger.info(f"Current parameters: {self.parameters}")
            
            # Check if we need to exit current trade
            if self.current_trade:
                if self.check_exit_conditions():
                    logger.info("Exited trade based on exit conditions")
                    return
            
            # Get current positions for this strategy
            current_positions = self.broker.get_tracked_positions()
            logger.info(f"Position structure from broker: {current_positions}")
            
            symbol = self.parameters.get('symbol')
            if not symbol:
                error_msg = "No trading symbol specified in parameters"
                logger.error(error_msg)
                self.log_status(f"❌ {error_msg}")
                return

            # Get price action and EMA
            price_action = self.get_price_action(symbol)
            if not price_action:
                return
            
            current_price = price_action['current_price']
            ema_value = price_action['ema']
            
            # Log current market conditions
            self.log_status(
                f"📊 Market Analysis:\n"
                f"   Price: {current_price:.5f}\n"
                f"   EMA: {ema_value:.5f}\n"
                f"   Trend: {price_action['trend']}"
            )

            # Check existing positions
            symbol_positions = []
            for pos in current_positions.values():
                if isinstance(pos, dict) and 'instrument' in pos:
                    if pos['instrument'] == symbol:
                        symbol_positions.append(pos)
            
            # Always use max_concurrent_trades = 1
            if len(symbol_positions) >= 1:
                self.log_status(f"Maximum trades reached ({len(symbol_positions)}). Waiting for positions to close.")
                return
            
            # Check if we should enter a trade based on EMA signal
            should_trade = self.should_enter_trade(symbol)
            logger.info(f"Should enter trade based on EMA signal: {should_trade}")
            
            if should_trade:
                # Execute trade immediately based on EMA position
                logger.info("About to execute trade...")
                self.log_status("💡 Executing trade based on EMA position")
                result = self.execute_trade()
                logger.info(f"Trade execution result: {result}")
            else:
                self.log_status("⏳ Waiting for EMA signal to enter trade")
            
        except Exception as e:
            self.log_status(f"❌ Error in trading logic: {str(e)}")
            logger.error(f"Trading error: {e}", exc_info=True)

    # Add clear_logs method
    def clear_logs(self):
        """Manually clear all status updates/logs"""
        self.status_updates = []
        self.log_status("🧹 Logs have been manually cleared")
        return True 