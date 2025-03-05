from datetime import datetime
import logging
import time

logger = logging.getLogger(__name__)

class BBStrategy:
    def __init__(self, broker, parameters=None):
        """Initialize the Bollinger Bands strategy"""
        self.name = "Bollinger Bands Strategy"
        self.description = "A strategy that uses Bollinger Bands to identify overbought and oversold conditions."
        self.broker = broker
        self.status_messages = []
        self.current_trade = None
        
        # Initialize performance tracking
        self.performance = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0
        }
        
        # Default parameters
        default_params = {
            'symbol': None,
            'quantity': 1,
            'bb_length': 20,
            'bb_std_dev': 2,
            'risk_percent': 1.0,  # Risk 1% of account per trade
            'check_interval': 60,  # Check every 60 seconds by default
            'continue_after_trade': True,  # Continue after a trade by default
        }
        
        # Override defaults with any provided parameters
        self.parameters = default_params.copy()
        if parameters:
            self.parameters.update(parameters)
            
        # Log initialization
        self.log_status(f"🔄 Initialized {self.name}")
        self.log_status(f"📊 Parameters: {self.parameters}")
        
        self.available_instruments = [
            'EUR_USD', 'GBP_USD', 'USD_JPY', 'AUD_USD', 
            'USD_CAD', 'BTC_USD', 'SPX500_USD', 'NAS100_USD', 
            'XAU_USD', 'BCO_USD'
        ]
        self._continue = False
        self.trades_history = []

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
            'recent_updates': self.status_messages,
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
            f"   Symbol: {self.parameters.get('symbol', 'Not set')}\n"
            f"   BB Length: {self.parameters.get('bb_length', 20)}\n"
            f"   BB StdDev: {self.parameters.get('bb_std_dev', 2)}\n"
            f"   Check Interval: {self.parameters.get('check_interval', 60)}s\n"
            f"   Continue After Trade: {'Yes' if self.parameters.get('continue_after_trade', True) else 'No'}\n"
            f"   Risk Per Trade: {self.parameters.get('risk_percent', 1.0)}%"
        )
        
        # Check if symbol is set
        if not self.parameters.get('symbol'):
            self.log_status("⚠️ No trading symbol set. Please set a symbol to start trading.")
            return
            
        self.log_status(f"👀 Watching {self.parameters['symbol']} for Bollinger Bands signals...")
        
        while self._continue:
            try:
                self.check_and_trade()
                interval = self.parameters.get('check_interval', 60)
                self.log_status(f"⏱️ Waiting {interval} seconds before next check...")
                time.sleep(interval)
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

            # Log the start of the trading check
            self.log_status(f"🔍 Checking for trading opportunities on {symbol}...")

            # Check for existing positions
            symbol_positions = [pos for pos in current_positions.values() 
                              if pos['symbol'] == symbol]
            
            # Always use max_concurrent_trades = 1
            if len(symbol_positions) >= 1:
                self.log_status(f"⚠️ Maximum trades reached ({len(symbol_positions)}). Waiting for positions to close.")
                # Log details about current positions
                for pos in symbol_positions:
                    side = pos.get('side', 'unknown')
                    entry_price = pos.get('entry_price', 'unknown')
                    quantity = pos.get('quantity', 'unknown')
                    self.log_status(f"📊 Current position: {side.upper()} {quantity} {symbol} @ {entry_price}")
                
                # Check if we need to exit any positions
                self.log_status("🔄 Checking exit conditions for current positions...")
                self.check_exit_conditions(symbol_positions)
                return

            # Get price action with Bollinger Bands
            self.log_status(f"📈 Retrieving price data and calculating Bollinger Bands for {symbol}...")
            price_action = self.get_price_action(symbol)
            if not price_action:
                self.log_status("❌ Failed to get price data or calculate Bollinger Bands")
                return
                
            # Extract values from price action
            current_price = price_action['current_price']
            bb_upper = price_action['bb_upper']
            bb_lower = price_action['bb_lower']
            bb_middle = price_action['bb_middle']
            
            # Check for entry signals
            self.log_status("🔍 Analyzing price action for entry signals...")
            entry_signal = self.check_entry_signals(current_price, bb_upper, bb_lower, bb_middle)
            
            if entry_signal:
                side, entry_price, stop_loss, take_profit = entry_signal
                
                # Calculate position size based on risk
                self.log_status("💰 Calculating position size based on risk parameters...")
                position_size = self.calculate_position_size(entry_price, stop_loss)
                
                # Execute the trade
                self.log_status(f"🚀 Executing {side.upper()} trade for {symbol}...")
                self.execute_trade(symbol, side, position_size, entry_price, stop_loss, take_profit)
            else:
                self.log_status("⏳ No valid entry signals detected. Waiting for next check.")

        except Exception as e:
            self.log_status(f"❌ Error in trading logic: {str(e)}")
            logger.error(f"Trading error: {e}", exc_info=True)

    def get_price_action(self, symbol=None):
        """Get price action and calculate Bollinger Bands"""
        try:
            symbol = symbol or self.parameters.get('symbol')
            if not symbol:
                logger.error("No trading symbol specified")
                self.log_status("❌ No trading symbol specified")
                return None

            # Log that we're fetching the current price
            self.log_status(f"📊 Fetching current price for {symbol}...")
            current_price = self.broker.get_last_price(symbol)
            if not current_price:
                error_msg = f"Could not get current price for {symbol}"
                logger.error(error_msg)
                self.log_status(f"❌ {error_msg}")
                return None
            
            self.log_status(f"💰 Current price for {symbol}: {current_price}")

            # Log that we're fetching historical data
            self.log_status(f"📈 Fetching historical data for Bollinger Bands calculation...")
            
            # Get historical prices for Bollinger Bands calculation
            try:
                historical_data = self.broker.get_historical_prices(
                    symbol, 
                    count=self.parameters['bb_length'] * 3,  # Get more data than needed for better accuracy
                    granularity='M1'  # 1-minute candles
                )
            except Exception as e:
                error_msg = f"Error fetching historical data: {str(e)}"
                logger.error(error_msg)
                self.log_status(f"❌ {error_msg}")
                return None
            
            if not historical_data:
                error_msg = f"No historical data returned for {symbol}"
                logger.error(error_msg)
                self.log_status(f"❌ {error_msg}")
                return None
                
            if len(historical_data) < self.parameters['bb_length']:
                error_msg = f"Not enough historical data for {symbol} to calculate Bollinger Bands. Got {len(historical_data)} candles, need at least {self.parameters['bb_length']}."
                logger.error(error_msg)
                self.log_status(f"❌ {error_msg}")
                return None
            
            self.log_status(f"📊 Got {len(historical_data)} candles of historical data")
                
            # Extract close prices
            try:
                close_prices = [float(candle['c']) for candle in historical_data]
            except (KeyError, TypeError, ValueError) as e:
                error_msg = f"Error extracting close prices from historical data: {str(e)}"
                logger.error(error_msg)
                self.log_status(f"❌ {error_msg}")
                # Log the first candle to help debug
                if historical_data and len(historical_data) > 0:
                    logger.error(f"First candle data: {historical_data[0]}")
                    self.log_status(f"⚠️ First candle data: {historical_data[0]}")
                return None
            
            # Calculate SMA (middle band)
            bb_length = self.parameters['bb_length']
            sma = sum(close_prices[:bb_length]) / bb_length
            
            # Calculate standard deviation
            variance = sum([(price - sma) ** 2 for price in close_prices[:bb_length]]) / bb_length
            std_dev = variance ** 0.5
            
            # Calculate upper and lower bands
            bb_std = self.parameters['bb_std_dev']
            bb_upper = sma + (std_dev * bb_std)
            bb_lower = sma - (std_dev * bb_std)
            
            self.log_status(f"✅ Bollinger Bands calculated: Middle={sma:.2f}, Upper={bb_upper:.2f}, Lower={bb_lower:.2f}")
            
            return {
                'current_price': current_price,
                'bb_upper': bb_upper,
                'bb_lower': bb_lower,
                'bb_middle': sma
            }
            
        except Exception as e:
            error_msg = f"Error calculating Bollinger Bands: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.log_status(f"❌ {error_msg}")
            return None

    def calculate_position_size(self, entry_price, stop_loss):
        """Calculate position size based on risk percentage"""
        try:
            # Get account balance
            account_info = self.broker.get_account_info()
            if not account_info or 'balance' not in account_info:
                self.log_status("❌ Could not retrieve account balance")
                return self.parameters.get('quantity', 1)  # Use default quantity
                
            account_balance = account_info['balance']
            
            # Calculate risk amount in account currency
            risk_percent = self.parameters.get('risk_percent', 1.0)
            risk_amount = account_balance * (risk_percent / 100)
            
            # Calculate price difference between entry and stop loss
            price_difference = abs(entry_price - stop_loss)
            if price_difference == 0:
                self.log_status("⚠️ Entry and stop loss are the same price, using default quantity")
                return self.parameters.get('quantity', 1)
                
            # Calculate position size based on risk
            position_size = risk_amount / price_difference
            
            # Round down to appropriate decimal places based on asset type
            symbol = self.parameters.get('symbol')
            
            # For forex, position size is typically in lots (100,000 units)
            if '_' in symbol and any(curr in symbol for curr in ['EUR', 'GBP', 'USD', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD']):
                # Convert to mini lots (10,000 units) and round to 2 decimal places
                position_size = round(position_size / 10000, 2)
                # Ensure minimum size
                position_size = max(position_size, 0.01)
                
            # For stocks and indices, round to whole shares
            elif any(asset in symbol for asset in ['SPX', 'NAS']):
                position_size = max(1, int(position_size))
                
            # For crypto, allow fractional units but with minimum size
            elif 'BTC' in symbol:
                position_size = round(position_size, 4)
                position_size = max(position_size, 0.001)
                
            # For other assets, use default rounding
            else:
                position_size = round(position_size, 2)
                position_size = max(position_size, 1)
                
            # Log the position size calculation
            self.log_status(
                f"💰 Position sizing:\n"
                f"   Account Balance: {account_balance:.2f}\n"
                f"   Risk: {risk_percent}% = {risk_amount:.2f}\n"
                f"   Price Difference: {price_difference:.5f}\n"
                f"   Position Size: {position_size}"
            )
            
            return position_size
            
        except Exception as e:
            self.log_status(f"❌ Error calculating position size: {str(e)}")
            logger.error(f"Position size calculation error: {e}", exc_info=True)
            return self.parameters.get('quantity', 1)  # Use default quantity on error

    def stop(self):
        """Stop the strategy"""
        self._continue = False
        self.log_status("🔴 Strategy stopped")

    def log_status(self, message):
        """Add a timestamped status message"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = f"[{timestamp}] {message}"
        logger.info(status)
        self.status_messages.append(status)
        # Keep only last 100 updates
        if len(self.status_messages) > 100:
            self.status_messages.pop(0)

    def clear_logs(self):
        """Clear status updates and log history"""
        try:
            # Clear the status messages list
            self.status_messages = []
            
            # Add a confirmation message
            self.log_status("🧹 Logs cleared")
            
            # Log to the console for debugging
            logger.info("Logs cleared successfully")
            
            return True
        except Exception as e:
            logger.error(f"Error clearing logs: {e}")
            # Try to add a message even if there was an error
            try:
                self.status_messages = ["❌ Error clearing logs"]
            except:
                pass
            return False

    def check_entry_signals(self, current_price, bb_upper, bb_lower, bb_middle):
        """Check for entry signals based on Bollinger Bands"""
        try:
            # Calculate band width as percentage
            band_width_percent = ((bb_upper - bb_lower) / bb_middle) * 100
            
            # Calculate price position within bands as percentage (0% = lower band, 100% = upper band)
            if bb_upper != bb_lower:  # Avoid division by zero
                price_position = ((current_price - bb_lower) / (bb_upper - bb_lower)) * 100
            else:
                price_position = 50  # Default to middle if bands are equal (unlikely)
            
            # Log the current price relative to bands
            self.log_status(
                f"💹 Price Analysis:\n"
                f"   Current: {current_price:.2f}\n"
                f"   Upper Band: {bb_upper:.2f}\n"
                f"   Middle Band: {bb_middle:.2f}\n"
                f"   Lower Band: {bb_lower:.2f}\n"
                f"   Band Width: {band_width_percent:.2f}%\n"
                f"   Position in Band: {price_position:.1f}%"
            )
            
            # Buy signal: Price touches or crosses below lower band (oversold)
            if current_price <= bb_lower:
                self.log_status("🔔 BUY SIGNAL: Price at or below lower Bollinger Band (oversold)")
                
                # Set stop loss below recent low, or fixed percentage if needed
                stop_loss = current_price * 0.99  # 1% below entry as a simple default
                
                # Set take profit at middle band
                take_profit = bb_middle
                
                return ('buy', current_price, stop_loss, take_profit)
                
            # Sell signal: Price touches or crosses above upper band (overbought)
            elif current_price >= bb_upper:
                self.log_status("🔔 SELL SIGNAL: Price at or above upper Bollinger Band (overbought)")
                
                # Set stop loss above recent high, or fixed percentage if needed
                stop_loss = current_price * 1.01  # 1% above entry as a simple default
                
                # Set take profit at middle band
                take_profit = bb_middle
                
                return ('sell', current_price, stop_loss, take_profit)
                
            # No signal
            else:
                # Provide more detailed information about why no signal
                if price_position < 20:
                    self.log_status("⏳ No signal yet: Price approaching lower band (potential buy soon)")
                elif price_position > 80:
                    self.log_status("⏳ No signal yet: Price approaching upper band (potential sell soon)")
                else:
                    self.log_status("⏳ No trading signal: Price within normal range of Bollinger Bands")
                return None
                
        except Exception as e:
            logger.error(f"Error checking entry signals: {e}")
            self.log_status(f"❌ Error checking entry signals: {e}")
            return None
            
    def execute_trade(self, symbol, side, quantity, entry_price, stop_loss, take_profit):
        """Execute a trade based on the given parameters"""
        try:
            # Format prices for the broker API
            entry_price_formatted = self.format_price(symbol, entry_price)
            stop_loss_formatted = self.format_price(symbol, stop_loss)
            take_profit_formatted = self.format_price(symbol, take_profit)
            
            # Log the trade details
            self.log_status(
                f"🚀 Executing {side.upper()} order:\n"
                f"   Symbol: {symbol}\n"
                f"   Quantity: {quantity}\n"
                f"   Entry: {entry_price_formatted}\n"
                f"   Stop Loss: {stop_loss_formatted}\n"
                f"   Take Profit: {take_profit_formatted}"
            )
            
            # Submit the order to the broker
            order_result = self.broker.submit_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type='market',
                price=entry_price_formatted,
                stop_loss=stop_loss_formatted,
                take_profit=take_profit_formatted
            )
            
            if order_result and 'order_id' in order_result:
                self.log_status(f"✅ Order submitted successfully: ID {order_result['order_id']}")
                
                # Store the current trade info
                self.current_trade = {
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'order_id': order_result['order_id'],
                    'entry_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                return True
            else:
                self.log_status(f"❌ Failed to submit order: {order_result}")
                return False
                
        except Exception as e:
            self.log_status(f"❌ Error executing trade: {str(e)}")
            logger.error(f"Trade execution error: {e}", exc_info=True)
            return False
            
    def check_exit_conditions(self, positions):
        """Check if we should exit any positions based on Bollinger Bands"""
        try:
            if not positions:
                return
                
            # Get current price action
            symbol = self.parameters.get('symbol')
            price_action = self.get_price_action(symbol)
            
            if not price_action:
                return
                
            current_price = price_action['current_price']
            bb_middle = price_action['bb_middle']
            
            for position in positions:
                position_id = position.get('id')
                side = position.get('side', '').lower()
                
                # Exit long positions when price crosses above middle band
                if side == 'buy' and current_price >= bb_middle:
                    self.log_status(f"🔔 EXIT SIGNAL for LONG position: Price crossed above middle band")
                    self.exit_trade(position_id)
                    
                # Exit short positions when price crosses below middle band
                elif side == 'sell' and current_price <= bb_middle:
                    self.log_status(f"🔔 EXIT SIGNAL for SHORT position: Price crossed below middle band")
                    self.exit_trade(position_id)
                    
        except Exception as e:
            self.log_status(f"❌ Error checking exit conditions: {str(e)}")
            logger.error(f"Exit condition error: {e}", exc_info=True)
            
    def exit_trade(self, position_id):
        """Exit a trade by position ID"""
        try:
            self.log_status(f"🛑 Closing position {position_id}")
            
            # Close the position through the broker
            result = self.broker.close_position(position_id)
            
            if result and result.get('success'):
                self.log_status(f"✅ Position closed successfully")
                
                # Clear current trade info
                self.current_trade = None
                
                # Update performance metrics if needed
                if 'profit_loss' in result:
                    profit_loss = result['profit_loss']
                    self.update_performance(profit_loss)
                    
                return True
            else:
                self.log_status(f"❌ Failed to close position: {result}")
                return False
                
        except Exception as e:
            self.log_status(f"❌ Error exiting trade: {str(e)}")
            logger.error(f"Trade exit error: {e}", exc_info=True)
            return False
            
    def update_performance(self, profit_loss):
        """Update performance metrics after a trade"""
        try:
            # Increment total trades
            self.performance['total_trades'] += 1
            
            # Update profit/loss metrics
            self.performance['total_profit_loss'] += profit_loss
            
            # Update win/loss counts
            if profit_loss > 0:
                self.performance['winning_trades'] += 1
                self.performance['largest_win'] = max(self.performance['largest_win'], profit_loss)
            else:
                self.performance['losing_trades'] += 1
                self.performance['largest_loss'] = min(self.performance['largest_loss'], profit_loss)
                
            # Log the performance update
            win_rate = 0
            if self.performance['total_trades'] > 0:
                win_rate = (self.performance['winning_trades'] / self.performance['total_trades']) * 100
                
            self.log_status(
                f"📈 Performance updated:\n"
                f"   Total Trades: {self.performance['total_trades']}\n"
                f"   Win Rate: {win_rate:.1f}%\n"
                f"   Total P/L: {self.performance['total_profit_loss']:.2f}"
            )
            
        except Exception as e:
            logger.error(f"Error updating performance: {e}")
            
    def format_price(self, symbol, price):
        """Format price according to instrument precision"""
        try:
            # Specific handling for common forex pairs
            if symbol == 'USD_JPY' or symbol == 'EUR_JPY' or 'JPY' in symbol:
                return round(price, 3)  # JPY pairs typically use 3 decimal places
            
            # Check for forex pairs (they contain an underscore and currency codes)
            elif '_' in symbol and any(curr in symbol for curr in ['EUR', 'GBP', 'USD', 'AUD', 'CAD', 'CHF', 'NZD']):
                # Make sure it's not a crypto or index that happens to have USD in the name
                if not any(asset in symbol for asset in ['BTC', 'SPX', 'NAS', 'XAU']):
                    return round(price, 5)  # 5 decimal places for forex pairs
            
            # Then check for specific instruments
            elif 'XAU' in symbol:
                return round(price, 2)  # 2 decimal places for Gold
            elif 'BTC' in symbol:
                return round(price, 1)  # 1 decimal place for Bitcoin
            elif 'SPX' in symbol or 'NAS' in symbol:
                return round(price, 1)  # 1 decimal place for indices
            else:
                return round(price, 2)  # Default to 2 decimal places for safety
                
        except Exception as e:
            logger.error(f"Error formatting price: {e}")
            return round(price, 2)  # Return rounded price if formatting fails

    # ... rest of the BB strategy implementation ... 