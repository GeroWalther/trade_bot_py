import logging
from datetime import datetime, timedelta
import threading
import time
import json
import aiohttp
import asyncio
import os
import pickle
import requests

logger = logging.getLogger(__name__)

class AIStrategy:
    def __init__(self, broker, parameters=None):
        logger.info(f"Initializing AI Strategy with parameters: {parameters}")
        
        self.broker = broker
        # Update available instruments to match the AI Analysis dropdown options
        self.available_instruments = [
            'EUR_USD',  # EUR/USD in analysis
            'USD_JPY',  # USD/JPY in analysis
            'BTC_USD',  # Bitcoin (BTC/USD) in analysis
            'SPX500_USD',  # S&P 500 in analysis
            'NAS100_USD',  # Nasdaq in analysis
            'XAU_USD'   # Gold in analysis
        ]
        
        # Map trading symbols to AI analysis asset names
        self.symbol_to_asset_map = {
            'EUR_USD': 'EUR/USD',  # This maps EUR_USD to EUR/USD in the analysis
            'USD_JPY': 'USD/JPY',  # This maps USD_JPY to USD/JPY in the analysis
            'BTC_USD': 'BTCUSD',   # This maps BTC_USD to BTCUSD in the analysis
            'SPX500_USD': 'S&P500', # This maps SPX500_USD to S&P500 in the analysis
            'NAS100_USD': 'Nasdaq', # This maps NAS100_USD to Nasdaq in the analysis
            'XAU_USD': 'Gold'       # This maps XAU_USD to Gold in the analysis
        }
        
        # Default parameters - make a deep copy to avoid reference issues
        self.parameters = {
            'symbol': 'XAU_USD',  # Default to Gold
            'quantity': 0.1,
            'check_interval': 3600,  # Check every hour by default
            'continue_after_trade': True,
            'max_concurrent_trades': 1,
            'trading_term': 'Day trade',  # Default trading term
            'risk_level': 'conservative',  # Default risk level
        }
        
        # Update with provided parameters if any
        if parameters:
            self.parameters.update(parameters)
        
        # Track the current symbol to detect changes
        self.current_symbol = self.parameters['symbol']
        
        logger.info(f"Final parameters after initialization: {self.parameters}")
        
        self._continue = False
        self.last_check_time = None
        self.status_updates = []
        self.trades_history = []
        
        # Define the cache directory and file path
        self.cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, f"ai_strategy_{self.parameters['symbol']}_performance.pkl")
        
        # Load performance data from cache if it exists
        self.performance = self.load_performance_from_cache() or {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit_loss': 0,
            'largest_win': 0,
            'largest_loss': 0,
            'current_drawdown': 0,
            'max_drawdown': 0,
            'daily_pl': 0.0,
            'last_update_date': datetime.now().strftime('%Y-%m-%d')
        }
        
        # Replace single current_trade with a dictionary of active trades
        self.active_trades = {}
        self.last_analysis = None
        self._thread = None
        self.ai_analysis = None
        self.last_analysis_time = None
        
        # Start a separate thread to periodically update and save performance metrics
        self._performance_thread = threading.Thread(target=self.performance_update_loop, daemon=True)
        self._performance_thread.start()

    def load_performance_from_cache(self):
        """Load performance data from cache file"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    performance_data = pickle.load(f)
                logger.info(f"Loaded performance data from cache: {performance_data}")
                return performance_data
        except Exception as e:
            logger.error(f"Error loading performance data from cache: {e}", exc_info=True)
        return None
        
    def save_performance_to_cache(self):
        """Save performance data to cache file"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.performance, f)
            logger.info(f"Saved performance data to cache: {self.performance}")
        except Exception as e:
            logger.error(f"Error saving performance data to cache: {e}", exc_info=True)
    
    def clear_performance_cache(self):
        """Clear the performance cache file"""
        try:
            if os.path.exists(self.cache_file):
                os.remove(self.cache_file)
                self.performance = {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'total_profit_loss': 0,
                    'largest_win': 0,
                    'largest_loss': 0,
                    'current_drawdown': 0,
                    'max_drawdown': 0,
                    'daily_pl': 0.0,
                    'last_update_date': datetime.now().strftime('%Y-%m-%d')
                }
                logger.info("Performance cache cleared")
                self.log_status("🧹 Performance metrics have been reset")
        except Exception as e:
            logger.error(f"Error clearing performance cache: {e}", exc_info=True)
    
    def performance_update_loop(self):
        """Background thread to update and save performance metrics periodically"""
        while True:
            try:
                # Check if we need to reset daily P/L (new day)
                today = datetime.now().strftime('%Y-%m-%d')
                if self.performance.get('last_update_date') != today:
                    self.log_status(f"📊 New trading day: Resetting daily P/L from €{self.performance.get('daily_pl', 0.0):.2f} to €0.00")
                    self.performance['daily_pl'] = 0.0
                    self.performance['last_update_date'] = today
                
                # Update real-time metrics if we have open positions
                if self.active_trades:
                    total_unrealized_pl = 0
                    
                    # Update each active trade
                    for trade_id, trade in list(self.active_trades.items()):
                        symbol = trade['symbol']
                        current_price = self.broker.get_last_price(symbol)
                        
                        if current_price:
                            try:
                                # Ensure values are floats before calculations
                                current_price = float(current_price)
                                entry_price = float(trade['entry_price'])
                                quantity = float(trade['quantity'])
                                
                                # Calculate unrealized P/L
                                if trade['side'] == 'BUY':
                                    unrealized_pl = (current_price - entry_price) * quantity
                                else:  # SELL
                                    unrealized_pl = (entry_price - current_price) * quantity
                                
                                # Update trade with unrealized P/L
                                trade['unrealized_pl'] = unrealized_pl
                                total_unrealized_pl += unrealized_pl
                                
                            except (ValueError, TypeError) as e:
                                logger.error(f"Error calculating unrealized P/L for trade {trade_id}: {e}")
                    
                    # Update total unrealized P/L in performance
                    self.performance['unrealized_pl'] = total_unrealized_pl
                else:
                    # No active trades, set unrealized P/L to 0
                    self.performance['unrealized_pl'] = 0.0
                
                # Save performance data to cache
                self.save_performance_to_cache()
                
                # Sleep for 30 seconds before the next update
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in performance update loop: {e}", exc_info=True)
                time.sleep(60)  # Sleep for a minute if there's an error

    async def get_ai_analysis(self):
        """Get AI analysis for the configured asset, term, and risk level"""
        try:
            # Get the asset name for AI analysis from the symbol mapping
            asset_name = self.symbol_to_asset_map.get(self.parameters['symbol'], 'Gold')
            trading_term = self.parameters['trading_term']
            risk_level = self.parameters['risk_level']
            
            logger.info(f"Requesting AI analysis for {asset_name} ({trading_term}, {risk_level})")
            self.log_status(f"🔍 Requesting AI analysis for {asset_name} ({trading_term}, {risk_level})")
            
            async with aiohttp.ClientSession() as session:
                url = "http://localhost:5005/api/advanced-market-analysis"
                payload = {
                    "asset": asset_name,
                    "term": trading_term,
                    "riskLevel": risk_level
                }
                
                async with session.post(url, json=payload, timeout=60) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result.get('status') == 'success' and result.get('data'):
                            self.ai_analysis = result['data']
                            self.last_analysis_time = datetime.now()
                            
                            # Log the analysis summary - simplified to just one line
                            self.log_status(f"✅ Received AI analysis for {asset_name}")
                            
                            # Return the analysis
                            return self.ai_analysis
                        else:
                            error_msg = result.get('message', 'Unknown error')
                            self.log_status(f"❌ AI analysis failed: {error_msg}")
                            return None
                    else:
                        self.log_status(f"❌ AI analysis request failed with status {response.status}")
                        return None
                        
        except Exception as e:
            self.log_status(f"❌ Error getting AI analysis: {str(e)}")
            logger.error(f"Error getting AI analysis: {e}", exc_info=True)
            return None

    def should_enter_trade(self):
        """Determine if we should enter a trade based on AI analysis"""
        try:
            if not self.ai_analysis:
                self.log_status("❌ No AI analysis available")
                return False
                
            # Get current positions for the symbol
            symbol = self.parameters['symbol']
            
            # Get all positions from the broker
            current_positions = self.broker.get_tracked_positions()
            
            # Filter positions for the current symbol
            symbol_positions = []
            for pos_id, pos in current_positions.items():
                if pos['symbol'] == symbol:
                    symbol_positions.append(pos)
            
            # Check if we've reached max concurrent trades
            if len(symbol_positions) >= self.parameters['max_concurrent_trades']:
                self.log_status(f"ℹ️ Maximum concurrent trades reached ({len(symbol_positions)}/{self.parameters['max_concurrent_trades']}), skipping entry signal check")
                return False
                
            # Get current price
            current_price = self.broker.get_last_price(symbol)
            if not current_price:
                self.log_status(f"❌ Could not get current price for {symbol}")
                return False
                
            # Ensure current_price is a float
            try:
                current_price = float(current_price)
            except (ValueError, TypeError):
                self.log_status(f"❌ Invalid current price format: {current_price}")
                return False
                
            # Get AI recommendation
            trading_strategy = self.ai_analysis['trading_strategy']
            direction = trading_strategy['direction']
            entry_price = trading_strategy['entry']['price']
            
            # Try to parse the entry price (it might be a string with commas or a range)
            try:
                if isinstance(entry_price, str):
                    # Handle price ranges like "2890-2900" or "2890 - 2900"
                    if "-" in entry_price:
                        # Remove spaces around the hyphen
                        entry_price_clean = entry_price.replace(" ", "")
                        # Split by hyphen
                        price_parts = entry_price_clean.split("-")
                        if len(price_parts) == 2:
                            # Try to get both prices
                            try:
                                price1 = float(price_parts[0].replace(',', ''))
                                price2 = float(price_parts[1].replace(',', ''))
                                # Take the middle value and round down
                                entry_price = int((price1 + price2) / 2)
                                # Comment out this line to avoid duplicate log message
                                # self.log_status(f"📊 Price range detected: {entry_price_clean}, using middle value: {entry_price}")
                            except (ValueError, TypeError):
                                # If conversion fails, just use the first part
                                entry_price = float(price_parts[0].replace(',', ''))
                                self.log_status(f"📊 Price range detected: {entry_price_clean}, using first value: {entry_price}")
                        else:
                            # Just use the first part if splitting doesn't work as expected
                            entry_price = float(price_parts[0].replace(',', ''))
                    else:
                        # No range, just convert to float
                        entry_price = float(entry_price.replace(',', ''))
                else:
                    entry_price = float(entry_price)
            except (ValueError, TypeError):
                self.log_status(f"❌ Invalid entry price format: {entry_price}")
                return False
                
            # Get take profit and stop loss values
            try:
                take_profit_str = trading_strategy['take_profit_1']['price']
                if isinstance(take_profit_str, str):
                    # Handle price ranges like "2890-2900" or "2890 - 2900"
                    if "-" in take_profit_str:
                        # Remove spaces around the hyphen
                        tp_clean = take_profit_str.replace(" ", "")
                        # Split by hyphen
                        price_parts = tp_clean.split("-")
                        if len(price_parts) == 2:
                            # Try to get both prices
                            try:
                                price1 = float(price_parts[0].replace(',', ''))
                                price2 = float(price_parts[1].replace(',', ''))
                                # Take the middle value and round down
                                take_profit = int((price1 + price2) / 2)
                                self.log_status(f"📊 TP range detected: {tp_clean}, using middle value: {take_profit}")
                            except (ValueError, TypeError):
                                # If conversion fails, just use the first part
                                take_profit = float(price_parts[0].replace(',', ''))
                        else:
                            # Just use the first part if splitting doesn't work as expected
                            take_profit = float(price_parts[0].replace(',', ''))
                    else:
                        # No range, just convert to float
                        take_profit = float(take_profit_str.replace(',', ''))
                else:
                    take_profit = float(take_profit_str)
                
                stop_loss_str = trading_strategy['stop_loss']['price']
                if isinstance(stop_loss_str, str):
                    # Handle price ranges like "2890-2900" or "2890 - 2900"
                    if "-" in stop_loss_str:
                        # Remove spaces around the hyphen
                        sl_clean = stop_loss_str.replace(" ", "")
                        # Split by hyphen
                        price_parts = sl_clean.split("-")
                        if len(price_parts) == 2:
                            # Try to get both prices
                            try:
                                price1 = float(price_parts[0].replace(',', ''))
                                price2 = float(price_parts[1].replace(',', ''))
                                # Take the middle value and round down
                                stop_loss = int((price1 + price2) / 2)
                                self.log_status(f"📊 SL range detected: {sl_clean}, using middle value: {stop_loss}")
                            except (ValueError, TypeError):
                                # If conversion fails, just use the first part
                                stop_loss = float(price_parts[0].replace(',', ''))
                        else:
                            # Just use the first part if splitting doesn't work as expected
                            stop_loss = float(price_parts[0].replace(',', ''))
                    else:
                        # No range, just convert to float
                        stop_loss = float(stop_loss_str.replace(',', ''))
                else:
                    stop_loss = float(stop_loss_str)
            except (ValueError, TypeError, KeyError):
                self.log_status("⚠️ Could not parse take profit or stop loss values, proceeding without them")
                take_profit = None
                stop_loss = None
                
            # Ensure both values are floats before calculation
            try:
                # Convert to float if they aren't already
                current_price = float(current_price)
                entry_price = float(entry_price)
                
                # Format prices with proper precision
                current_price = self.format_price(symbol, current_price)
                entry_price = self.format_price(symbol, entry_price)
                
                # Format take profit and stop loss if they exist
                if take_profit is not None:
                    take_profit = self.format_price(symbol, take_profit)
                if stop_loss is not None:
                    stop_loss = self.format_price(symbol, stop_loss)
                
                # Calculate price difference percentage
                price_diff_pct = abs(current_price - entry_price) / entry_price * 100
                
                # Log the price difference but don't use it as a condition to prevent trade entry
                self.log_status(f"ℹ️ Current price ({current_price:.2f}) is {price_diff_pct:.2f}% away from recommended entry ({entry_price:.2f})")
                
            except (TypeError, ValueError) as e:
                self.log_status(f"❌ Error calculating price difference: {str(e)}")
                return False
                
            # Determine if we should enter based on the AI direction
            should_enter = direction in ["LONG", "SHORT"]
            
            if should_enter:
                # Create a comprehensive log message with all trade details
                tp_sl_info = ""
                if take_profit and stop_loss:
                    tp_sl_info = f" with Take Profit @{take_profit:.2f}, Stop Loss @{stop_loss:.2f}"
                
                if direction == "LONG":
                    self.log_status(f"🚀 AI recommends LONG position - Going LONG @{entry_price:.2f} on {symbol}{tp_sl_info}")
                else:
                    self.log_status(f"🔻 AI recommends SHORT position - Going SHORT @{entry_price:.2f} on {symbol}{tp_sl_info}")
            
            return should_enter
            
        except Exception as e:
            self.log_status(f"❌ Error in should_enter_trade: {str(e)}")
            logger.error(f"Error in should_enter_trade: {e}", exc_info=True)
            return False

    def execute_trade(self):
        """Execute trade based on AI strategy"""
        try:
            if not self.ai_analysis:
                self.log_status("❌ Cannot execute trade - No AI analysis available")
                return False
                
            # Use the symbol from parameters - this is the correct trading symbol
            symbol = self.parameters['symbol']
            quantity = self.parameters['quantity']
            risk_percent = self.parameters.get('risk_percent', None)
            
            # Get current positions for the symbol
            current_positions = self.broker.get_tracked_positions()
            
            # Filter positions for the current symbol
            symbol_positions = []
            for pos_id, pos in current_positions.items():
                if pos['symbol'] == symbol:
                    symbol_positions.append(pos)
            
            # Check if we've reached max concurrent trades
            if len(symbol_positions) >= self.parameters['max_concurrent_trades']:
                self.log_status(f"❌ Maximum concurrent trades reached ({len(symbol_positions)}/{self.parameters['max_concurrent_trades']}), cannot open new position")
                return False
            
            # Get current price
            current_price = self.broker.get_last_price(symbol)
            if not current_price:
                self.log_status(f"❌ Cannot execute trade - No price data for {symbol}")
                return False
                
            # Ensure current_price is a float
            try:
                current_price = float(current_price)
            except (ValueError, TypeError):
                self.log_status(f"❌ Invalid current price format: {current_price}")
                return False
                
            # Get AI recommendation
            trading_strategy = self.ai_analysis['trading_strategy']
            direction = trading_strategy['direction']
            
            # Parse entry price from AI recommendation
            try:
                entry_price_str = trading_strategy['entry']['price']
                if isinstance(entry_price_str, str):
                    # Handle price ranges like "2890-2900" or "2890 - 2900"
                    if "-" in entry_price_str:
                        # Remove spaces around the hyphen
                        entry_price_clean = entry_price_str.replace(" ", "")
                        # Split by hyphen
                        price_parts = entry_price_clean.split("-")
                        if len(price_parts) == 2:
                            # Try to get both prices
                            try:
                                price1 = float(price_parts[0].replace(',', ''))
                                price2 = float(price_parts[1].replace(',', ''))
                                # Take the middle value and round down
                                entry_price = int((price1 + price2) / 2)
                                # Comment out this line to avoid duplicate log message
                                # self.log_status(f"📊 Price range detected: {entry_price_clean}, using middle value: {entry_price}")
                            except (ValueError, TypeError):
                                # If conversion fails, just use the first part
                                entry_price = float(price_parts[0].replace(',', ''))
                                self.log_status(f"📊 Price range detected: {entry_price_clean}, using first value: {entry_price}")
                        else:
                            # Just use the first part if splitting doesn't work as expected
                            entry_price = float(price_parts[0].replace(',', ''))
                    else:
                        # No range, just convert to float
                        entry_price = float(entry_price_str.replace(',', ''))
                else:
                    entry_price = float(entry_price_str)
            except (ValueError, TypeError):
                self.log_status(f"❌ Invalid entry price format: {trading_strategy['entry']['price']}")
                return False
            
            # Determine trade direction
            if direction == "LONG":
                side = 'buy'
            elif direction == "SHORT":
                side = 'sell'
            else:
                self.log_status(f"❌ AI does not recommend a trade direction: {direction}")
                return False
            
            # Prepare order - use the AI-recommended entry price instead of current price
            order = {
                'symbol': symbol,  # This is the correct symbol from parameters
                'quantity': quantity,
                'side': side
            }
            
            # Always use a pending order with the AI-recommended entry price
            order['order_type'] = 'pending'
            order['price'] = entry_price
            self.log_status(f"📈 Using pending order with entry price: {entry_price:.2f} (current: {current_price:.2f})")
            
            # Parse take profit and stop loss from AI recommendation before submitting the order
            try:
                take_profit = trading_strategy['take_profit_1']['price']
                
                if isinstance(take_profit, str):
                    # Handle price ranges like "2890-2900" or "2890 - 2900"
                    if "-" in take_profit:
                        # Remove spaces around the hyphen
                        tp_clean = take_profit.replace(" ", "")
                        # Split by hyphen
                        price_parts = tp_clean.split("-")
                        if len(price_parts) == 2:
                            # Try to get both prices
                            try:
                                price1 = float(price_parts[0].replace(',', ''))
                                price2 = float(price_parts[1].replace(',', ''))
                                # Take the middle value and round down
                                take_profit = int((price1 + price2) / 2)
                                self.log_status(f"📊 TP range detected: {tp_clean}, using middle value: {take_profit}")
                            except (ValueError, TypeError):
                                # If conversion fails, just use the first part
                                take_profit = float(price_parts[0].replace(',', ''))
                        else:
                            # Just use the first part if splitting doesn't work as expected
                            take_profit = float(price_parts[0].replace(',', ''))
                    else:
                        # No range, just convert to float
                        take_profit = float(take_profit.replace(',', ''))
                else:
                    take_profit = float(take_profit)
                
                stop_loss_str = trading_strategy['stop_loss']['price']
                
                if isinstance(stop_loss_str, str):
                    # Handle price ranges like "2890-2900" or "2890 - 2900"
                    if "-" in stop_loss_str:
                        # Remove spaces around the hyphen
                        sl_clean = stop_loss_str.replace(" ", "")
                        # Split by hyphen
                        price_parts = sl_clean.split("-")
                        if len(price_parts) == 2:
                            # Try to get both prices
                            try:
                                price1 = float(price_parts[0].replace(',', ''))
                                price2 = float(price_parts[1].replace(',', ''))
                                # Take the middle value and round down
                                stop_loss = int((price1 + price2) / 2)
                                self.log_status(f"📊 SL range detected: {sl_clean}, using middle value: {stop_loss}")
                            except (ValueError, TypeError):
                                # If conversion fails, just use the first part
                                stop_loss = float(price_parts[0].replace(',', ''))
                        else:
                            # Just use the first part if splitting doesn't work as expected
                            stop_loss = float(price_parts[0].replace(',', ''))
                    else:
                        # No range, just convert to float
                        stop_loss = float(stop_loss_str.replace(',', ''))
                else:
                    stop_loss = float(stop_loss_str)
                    
                # Calculate position size based on risk percentage if enabled
                if risk_percent is not None and stop_loss is not None:
                    try:
                        # Get account balance
                        cash, _, total_value = self.broker._get_balances_at_broker()
                        
                        # Calculate risk amount in account currency
                        risk_amount = total_value * (risk_percent / 100)
                        
                        # Calculate price difference between entry and stop loss
                        price_diff = abs(entry_price - stop_loss)
                        
                        if price_diff > 0:
                            # Calculate position size based on risk
                            risk_based_quantity = risk_amount / price_diff
                            
                            # Format quantity according to instrument requirements
                            risk_based_quantity = self.format_quantity(symbol, risk_based_quantity)
                            
                            # Log the formatted quantity
                            self.log_status(f"🔄 Adjusted {symbol} quantity to match instrument requirements: {risk_based_quantity}")
                            
                            self.log_status(f"💹 Risk-based position sizing: {risk_percent}% risk = {risk_amount:.2f} currency units")
                            self.log_status(f"💹 Price difference to stop loss: {price_diff:.2f}, calculated quantity: {risk_based_quantity}")
                            
                            # Update quantity with risk-based calculation
                            quantity = risk_based_quantity
                            self.log_status(f"💹 Using risk-based position size: {quantity}")
                            
                            # Update the order quantity immediately
                            order['quantity'] = quantity
                        else:
                            self.log_status("⚠️ Cannot calculate risk-based position size: Entry price and stop loss are too close")
                    except Exception as e:
                        self.log_status(f"⚠️ Error calculating risk-based position size: {str(e)}")
                        self.log_status("⚠️ Using default quantity from parameters")
                
                # Ensure the order has the latest quantity value
                order['quantity'] = quantity
                
            except (ValueError, TypeError, KeyError) as e:
                self.log_status(f"⚠️ Could not parse take profit or stop loss values: {str(e)}, proceeding without them")
                take_profit = None
                stop_loss = None
            
            # Ensure proper quantity precision for all instruments before submitting
            try:
                # Apply instrument-specific precision rules using the format_quantity method
                order['quantity'] = self.format_quantity(symbol, order['quantity'])
                self.log_status(f"🔄 Final quantity after precision adjustment: {order['quantity']}")
                
                # Format entry price, take profit, and stop loss with proper precision
                entry_price = self.format_price(symbol, entry_price)
                if take_profit is not None:
                    take_profit = self.format_price(symbol, take_profit)
                    order['take_profit'] = take_profit
                    self.log_status(f"💰 Adding Take Profit: {take_profit:.2f}")
                if stop_loss is not None:
                    stop_loss = self.format_price(symbol, stop_loss)
                    order['stop_loss'] = stop_loss
                    self.log_status(f"🛑 Adding Stop Loss: {stop_loss:.2f}")
                
                # Update the entry price in the order
                if 'entry_price' in order:
                    order['entry_price'] = entry_price
            except Exception as e:
                self.log_status(f"⚠️ Error adjusting precision: {str(e)}")
            
            # Log the complete order details with the correct symbol
            self.log_status(f"📋 Submitting order: {order}")
            
            # Submit the order
            try:
                order_id = self.broker.submit_order(order)
                
                if order_id:
                    # Record the trade with the correct symbol
                    new_trade = {
                        'order_id': order_id,
                        'symbol': symbol,  # This is the correct symbol from parameters
                        'quantity': quantity,
                        'entry_price': entry_price,  # Use AI-recommended entry price
                        'entry_time': self.broker.get_time(),
                        'side': side.upper(),
                        'take_profit': take_profit,
                        'stop_loss': stop_loss,
                        'ai_analysis': self.ai_analysis
                    }
                    
                    # Add the trade to active trades
                    self.active_trades[order_id] = new_trade
                    
                    # Determine the order type for the log message
                    if order.get('order_type') == 'pending':
                        # For pending orders, determine if it's a limit or stop order
                        if side == 'buy':
                            # Buy Limit if entry price is below current price, Buy Stop if above
                            order_type_str = "Buy Limit" if entry_price < current_price else "Buy Stop"
                        else:  # sell
                            # Sell Limit if entry price is above current price, Sell Stop if below
                            order_type_str = "Sell Limit" if entry_price > current_price else "Sell Stop"
                        
                        self.log_status(f"✅ {order_type_str} Pending Order placed successfully at {entry_price:.2f}")
                    else:
                        # For market orders
                        self.log_status(f"✅ Market Order opened successfully at {entry_price:.2f}")
                    
                    # If Continue After Trade is set to No, stop the bot immediately after entering a trade
                    if not self.parameters.get('continue_after_trade', True):
                        self.log_status("🛑 Continue After Trade is set to No, stopping the bot")
                        self.stop()
                    
                    return True
                else:
                    self.log_status("❌ Order submission failed")
            except Exception as e:
                self.log_status(f"❌ Error executing trade: {str(e)}")
                
                # Check for specific error types
                error_str = str(e).lower()
                if "precision" in error_str:
                    self.log_status("⚠️ Precision error detected. Try adjusting the quantity to match instrument requirements.")
                elif "insufficient margin" in error_str:
                    self.log_status("⚠️ Insufficient margin available for this trade.")
                elif "market halted" in error_str:
                    self.log_status("⚠️ Market is currently halted or closed.")
                
                return False
            
        except Exception as e:
            self.log_status(f"❌ Error executing trade: {str(e)}")
            logger.error(f"Error executing trade: {e}", exc_info=True)
            return False

    def check_exit_conditions(self):
        """Check if we should exit any open trades"""
        try:
            # If no active trades, nothing to check
            if not self.active_trades:
                return False
            
            # Track trades to exit
            trades_to_exit = []
            
            # Check each active trade
            for trade_id, trade in self.active_trades.items():
                symbol = trade['symbol']
                current_price = self.broker.get_last_price(symbol)
                
                if not current_price:
                    self.log_status(f"❌ Could not get current price for {symbol}, skipping exit check")
                    continue
                
                # Ensure current_price is a float
                try:
                    current_price = float(current_price)
                except (ValueError, TypeError):
                    self.log_status(f"❌ Invalid current price format: {current_price}")
                    continue
                
                # Get trade details
                entry_price = trade['entry_price']
                side = trade['side']
                take_profit = trade.get('take_profit')
                stop_loss = trade.get('stop_loss')
                
                # Check if we should exit based on take profit or stop loss
                should_exit = False
                exit_reason = ""
                
                if side == 'BUY':
                    # For long positions
                    if take_profit and current_price >= take_profit:
                        should_exit = True
                        exit_reason = f"Take profit reached: {current_price:.2f} >= {take_profit:.2f}"
                    elif stop_loss and current_price <= stop_loss:
                        should_exit = True
                        exit_reason = f"Stop loss triggered: {current_price:.2f} <= {stop_loss:.2f}"
                else:  # SELL
                    # For short positions
                    if take_profit and current_price <= take_profit:
                        should_exit = True
                        exit_reason = f"Take profit reached: {current_price:.2f} <= {take_profit:.2f}"
                    elif stop_loss and current_price >= stop_loss:
                        should_exit = True
                        exit_reason = f"Stop loss triggered: {current_price:.2f} >= {stop_loss:.2f}"
                
                # If we should exit this trade, add it to the list
                if should_exit:
                    trades_to_exit.append((trade_id, exit_reason))
            
            # If we have trades to exit, log and return True
            if trades_to_exit:
                for trade_id, reason in trades_to_exit:
                    self.log_status(f"🔄 Exit signal for trade {trade_id}: {reason}")
                return True
            
            return False
            
        except Exception as e:
            self.log_status(f"❌ Error checking exit conditions: {str(e)}")
            logger.error(f"Error checking exit conditions: {e}", exc_info=True)
            return False

    def exit_trade(self):
        """Exit all active trades"""
        try:
            # If no active trades, nothing to exit
            if not self.active_trades:
                self.log_status("ℹ️ No active trades to exit")
                return False
            
            exit_results = []
            
            # Exit each active trade
            for trade_id, trade in list(self.active_trades.items()):
                symbol = trade['symbol']
                side = trade['side']
                quantity = trade['quantity']
                entry_price = trade['entry_price']
                entry_time = trade['entry_time']
                
                # Get current price
                current_price = self.broker.get_last_price(symbol)
                if not current_price:
                    self.log_status(f"❌ Could not get current price for {symbol}, cannot exit trade")
                    continue
                
                # Ensure current_price is a float
                try:
                    current_price = float(current_price)
                except (ValueError, TypeError):
                    self.log_status(f"❌ Invalid current price format: {current_price}")
                    continue
                
                # Calculate profit/loss
                if side == 'BUY':
                    profit_loss = (current_price - entry_price) * quantity
                    profit_pct = (current_price - entry_price) / entry_price * 100
                else:  # SELL
                    profit_loss = (entry_price - current_price) * quantity
                    profit_pct = (entry_price - current_price) / entry_price * 100
                
                # Create exit order (opposite of entry)
                exit_side = 'sell' if side == 'BUY' else 'buy'
                exit_order = {
                    'symbol': symbol,
                    'quantity': quantity,
                    'side': exit_side,
                    'order_type': 'market'
                }
                
                # Submit exit order
                try:
                    exit_order_id = self.broker.submit_order(exit_order)
                    
                    if exit_order_id:
                        # Record trade result
                        trade_result = {
                            'symbol': symbol,
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'quantity': quantity,
                            'side': side,
                            'entry_time': entry_time,
                            'exit_time': self.broker.get_time(),
                            'profit_loss': profit_loss,
                            'profit_pct': profit_pct
                        }
                        
                        # Add to trade history
                        self.trades_history.append(trade_result)
                        
                        # Update performance metrics
                        self.update_performance(trade_result)
                        
                        # Log exit
                        profit_emoji = "📈" if profit_loss > 0 else "📉"
                        self.log_status(
                            f"{profit_emoji} Exited {side} position on {symbol} at {current_price:.2f} "
                            f"(Entry: {entry_price:.2f}, P/L: {profit_loss:.2f} EUR, {profit_pct:.2f}%)"
                        )
                        
                        # Remove from active trades
                        del self.active_trades[trade_id]
                        
                        exit_results.append(True)
                    else:
                        self.log_status(f"❌ Failed to exit trade {trade_id}")
                        exit_results.append(False)
                        
                except Exception as e:
                    self.log_status(f"❌ Error exiting trade {trade_id}: {str(e)}")
                    exit_results.append(False)
            
            # Return True if all trades were exited successfully
            return all(exit_results) if exit_results else False
            
        except Exception as e:
            self.log_status(f"❌ Error exiting trades: {str(e)}")
            logger.error(f"Error exiting trades: {e}", exc_info=True)
            return False

    def run(self):
        """Main strategy loop"""
        self.log_status(
            f"🟢 AI Strategy activated:\n"
            f"   Asset: {self.parameters['symbol']}\n"
            f"   Term: {self.parameters['trading_term']}\n"
            f"   Risk Level: {self.parameters['risk_level']}\n"
            f"   Check Interval: {self.parameters['check_interval']}s\n"
            f"   Continue After Trade: {'Yes' if self.parameters['continue_after_trade'] else 'No'}\n"
            f"   Trailing Stop Loss: {'Enabled' if self.parameters.get('trailing_stop_loss', False) else 'Disabled'}"
        )
        
        while self._continue:
            try:
                # Run the check_and_trade method
                asyncio.run(self.check_and_trade())
                
                # Sleep for the check interval
                time.sleep(self.parameters['check_interval'])
            except Exception as e:
                self.log_status(f"❌ Error in strategy loop: {str(e)}")
                logger.error(f"Error in strategy loop: {e}", exc_info=True)
                time.sleep(60)  # Sleep for a minute if there's an error

    async def check_and_trade(self):
        """Main trading logic"""
        try:
            # Check if the symbol has changed
            if self.current_symbol != self.parameters['symbol']:
                self.log_status(f"🔄 Symbol changed from {self.current_symbol} to {self.parameters['symbol']}, requesting new analysis")
                # Update the tracked symbol
                self.current_symbol = self.parameters['symbol']
                # Reset the analysis to force a new request
                self.ai_analysis = None
                self.last_analysis_time = None
                
                # Update the cache file path for the new symbol
                self.cache_file = os.path.join(self.cache_dir, f"ai_strategy_{self.parameters['symbol']}_performance.pkl")
                # Load performance data for the new symbol
                new_performance = self.load_performance_from_cache()
                if new_performance:
                    self.performance = new_performance
            
            # Check if we need to get a new AI analysis
            current_time = datetime.now()
            analysis_age_hours = 0
            
            if self.last_analysis_time:
                analysis_age = current_time - self.last_analysis_time
                analysis_age_hours = analysis_age.total_seconds() / 3600
            
            # Get new analysis if we don't have one or if it's older than 4 hours
            if not self.ai_analysis or analysis_age_hours > 4:
                # Get the asset name for display - this is now correctly using the current symbol
                asset_name = self.symbol_to_asset_map.get(self.parameters['symbol'], 'Gold')
                trading_term = self.parameters['trading_term']
                risk_level = self.parameters['risk_level']
                
                # Request new AI analysis with the correct asset name
                self.ai_analysis = await self.get_ai_analysis()
                if not self.ai_analysis:
                    self.log_status("❌ Failed to get AI analysis, skipping trading cycle")
                    return
            
            # Get current positions from the broker
            current_positions = self.broker.get_tracked_positions()
            
            # Filter positions for the current symbol
            symbol = self.parameters['symbol']
            symbol_positions = []
            for pos_id, pos in current_positions.items():
                if pos['symbol'] == symbol:
                    symbol_positions.append(pos)
            
            # Log the number of active positions
            self.log_status(f"ℹ️ Current active positions for {symbol}: {len(symbol_positions)}/{self.parameters['max_concurrent_trades']}")
            
            # Check if we have any active trades to exit
            if self.active_trades:
                # Check if we should exit any trades
                if self.check_exit_conditions():
                    self.exit_trade()
                    
                    # If Continue After Trade is set to No, stop the bot
                    if not self.parameters.get('continue_after_trade', True):
                        self.log_status("🛑 Continue After Trade is set to No, stopping the bot")
                        self.stop()
                        # Clear analysis data
                        self.ai_analysis = None
                        self.last_analysis_time = None
                        return
            
            # Check if we can enter a new trade (if we haven't reached max_concurrent_trades)
            if len(symbol_positions) < self.parameters['max_concurrent_trades']:
                # Check if we should enter a trade
                if self.should_enter_trade():
                    self.execute_trade()
                    # The execute_trade method now handles stopping if Continue After Trade is No
            else:
                self.log_status(f"ℹ️ Maximum concurrent trades reached ({len(symbol_positions)}/{self.parameters['max_concurrent_trades']}), waiting for positions to close")

        except Exception as e:
            self.log_status(f"❌ Error in trading logic: {str(e)}")
            logger.error(f"Trading error: {e}", exc_info=True)

    def start(self):
        """Start the strategy"""
        if self._continue:
            self.log_status("ℹ️ Strategy is already running")
            return
            
        self._continue = True
        self.log_status("🟢 Strategy started")
        
        # Start the strategy in a new thread
        self._thread = threading.Thread(target=self.run)
        self._thread.daemon = True
        self._thread.start()

    def stop(self):
        """Stop the strategy"""
        if not self._continue:
            self.log_status("ℹ️ Strategy is already stopped")
            return
            
        self.log_status("🛑 Strategy stopped")
        # Set the continue flag to False first
        self._continue = False
        # Log that we're stopping the strategy
        logger.info("AI Strategy stopping - setting _continue to False")
        # Clear data but don't clear logs
        self.clear_data()
        # Add a clear separator in the logs to indicate the bot has stopped
        self.log_status("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        # Log that the strategy has been stopped
        logger.info("AI Strategy has been stopped")
        # Double-check that the continue flag is False
        if self._continue:
            logger.error("Failed to set _continue to False")
            self._continue = False

    def clear_data(self):
        """Clear all strategy data except status updates"""
        self.ai_analysis = None
        self.last_analysis_time = None
        self.active_trades = {}  # Clear all active trades
        # Never clear status updates
        # if not self._continue:
        #     self.status_updates = []
        # Keep performance data but reset current trade data
        if hasattr(self, 'performance'):
            self.performance['current_trade'] = None

    def should_continue(self):
        """Check if strategy should continue running"""
        return self._continue

    def get_status(self):
        """Get current strategy status"""
        # Get the asset name for display
        asset_name = self.symbol_to_asset_map.get(self.parameters['symbol'], 'Gold')
        trading_term = self.parameters['trading_term']
        risk_level = self.parameters['risk_level']
        
        strategy_info = {
            'name': 'AI-Driven Trading Strategy',
            'period': f"{trading_term} ({risk_level})",
            'description': f"Uses AI analysis to trade {asset_name} with {risk_level} risk profile",
            'rules': [
                f"Requests AI analysis for {asset_name} with {trading_term} timeframe",
                "Follows AI-recommended direction (LONG/SHORT)",
                "Uses AI-recommended entry, take profit, and stop loss levels",
                "Refreshes analysis every 4 hours",
                "Monitors positions for take profit and stop loss conditions",
                f"Manages up to {self.parameters['max_concurrent_trades']} concurrent trades"
            ]
        }
        
        # Get current positions info
        current_positions = []
        for trade_id, trade in self.active_trades.items():
            symbol = trade['symbol']
            current_price = self.broker.get_last_price(symbol)
            
            if current_price:
                entry_price = trade['entry_price']
                if trade['side'] == 'BUY':
                    profit_pct = (current_price - entry_price) / entry_price * 100
                else:
                    profit_pct = (entry_price - current_price) / entry_price * 100
                    
                current_positions.append({
                    'trade_id': trade_id,
                    'symbol': symbol,
                    'side': trade['side'],
                    'entry_price': entry_price,
                    'current_price': current_price,
                    'profit_pct': profit_pct,
                    'take_profit': trade['take_profit'],
                    'stop_loss': trade['stop_loss']
                })
        
        # Calculate win rate
        win_rate = 0.0
        if self.performance['total_trades'] > 0:
            win_rate = (self.performance['winning_trades'] / self.performance['total_trades']) * 100
        
        # Include performance metrics in the status
        performance_data = {
            **self.performance,
            'win_rate': win_rate
        }
        
        return {
            'name': f"AI {asset_name} Strategy",
            'running': self._continue,
            'parameters': self.parameters,
            'performance': performance_data,
            'recent_updates': self.status_updates,
            'available_instruments': self.available_instruments,
            'active_trades': current_positions,
            'active_trades_count': len(current_positions),
            'strategy_info': strategy_info,
            'last_analysis_time': self.last_analysis_time.strftime('%Y-%m-%d %H:%M:%S') if self.last_analysis_time else None
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

    def update_performance(self, trade_result):
        """Update performance metrics with a new trade result"""
        profit_loss = trade_result['profit_loss']
        
        self.performance['total_trades'] += 1
        self.performance['total_profit_loss'] += profit_loss
        
        # Update daily P/L
        self.performance['daily_pl'] = self.performance.get('daily_pl', 0.0) + profit_loss
        
        if profit_loss > 0:
            self.performance['winning_trades'] += 1
            if profit_loss > self.performance['largest_win']:
                self.performance['largest_win'] = profit_loss
        else:
            self.performance['losing_trades'] += 1
            if profit_loss < self.performance['largest_loss']:
                self.performance['largest_loss'] = profit_loss
                
        # Calculate drawdown
        self.calculate_drawdown()
        
        # Save updated performance to cache
        self.save_performance_to_cache()
        
        # Log performance update
        win_rate = 0.0
        if self.performance['total_trades'] > 0:
            win_rate = (self.performance['winning_trades'] / self.performance['total_trades']) * 100
            
        self.log_status(
            f"📊 Performance Update: Win Rate: {win_rate:.1f}%, "
            f"Total Trades: {self.performance['total_trades']}, "
            f"Daily P/L: €{self.performance['daily_pl']:.2f}"
        )

    def calculate_drawdown(self):
        """Calculate current and maximum drawdown"""
        if not self.trades_history:
            return
            
        # Calculate running balance
        balance = 0
        peak = 0
        drawdown = 0
        
        for trade in self.trades_history:
            balance += trade['profit_loss']
            if balance > peak:
                peak = balance
            
            current_drawdown = peak - balance
            if current_drawdown > drawdown:
                drawdown = current_drawdown
                
        self.performance['current_drawdown'] = peak - balance
        self.performance['max_drawdown'] = drawdown 

    def set_manual_analysis(self, symbol, direction, entry_price, take_profit, stop_loss):
        """Manually set AI analysis data for testing purposes"""
        self.log_status(f"🔧 Manually setting analysis data for {symbol}")
        
        # Create a simplified analysis structure
        self.ai_analysis = {
            "market_summary": f"Manual analysis for {symbol}",
            "key_drivers": ["Manual test"],
            "technical_analysis": "Manual technical analysis",
            "risk_assessment": "Manual risk assessment",
            "trading_strategy": {
                "direction": direction.upper(),
                "rationale": "Manual strategy",
                "entry": {
                    "price": str(entry_price),
                    "rationale": "Manual entry point"
                },
                "stop_loss": {
                    "price": str(stop_loss),
                    "rationale": "Manual stop loss"
                },
                "take_profit_1": {
                    "price": str(take_profit),
                    "rationale": "Manual take profit"
                },
                "take_profit_2": {
                    "price": str(take_profit * 0.9 if direction.upper() == "SHORT" else take_profit * 1.1),
                    "rationale": "Manual take profit 2"
                }
            }
        }
        
        # Update the last analysis time
        self.last_analysis_time = datetime.now()
        
        # Log the manual analysis
        self.log_status(f"✅ Manual analysis set for {symbol}: {direction.upper()} at {entry_price}")
        self.log_status(f"💰 Take Profit: {take_profit}, 🛑 Stop Loss: {stop_loss}")
        
        return True 

    def clear_logs(self):
        """Manually clear all status updates/logs"""
        self.status_updates = []
        self.log_status("🧹 Logs have been manually cleared")
        return True 

    def get_instrument_precision(self, symbol):
        """Get the required precision for a given instrument"""
        # Check for forex pairs first (they contain an underscore and currency codes)
        if '_' in symbol and any(curr in symbol for curr in ['EUR', 'GBP', 'USD', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD']):
            # Make sure it's not a crypto or index that happens to have USD in the name
            if not any(asset in symbol for asset in ['BTC', 'SPX', 'NAS', 'XAU']):
                return 4  # 4 decimal places for forex pairs
        
        # Then check for specific instruments
        if 'XAU' in symbol:
            return 0  # Whole numbers only for Gold
        elif 'BTC' in symbol:
            return 2  # 2 decimal places for Bitcoin
        elif 'SPX' in symbol or 'NAS' in symbol:
            return 1  # 1 decimal place for indices
        else:
            return 2  # Default to 2 decimal places for safety

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
            self.log_status(f"⚠️ Error formatting price: {str(e)}")
            # Return original price as fallback
            return price

    def get_min_quantity(self, symbol):
        """Get the minimum allowed quantity for a given instrument"""
        if 'XAU' in symbol:
            return 1  # Minimum 1 unit for Gold
        elif 'BTC' in symbol:
            return 0.01  # Minimum 0.01 for Bitcoin
        elif 'SPX' in symbol or 'NAS' in symbol:
            return 0.1  # Minimum 0.1 for indices
        elif '_' in symbol and any(curr in symbol for curr in ['EUR', 'GBP', 'USD', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD']):
            return 1  # Minimum 1 unit for forex pairs
        else:
            return 0.01  # Default minimum

    def format_quantity(self, symbol, quantity):
        """Format quantity according to instrument precision requirements"""
        try:
            precision = self.get_instrument_precision(symbol)
            min_quantity = self.get_min_quantity(symbol)
            
            if precision == 0:
                # For whole number instruments like XAU
                formatted_quantity = int(quantity)
            else:
                # For decimal precision instruments
                formatted_quantity = round(float(quantity), precision)
            
            # Ensure minimum quantity
            formatted_quantity = max(min_quantity, formatted_quantity)
            
            return formatted_quantity
        except Exception as e:
            self.log_status(f"⚠️ Error formatting quantity: {str(e)}")
            # Return original quantity as fallback
            return quantity 