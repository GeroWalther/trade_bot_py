import logging
from datetime import datetime, timedelta
import threading
import time
import json
import aiohttp
import asyncio
import os
import pickle

logger = logging.getLogger(__name__)

class AIStrategy:
    def __init__(self, broker, parameters=None):
        logger.info(f"Initializing AI Strategy with parameters: {parameters}")
        
        self.broker = broker
        # Update available instruments to match the AI Analysis dropdown options
        self.available_instruments = [
            'EUR_USD',  # USD/JPY in analysis
            'BTC_USD',  # Bitcoin (BTC/USD) in analysis
            'SPX500_USD',  # S&P 500 in analysis
            'NAS100_USD',  # Nasdaq in analysis
            'XAU_USD'   # Gold in analysis
        ]
        
        # Map trading symbols to AI analysis asset names
        self.symbol_to_asset_map = {
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
        
        self.current_trade = None
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
                
                # Update real-time metrics if we have an open position
                if self.current_trade:
                    symbol = self.current_trade['symbol']
                    current_price = self.broker.get_last_price(symbol)
                    
                    if current_price:
                        entry_price = self.current_trade['entry_price']
                        quantity = self.current_trade['quantity']
                        
                        # Calculate unrealized P/L
                        if self.current_trade['side'] == 'BUY':
                            unrealized_pl = (current_price - entry_price) * quantity
                        else:  # SELL
                            unrealized_pl = (entry_price - current_price) * quantity
                        
                        # Update daily P/L with unrealized profit/loss
                        # Note: This is just for display, we'll adjust when the trade is closed
                        self.performance['unrealized_pl'] = unrealized_pl
                
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
                url = "http://localhost:5003/api/advanced-market-analysis"
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
                
            # Check if we have a current trade
            if self.current_trade:
                self.log_status("ℹ️ Already in a trade, skipping entry signal check")
                return False
                
            # Get current price
            symbol = self.parameters['symbol']
            current_price = self.broker.get_last_price(symbol)
            if not current_price:
                self.log_status(f"❌ Could not get current price for {symbol}")
                return False
                
            # Get AI recommendation
            trading_strategy = self.ai_analysis['trading_strategy']
            direction = trading_strategy['direction']
            entry_price = trading_strategy['entry']['price']
            
            # Try to parse the entry price (it might be a string with commas or a range)
            try:
                if isinstance(entry_price, str):
                    # Handle price ranges like "2860 - 2865" by taking the first price
                    if " - " in entry_price:
                        entry_price = entry_price.split(" - ")[0]
                    entry_price = float(entry_price.replace(',', ''))
                else:
                    entry_price = float(entry_price)
            except (ValueError, TypeError):
                self.log_status(f"❌ Invalid entry price format: {entry_price}")
                return False
                
            # Get take profit and stop loss values
            try:
                take_profit = trading_strategy['take_profit_1']['price']
                if isinstance(take_profit, str):
                    take_profit = float(take_profit.replace(',', ''))
                
                stop_loss = trading_strategy['stop_loss']['price']
                if isinstance(stop_loss, str):
                    stop_loss = float(stop_loss.replace(',', ''))
            except (ValueError, TypeError, KeyError):
                take_profit = None
                stop_loss = None
                
            # Calculate price difference percentage
            price_diff_pct = abs(current_price - entry_price) / entry_price * 100
            
            # Check if current price is close to the recommended entry price (within 1%)
            if price_diff_pct > 1.0:
                self.log_status(f"ℹ️ Current price ({current_price:.2f}) is {price_diff_pct:.2f}% away from recommended entry ({entry_price:.2f})")
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
                
            symbol = self.parameters['symbol']
            quantity = self.parameters['quantity']
            
            # Get current price
            current_price = self.broker.get_last_price(symbol)
            if not current_price:
                self.log_status(f"❌ Cannot execute trade - No price data for {symbol}")
                return False
                
            # Get AI recommendation
            trading_strategy = self.ai_analysis['trading_strategy']
            direction = trading_strategy['direction']
            
            # Parse entry price from AI recommendation
            try:
                entry_price_str = trading_strategy['entry']['price']
                if isinstance(entry_price_str, str):
                    if " - " in entry_price_str:
                        entry_price_str = entry_price_str.split(" - ")[0]
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
                'symbol': symbol,
                'quantity': quantity,
                'side': side,
                'price': entry_price  # Use AI-recommended entry price for the order
            }
            
            # Log the trade - we already logged the comprehensive message in should_enter_trade
            
            # Submit the order
            order_id = self.broker.submit_order(order)
            
            if order_id:
                # Parse take profit and stop loss from AI recommendation
                try:
                    take_profit_1 = float(str(trading_strategy['take_profit_1']['price']).replace(',', ''))
                    stop_loss = float(str(trading_strategy['stop_loss']['price']).replace(',', ''))
                except (ValueError, TypeError, KeyError):
                    # Use default percentages if AI values can't be parsed
                    self.log_status("⚠️ Could not parse AI take profit/stop loss values, using fallback values")
                    if side == 'buy':
                        take_profit_1 = entry_price * 1.01  # Default 1% take profit
                        stop_loss = entry_price * 0.995  # Default 0.5% stop loss
                    else:
                        take_profit_1 = entry_price * 0.99  # Default 1% take profit
                        stop_loss = entry_price * 1.005  # Default 0.5% stop loss
                
                # Record the trade
                self.current_trade = {
                    'order_id': order_id,
                    'symbol': symbol,
                    'quantity': quantity,
                    'entry_price': entry_price,  # Use AI-recommended entry price
                    'entry_time': self.broker.get_time(),
                    'side': side.upper(),
                    'take_profit': take_profit_1,
                    'stop_loss': stop_loss,
                    'ai_analysis': self.ai_analysis
                }
                
                self.log_status(f"✅ Position opened successfully at {entry_price:.2f}")
                return True
            else:
                self.log_status("❌ Order submission failed")
            
            return False
            
        except Exception as e:
            self.log_status(f"❌ Error executing trade: {str(e)}")
            logger.error(f"Error executing trade: {e}", exc_info=True)
            return False

    def check_exit_conditions(self):
        """Check if we should exit the current trade based on take profit or stop loss"""
        try:
            if not self.current_trade:
                return False
                
            symbol = self.current_trade['symbol']
            entry_price = self.current_trade['entry_price']
            take_profit = self.current_trade['take_profit']
            stop_loss = self.current_trade['stop_loss']
            side = self.current_trade['side']
            
            # Get current price
            current_price = self.broker.get_last_price(symbol)
            if not current_price:
                self.log_status(f"❌ Could not get current price for {symbol}")
                return False
                
            # Calculate profit/loss
            if side == 'BUY':
                profit_pct = (current_price - entry_price) / entry_price * 100
                # Exit if price reaches take profit or falls below stop loss
                if current_price >= take_profit:
                    self.log_status(f"🎯 Take profit reached: {current_price:.2f} >= {take_profit:.2f}")
                    return True
                elif current_price <= stop_loss:
                    self.log_status(f"🛑 Stop loss triggered: {current_price:.2f} <= {stop_loss:.2f}")
                    return True
            else:  # SELL
                profit_pct = (entry_price - current_price) / entry_price * 100
                # Exit if price falls below take profit or rises above stop loss
                if current_price <= take_profit:
                    self.log_status(f"🎯 Take profit reached: {current_price:.2f} <= {take_profit:.2f}")
                    return True
                elif current_price >= stop_loss:
                    self.log_status(f"🛑 Stop loss triggered: {current_price:.2f} >= {stop_loss:.2f}")
                    return True
            
            # Log current position status
            self.log_status(f"📊 Position update: Entry: {entry_price:.2f}, Current: {current_price:.2f}, P/L: {profit_pct:.2f}%")
            
            return False
            
        except Exception as e:
            self.log_status(f"❌ Error checking exit conditions: {str(e)}")
            logger.error(f"Error checking exit conditions: {e}", exc_info=True)
            return False

    def exit_trade(self):
        """Exit current trade"""
        try:
            if not self.current_trade:
                return False
            
            symbol = self.current_trade['symbol']
            quantity = self.current_trade['quantity']
            entry_price = self.current_trade['entry_price']
            side = self.current_trade['side']
            
            current_price = self.broker.get_last_price(symbol)
            if not current_price:
                self.log_status("❌ Could not get current price for exit")
                return False
            
            # Calculate profit/loss
            if side == 'BUY':
                profit_loss = (current_price - entry_price) * quantity
                profit_pct = ((current_price - entry_price) / entry_price) * 100
                exit_side = 'sell'
            else:
                profit_loss = (entry_price - current_price) * quantity
                profit_pct = ((entry_price - current_price) / entry_price) * 100
                exit_side = 'buy'
            
            self.log_status(
                f"🔄 Closing position - Entry: {entry_price:.2f}, "
                f"Exit: {current_price:.2f}, "
                f"P/L: {profit_loss:.2f} ({profit_pct:.2f}%)"
            )
            
            order = {
                'symbol': symbol,
                'quantity': quantity,
                'side': exit_side
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
                    'timestamp': self.broker.get_time(),
                    'ai_analysis': self.current_trade.get('ai_analysis')
                }
                
                self.update_performance(trade_result)
                self.trades_history.append(trade_result)
                self.current_trade = None
                
                self.log_status(
                    f"✅ Position closed - Duration: {trade_result['duration']:.1f} minutes, "
                    f"P/L: {profit_loss:.2f} ({profit_pct:.2f}%), "
                    f"Total P/L: {self.performance['total_profit_loss']:.2f}"
                )
                return True
            
            self.log_status("❌ Failed to close position")
            return False
            
        except Exception as e:
            self.log_status(f"❌ Error exiting trade: {str(e)}")
            logger.error(f"Error exiting trade: {e}", exc_info=True)
            return False

    def run(self):
        """Main strategy loop"""
        self.log_status(
            f"🟢 AI Strategy activated:\n"
            f"   Asset: {self.parameters['symbol']}\n"
            f"   Term: {self.parameters['trading_term']}\n"
            f"   Risk Level: {self.parameters['risk_level']}\n"
            f"   Quantity: {self.parameters['quantity']}\n"
            f"   Check Interval: {self.parameters['check_interval']}s\n"
            f"   Continue After Trade: {'Yes' if self.parameters['continue_after_trade'] else 'No'}"
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
            # Check if we need to get a new AI analysis
            current_time = datetime.now()
            analysis_age_hours = 0
            
            if self.last_analysis_time:
                analysis_age = current_time - self.last_analysis_time
                analysis_age_hours = analysis_age.total_seconds() / 3600
            
            # Get new analysis if we don't have one or if it's older than 4 hours
            if not self.ai_analysis or analysis_age_hours > 4:
                # Get the asset name for display
                asset_name = self.symbol_to_asset_map.get(self.parameters['symbol'], 'Gold')
                trading_term = self.parameters['trading_term']
                risk_level = self.parameters['risk_level']
                
                # Simplified log message
                self.ai_analysis = await self.get_ai_analysis()
                if not self.ai_analysis:
                    self.log_status("❌ Failed to get AI analysis, skipping trading cycle")
                    return
            
            # Check if we have an open position
            if self.current_trade:
                # Check if we should exit the trade
                if self.check_exit_conditions():
                    self.exit_trade()
            else:
                # Check if we should enter a trade
                if self.should_enter_trade():
                    self.execute_trade()
                    
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
        self._continue = False
        self.log_status("🔴 Strategy stopped")

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
                "Monitors positions for take profit and stop loss conditions"
            ]
        }
        
        # Get current position info if we have one
        current_position = None
        if self.current_trade:
            symbol = self.current_trade['symbol']
            current_price = self.broker.get_last_price(symbol)
            
            if current_price:
                entry_price = self.current_trade['entry_price']
                if self.current_trade['side'] == 'BUY':
                    profit_pct = (current_price - entry_price) / entry_price * 100
                else:
                    profit_pct = (entry_price - current_price) / entry_price * 100
                    
                current_position = {
                    'symbol': symbol,
                    'side': self.current_trade['side'],
                    'entry_price': entry_price,
                    'current_price': current_price,
                    'profit_pct': profit_pct,
                    'take_profit': self.current_trade['take_profit'],
                    'stop_loss': self.current_trade['stop_loss']
                }
        
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
            'current_trade': current_position,
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