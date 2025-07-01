import threading
import time
import json
import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import talib
import importlib.util
import sys
import os
from dataclasses import dataclass
from enum import Enum

# Import the OANDA trader and market data service
from oanda_trader import OandaTrader
from .market_data_service import MarketDataService

class BotStatus(Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    ERROR = "error"
    PAUSED = "paused"

@dataclass
class BotInstance:
    id: str
    name: str
    description: str
    code: str
    instruments: List[str]
    risk_level: str
    execution_interval: int  # minutes
    trailing_stop_type: str  # 'none' or 'fixed_pips'
    trailing_stop_pips: int
    status: BotStatus = BotStatus.STOPPED
    created_at: datetime = None
    updated_at: datetime = None
    last_run: Optional[datetime] = None
    positions: Dict = None
    performance: Dict = None
    error_count: int = 0
    last_error: Optional[str] = None

class CustomBotService:
    def __init__(self, broker: OandaTrader, supabase_service=None):
        self.broker = broker
        self.market_data = MarketDataService()
        self.supabase_service = supabase_service  # For database updates
        self.bots: Dict[str, BotInstance] = {}
        self.bot_threads: Dict[str, threading.Thread] = {}
        self.bot_instances: Dict[str, Any] = {}  # Actual bot class instances
        self.bot_stop_flags: Dict[str, threading.Event] = {}  # Stop flags for threads
        self.logger = logging.getLogger(__name__)
        
        # Bot data storage (in production, use database)
        self.data_dir = "bot_data"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Load existing bots from storage
        self._load_bots_from_storage()
        
        # Start execution engine
        self._start_execution_engine()
        
    def _start_execution_engine(self):
        """Start the main execution engine that monitors and runs bots"""
        self.execution_thread = threading.Thread(target=self._execution_loop, daemon=True)
        self.execution_thread.start()
        self.logger.info("Custom bot execution engine started")
        
    def _execution_loop(self):
        """Main execution loop that runs all active bots"""
        while True:
            try:
                current_time = datetime.now()
                
                for bot_id, bot in self.bots.items():
                    if bot.status == BotStatus.RUNNING:
                        # Check if it's time to run this bot
                        if self._should_run_bot(bot, current_time):
                            self._execute_bot(bot_id)
                
                # Sleep for 30 seconds before checking again
                time.sleep(30)
                
            except Exception as e:
                self.logger.error(f"Error in execution loop: {e}")
                time.sleep(30)  # Continue after error
                
    def _should_run_bot(self, bot: BotInstance, current_time: datetime) -> bool:
        """Check if it's time to run a bot based on its execution interval"""
        if bot.last_run is None:
            return True
            
        time_since_last_run = current_time - bot.last_run
        interval_minutes = bot.execution_interval
        
        return time_since_last_run >= timedelta(minutes=interval_minutes)
        
    def _execute_bot(self, bot_id: str):
        """Execute a single bot's trading logic"""
        try:
            bot = self.bots[bot_id]
            
            # Create bot instance if not exists
            if bot_id not in self.bot_instances:
                bot_instance = self._create_bot_instance(bot)
                if bot_instance is None:
                    return
                self.bot_instances[bot_id] = bot_instance
            
            bot_instance = self.bot_instances[bot_id]
            
            # Get market data for all bot instruments
            market_data = {}
            for instrument in bot.instruments:
                try:
                    # Get recent price data with indicators
                    data = self._get_market_data_with_indicators(instrument)
                    if data is not None:
                        market_data[instrument] = data
                except Exception as e:
                    self.logger.error(f"Error fetching data for {instrument}: {e}")
                    continue
            
            if not market_data:
                self.logger.warning(f"No market data available for bot {bot_id}")
                return
            
            # Run bot analysis
            signals = bot_instance.run_live_analysis(market_data)
            
            if signals:
                # Execute signals
                self._execute_signals(bot_id, signals)
            
            # Update bot last run time with precise timestamp
            bot.last_run = datetime.now()
            bot.error_count = 0  # Reset error count on successful run
            
            # Update the database with precise execution time
            if self.supabase_service:
                try:
                    # Temporary user ID for database update
                    user_id = '11111111-1111-1111-1111-111111111111'
                    self.supabase_service.update_bot_status(user_id, bot_id, 'running')
                except Exception as e:
                    self.logger.warning(f"Failed to update database timestamp: {e}")
            
            self.logger.info(f"Bot {bot_id} executed successfully")
            
        except Exception as e:
            self.logger.error(f"Error executing bot {bot_id}: {e}")
            bot = self.bots[bot_id]
            bot.error_count += 1
            bot.last_error = str(e)
            
            # Stop bot if too many errors
            if bot.error_count >= 5:
                bot.status = BotStatus.ERROR
                self.logger.error(f"Bot {bot_id} stopped due to repeated errors")
            
            self._save_bots_to_storage()
            
    def _create_bot_instance(self, bot: BotInstance):
        """Create a bot instance from bot code"""
        try:
            # Prepare execution environment
            exec_globals = {
                'pd': pd,
                'np': np,
                'talib': talib,
                'datetime': datetime,
                'timedelta': timedelta,
                'logging': logging,
                'time': time
            }
            
            # Execute the bot code
            exec(bot.code, exec_globals)
            
            # Find the bot class (assumes class name matches pattern)
            bot_class = None
            for name, obj in exec_globals.items():
                if (isinstance(obj, type) and 
                    hasattr(obj, 'run_live_analysis') and 
                    name not in ['pd', 'np', 'talib', 'datetime', 'timedelta', 'logging', 'time']):
                    bot_class = obj
                    break
            
            if bot_class is None:
                raise Exception("No valid bot class found in code")
            
            # Create bot instance with configuration
            config = {
                'name': bot.name,
                'instruments': bot.instruments,
                'risk_level': bot.risk_level,
                'execution_interval': bot.execution_interval,
                'trailing_stop_type': bot.trailing_stop_type,
                'trailing_stop_pips': bot.trailing_stop_pips,
                'broker': self.broker,
                'logger': self.logger
            }
            
            bot_instance = bot_class(config)
            return bot_instance
            
        except Exception as e:
            self.logger.error(f"Error creating bot instance: {e}")
            return None
            
    def _get_market_data_with_indicators(self, instrument: str) -> Optional[Dict]:
        """Get market data with technical indicators for an instrument"""
        try:
            # Get recent price data (last 100 candles for indicators)
            candles = self.market_data.get_candles(
                instrument=instrument,
                granularity="M5",  # 5-minute candles
                count=100
            )
            
            if not candles or len(candles) < 20:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(candles)
            df['time'] = pd.to_datetime(df['time'])
            df = df.sort_values('time')
            
            # Calculate technical indicators
            close_prices = df['c'].astype(float).values
            high_prices = df['h'].astype(float).values
            low_prices = df['l'].astype(float).values
            volume = df['volume'].astype(float).values if 'volume' in df.columns else None
            
            # Moving averages
            ma_20 = talib.SMA(close_prices, timeperiod=20)
            ma_50 = talib.SMA(close_prices, timeperiod=50)
            
            # RSI
            rsi = talib.RSI(close_prices, timeperiod=14)
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(close_prices)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close_prices)
            
            # Current price and indicators
            current_data = {
                'symbol': instrument,
                'price': float(close_prices[-1]),
                'high': float(high_prices[-1]),
                'low': float(low_prices[-1]),
                'timestamp': df.iloc[-1]['time'],
                'indicators': {
                    'ma_20': float(ma_20[-1]) if not np.isnan(ma_20[-1]) else None,
                    'ma_50': float(ma_50[-1]) if not np.isnan(ma_50[-1]) else None,
                    'rsi': float(rsi[-1]) if not np.isnan(rsi[-1]) else None,
                    'macd': float(macd[-1]) if not np.isnan(macd[-1]) else None,
                    'macd_signal': float(macd_signal[-1]) if not np.isnan(macd_signal[-1]) else None,
                    'bb_upper': float(bb_upper[-1]) if not np.isnan(bb_upper[-1]) else None,
                    'bb_lower': float(bb_lower[-1]) if not np.isnan(bb_lower[-1]) else None,
                },
                'history': {
                    'prices': close_prices[-20:].tolist(),  # Last 20 prices
                    'ma_20': ma_20[-20:].tolist(),
                    'ma_50': ma_50[-20:].tolist(),
                    'rsi': rsi[-20:].tolist()
                }
            }
            
            return current_data
            
        except Exception as e:
            self.logger.error(f"Error getting market data for {instrument}: {e}")
            return None
            
    def _execute_signals(self, bot_id: str, signals: List[Dict]):
        """Execute trading signals from a bot"""
        try:
            bot = self.bots[bot_id]
            
            for signal in signals:
                if signal.get('action') in ['buy', 'sell']:
                    # Calculate position size based on risk management
                    position_size = self._calculate_position_size(bot, signal)
                    
                    # Create order
                    order = {
                        'symbol': signal['symbol'],
                        'quantity': position_size,
                        'side': signal['action'],
                        'order_type': 'market',
                        'stop_loss': signal.get('stop_loss'),
                        'take_profit': signal.get('take_profit'),
                        'metadata': {
                            'bot_id': bot_id,
                            'bot_name': bot.name,
                            'signal_type': signal.get('signal_type', 'unknown'),
                            'confidence': signal.get('confidence', 0.5)
                        }
                    }
                    
                    # Submit order
                    order_id = self.broker.submit_order(order)
                    
                    if order_id:
                        self.logger.info(f"Bot {bot_id} executed {signal['action']} order for {signal['symbol']}")
                        
                        # Update bot performance tracking
                        if 'total_trades' not in bot.performance:
                            bot.performance['total_trades'] = 0
                        bot.performance['total_trades'] += 1
                        
                elif signal.get('action') == 'close':
                    # Close existing position
                    self.close_bot_position(bot_id, signal['symbol'], "Bot signal close")
                    
        except Exception as e:
            self.logger.error(f"Error executing signals for bot {bot_id}: {e}")
            
    def _calculate_position_size(self, bot: BotInstance, signal: Dict) -> float:
        """Calculate position size based on risk management"""
        try:
            # Get account balance
            account_balance = self.broker.get_account_balance()
            
            # Risk percentages by level
            risk_percentages = {
                'low': 0.01,      # 1%
                'medium': 0.02,   # 2%
                'high': 0.03      # 3%
            }
            
            risk_percent = risk_percentages.get(bot.risk_level, 0.02)
            risk_amount = account_balance * risk_percent
            
            # Calculate position size based on stop loss
            current_price = signal.get('price', 0)
            stop_loss = signal.get('stop_loss', 0)
            
            if current_price and stop_loss:
                price_risk = abs(current_price - stop_loss)
                if price_risk > 0:
                    position_size = risk_amount / price_risk
                else:
                    position_size = risk_amount / (current_price * 0.01)  # 1% fallback
            else:
                # Fallback calculation
                position_size = risk_amount / (current_price * 0.01)
            
            # Apply instrument-specific limits
            symbol = signal['symbol']
            if 'XAU' in symbol:  # Gold
                position_size = min(position_size, 10)
            elif 'BTC' in symbol:  # Bitcoin
                position_size = min(position_size, 1)
            else:  # Forex
                position_size = min(position_size, 100000)
            
            return max(position_size, 100)  # Minimum position size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 1000  # Default fallback
    
    def create_bot(self, bot_data: Dict) -> str:
        """Create a new custom bot"""
        try:
            bot_id = f"custom_bot_{int(time.time())}"
            
            bot = BotInstance(
                id=bot_id,
                name=bot_data['name'],
                description=bot_data['description'],
                code=bot_data['code'],
                instruments=bot_data['instruments'],
                risk_level=bot_data['risk_level'],
                execution_interval=bot_data['execution_interval'],
                trailing_stop_type=bot_data['trailing_stop_type'],
                trailing_stop_pips=bot_data['trailing_stop_pips'],
                created_at=datetime.now(),
                updated_at=datetime.now(),
                positions={},
                performance={'total_trades': 0, 'win_rate': 0, 'total_pnl': 0}
            )
            
            self.bots[bot_id] = bot
            self._save_bots_to_storage()
            
            self.logger.info(f"Created bot {bot_id}: {bot.name}")
            return bot_id
            
        except Exception as e:
            self.logger.error(f"Error creating bot: {e}")
            raise
    
    def get_all_bots(self) -> List[BotInstance]:
        """Get all bots"""
        return list(self.bots.values())
    
    def get_bot(self, bot_id: str) -> Optional[BotInstance]:
        """Get a specific bot"""
        return self.bots.get(bot_id)
    
    def update_bot(self, bot_id: str, bot_data: Dict) -> bool:
        """Update an existing bot"""
        try:
            if bot_id not in self.bots:
                raise Exception(f"Bot {bot_id} not found")
            
            bot = self.bots[bot_id]
            
            # Update bot data
            bot.name = bot_data.get('name', bot.name)
            bot.description = bot_data.get('description', bot.description)
            bot.code = bot_data.get('code', bot.code)
            bot.instruments = bot_data.get('instruments', bot.instruments)
            bot.risk_level = bot_data.get('risk_level', bot.risk_level)
            bot.execution_interval = bot_data.get('execution_interval', bot.execution_interval)
            bot.trailing_stop_type = bot_data.get('trailing_stop_type', bot.trailing_stop_type)
            bot.trailing_stop_pips = bot_data.get('trailing_stop_pips', bot.trailing_stop_pips)
            bot.updated_at = datetime.now()
            
            self._save_bots_to_storage()
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating bot {bot_id}: {e}")
            raise
    
    def delete_bot(self, bot_id: str) -> bool:
        """Delete a bot"""
        try:
            if bot_id not in self.bots:
                raise Exception(f"Bot {bot_id} not found")
            
            del self.bots[bot_id]
            self._save_bots_to_storage()
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting bot {bot_id}: {e}")
            raise
    
    def load_bot_from_database(self, bot_data: Dict) -> bool:
        """Load a bot from database data into execution service"""
        try:
            bot_id = bot_data['id']
            
            # Convert database bot to BotInstance
            bot = BotInstance(
                id=bot_id,
                name=bot_data['name'],
                description=bot_data['description'],
                code=bot_data['code'],
                instruments=bot_data['instruments'],
                risk_level=bot_data['risk_level'],
                execution_interval=bot_data['execution_interval'],
                trailing_stop_type=bot_data['trailing_stop_type'],
                trailing_stop_pips=bot_data['trailing_stop_pips'],
                status=BotStatus.STOPPED,  # Will be set to running by start_bot
                created_at=datetime.fromisoformat(bot_data['created_at']) if bot_data.get('created_at') else datetime.now(),
                updated_at=datetime.fromisoformat(bot_data['updated_at']) if bot_data.get('updated_at') else datetime.now(),
                last_run=datetime.fromisoformat(bot_data['last_run']) if bot_data.get('last_run') else None,
                positions={},
                performance={'total_trades': 0, 'win_rate': 0, 'total_pnl': 0}
            )
            
            # Add to execution service
            self.bots[bot_id] = bot
            self.logger.info(f"Loaded bot {bot_id} from database: {bot.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading bot from database: {e}")
            return False
    
    def start_bot(self, bot_id: str) -> bool:
        """Start a bot (will be loaded from database if not already loaded)"""
        try:
            # Bot should be loaded before calling this method
            if bot_id not in self.bots:
                raise Exception(f"Bot {bot_id} not found in execution service. Load it first with load_bot_from_database().")
            
            bot = self.bots[bot_id]
            
            # Validate bot code before starting
            if not self._validate_bot_code(bot.code):
                raise Exception("Bot code validation failed")
            
            # Clear any existing bot instance to force recreation
            if bot_id in self.bot_instances:
                del self.bot_instances[bot_id]
            
            bot.status = BotStatus.RUNNING
            bot.error_count = 0
            bot.last_error = None
            # Don't save to storage - database is the source of truth
            
            self.logger.info(f"Started bot {bot_id}: {bot.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting bot {bot_id}: {e}")
            # Set bot to error state
            if bot_id in self.bots:
                self.bots[bot_id].status = BotStatus.ERROR
                self.bots[bot_id].last_error = str(e)
            raise
    
    def stop_bot(self, bot_id: str) -> bool:
        """Stop a bot"""
        try:
            if bot_id not in self.bots:
                raise Exception(f"Bot {bot_id} not found")
            
            bot = self.bots[bot_id]
            bot.status = BotStatus.STOPPED
            
            # Clean up bot instance
            if bot_id in self.bot_instances:
                del self.bot_instances[bot_id]
            
            self._save_bots_to_storage()
            
            self.logger.info(f"Stopped bot {bot_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping bot {bot_id}: {e}")
            raise
    
    def get_bot_positions(self, bot_id: str) -> Dict:
        """Get positions for a specific bot"""
        if bot_id not in self.bots:
            return {}
        
        bot = self.bots[bot_id]
        
        # Get current positions from broker for bot's instruments
        all_positions = self.broker.get_tracked_positions()
        bot_positions = {}
        
        for symbol in bot.instruments:
            for pos_key, pos_data in all_positions.items():
                if pos_data['symbol'] == symbol:
                    bot_positions[pos_key] = pos_data
        
        return bot_positions
    
    def close_bot_position(self, bot_id: str, symbol: str, reason: str = "Manual close") -> bool:
        """Close a specific position for a bot"""
        try:
            positions = self.get_bot_positions(bot_id)
            
            for pos_key, pos_data in positions.items():
                if pos_data['symbol'] == symbol:
                    # Create close order
                    order = {
                        'symbol': symbol,
                        'quantity': abs(pos_data['quantity']),
                        'side': 'sell' if pos_data['quantity'] > 0 else 'buy',
                        'order_type': 'market'
                    }
                    
                    order_id = self.broker.submit_order(order)
                    self.logger.info(f"Closed position {symbol} for bot {bot_id}")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error closing position for bot {bot_id}: {e}")
            raise
    
    def _validate_bot_code(self, code: str) -> bool:
        """Validate bot code structure"""
        try:
            # Basic validation - check for required methods
            required_methods = ['run_live_analysis']
            
            for method in required_methods:
                if f"def {method}" not in code:
                    self.logger.error(f"Bot code missing required method: {method}")
                    return False
            
            # Check for required class structure
            if 'class ' not in code:
                self.logger.error("Bot code must contain a class definition")
                return False
            
            # Try to compile the code
            compile(code, '<string>', 'exec')
            return True
            
        except SyntaxError as e:
            self.logger.error(f"Bot code syntax error: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Bot code validation error: {e}")
            return False
    
    def _load_bots_from_storage(self):
        """Load bots from file storage"""
        try:
            bots_file = os.path.join(self.data_dir, "bots.json")
            if os.path.exists(bots_file):
                with open(bots_file, 'r') as f:
                    bots_data = json.load(f)
                    
                for bot_data in bots_data:
                    bot = BotInstance(
                        id=bot_data['id'],
                        name=bot_data['name'],
                        description=bot_data['description'],
                        code=bot_data['code'],
                        instruments=bot_data['instruments'],
                        risk_level=bot_data['risk_level'],
                        execution_interval=bot_data['execution_interval'],
                        trailing_stop_type=bot_data['trailing_stop_type'],
                        trailing_stop_pips=bot_data['trailing_stop_pips'],
                        status=BotStatus(bot_data.get('status', 'stopped')),
                        created_at=datetime.fromisoformat(bot_data['created_at']) if bot_data.get('created_at') else datetime.now(),
                        updated_at=datetime.fromisoformat(bot_data['updated_at']) if bot_data.get('updated_at') else datetime.now(),
                        last_run=datetime.fromisoformat(bot_data['last_run']) if bot_data.get('last_run') else None,
                        positions=bot_data.get('positions', {}),
                        performance=bot_data.get('performance', {'total_trades': 0, 'win_rate': 0, 'total_pnl': 0}),
                        error_count=bot_data.get('error_count', 0),
                        last_error=bot_data.get('last_error')
                    )
                    self.bots[bot.id] = bot
                    
                self.logger.info(f"Loaded {len(self.bots)} bots from storage")
        except Exception as e:
            self.logger.error(f"Error loading bots from storage: {e}")
    
    def _save_bots_to_storage(self):
        """Save bots to file storage"""
        try:
            bots_data = []
            for bot in self.bots.values():
                bots_data.append({
                    'id': bot.id,
                    'name': bot.name,
                    'description': bot.description,
                    'code': bot.code,
                    'instruments': bot.instruments,
                    'risk_level': bot.risk_level,
                    'execution_interval': bot.execution_interval,
                    'trailing_stop_type': bot.trailing_stop_type,
                    'trailing_stop_pips': bot.trailing_stop_pips,
                    'status': bot.status.value,
                    'created_at': bot.created_at.isoformat() if bot.created_at else None,
                    'updated_at': bot.updated_at.isoformat() if bot.updated_at else None,
                    'last_run': bot.last_run.isoformat() if bot.last_run else None,
                    'positions': bot.positions or {},
                    'performance': bot.performance or {'total_trades': 0, 'win_rate': 0, 'total_pnl': 0},
                    'error_count': bot.error_count,
                    'last_error': bot.last_error
                })
            
            bots_file = os.path.join(self.data_dir, "bots.json")
            with open(bots_file, 'w') as f:
                json.dump(bots_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving bots to storage: {e}")
