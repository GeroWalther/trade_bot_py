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

# Import the OANDA trader
from oanda_trader import OandaTrader

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
    def __init__(self, broker: OandaTrader):
        self.broker = broker
        self.bots: Dict[str, BotInstance] = {}
        self.bot_threads: Dict[str, threading.Thread] = {}
        self.bot_instances: Dict[str, Any] = {}  # Actual bot class instances
        self.logger = logging.getLogger(__name__)
        
        # Bot data storage (in production, use database)
        self.data_dir = "bot_data"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Load existing bots from storage
        self._load_bots_from_storage()
        
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
    
    def start_bot(self, bot_id: str) -> bool:
        """Start a bot"""
        try:
            if bot_id not in self.bots:
                raise Exception(f"Bot {bot_id} not found")
            
            bot = self.bots[bot_id]
            bot.status = BotStatus.RUNNING
            self._save_bots_to_storage()
            
            self.logger.info(f"Started bot {bot_id}: {bot.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting bot {bot_id}: {e}")
            raise
    
    def stop_bot(self, bot_id: str) -> bool:
        """Stop a bot"""
        try:
            if bot_id not in self.bots:
                raise Exception(f"Bot {bot_id} not found")
            
            bot = self.bots[bot_id]
            bot.status = BotStatus.STOPPED
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
