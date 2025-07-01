import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from supabase import create_client, Client
from dataclasses import dataclass
from enum import Enum
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class BotStatus(Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    ERROR = "error"
    PAUSED = "paused"

@dataclass
class CustomBot:
    id: str
    user_id: str
    name: str
    description: str
    code: str
    instruments: List[str]
    risk_level: str
    execution_interval: int
    trailing_stop_type: str
    trailing_stop_pips: int
    status: BotStatus
    is_active: bool
    created_at: datetime
    updated_at: datetime
    last_run: Optional[datetime] = None
    error_count: int = 0
    last_error: Optional[str] = None

class SupabaseService:
    def __init__(self):
        self.url = os.getenv('SUPABASE_URL')
        # Use service key for development (bypasses RLS) or anon key for production
        self.key = os.getenv('SUPABASE_SERVICE_KEY') or os.getenv('SUPABASE_KEY')
        
        if not self.url or not self.key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY (or SUPABASE_SERVICE_KEY) environment variables are required")
        
        self.supabase: Client = create_client(self.url, self.key)
        key_type = "service_role" if os.getenv('SUPABASE_SERVICE_KEY') else "anon"
        logger.info(f"Supabase service initialized successfully using {key_type} key")
    
    def set_user_context(self, user_token: str):
        """Set user authentication context for RLS"""
        try:
            # This sets the user context for Row Level Security
            self.supabase.auth.set_session(user_token)
        except Exception as e:
            logger.error(f"Error setting user context: {e}")
    
    # ========================================
    # USER MANAGEMENT
    # ========================================
    
    def get_user_profile(self, user_id: str) -> Optional[Dict]:
        """Get user profile"""
        try:
            response = self.supabase.table('user_profiles').select('*').eq('id', user_id).single().execute()
            return response.data
        except Exception as e:
            logger.error(f"Error getting user profile: {e}")
            return None
    
    def create_user_profile(self, user_data: Dict) -> bool:
        """Create user profile"""
        try:
            response = self.supabase.table('user_profiles').insert(user_data).execute()
            return len(response.data) > 0
        except Exception as e:
            logger.error(f"Error creating user profile: {e}")
            return False
    
    def get_user_subscription(self, user_id: str) -> Optional[Dict]:
        """Get user subscription info"""
        try:
            response = self.supabase.table('user_subscriptions').select('*').eq('user_id', user_id).single().execute()
            return response.data
        except Exception as e:
            logger.error(f"Error getting user subscription: {e}")
            return None
    
    # ========================================
    # BOT MANAGEMENT
    # ========================================
    
    def create_bot(self, user_id: str, bot_data: Dict) -> Optional[str]:
        """Create a new custom bot"""
        try:
            # Check user's bot limit
            user_profile = self.get_user_profile(user_id)
            if not user_profile:
                # Create a test user profile for development
                logger.info(f"Creating test user profile for {user_id}")
                test_profile = {
                    'id': user_id,
                    'email': 'test@example.com',
                    'full_name': 'Test User',
                    'subscription_tier': 'free',
                    'max_bots': 10  # Generous limit for testing
                }
                self.create_user_profile(test_profile)
                user_profile = test_profile
            
            # Count existing bots
            existing_bots = self.supabase.table('custom_bots').select('id').eq('user_id', user_id).eq('is_active', True).execute()
            bot_count = len(existing_bots.data)
            
            max_bots = user_profile.get('max_bots', 3)
            if bot_count >= max_bots:
                raise Exception(f"Bot limit reached. Maximum {max_bots} bots allowed for your subscription tier.")
            
            # Create bot
            bot_insert_data = {
                'user_id': user_id,
                'name': bot_data['name'],
                'description': bot_data.get('description', ''),
                'code': bot_data['code'],
                'instruments': bot_data['instruments'],
                'risk_level': bot_data['risk_level'],
                'execution_interval': bot_data['execution_interval'],
                'trailing_stop_type': bot_data.get('trailing_stop_type', 'none'),
                'trailing_stop_pips': bot_data.get('trailing_stop_pips', 20),
                'status': 'stopped'
            }
            
            response = self.supabase.table('custom_bots').insert(bot_insert_data).execute()
            
            if response.data:
                bot_id = response.data[0]['id']
                
                # Initialize performance record
                perf_data = {
                    'bot_id': bot_id,
                    'user_id': user_id,
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'total_pnl': 0
                }
                self.supabase.table('bot_performance').insert(perf_data).execute()
                
                logger.info(f"Created bot {bot_id} for user {user_id}")
                return bot_id
            
            return None
            
        except Exception as e:
            logger.error(f"Error creating bot: {e}")
            raise
    
    def get_user_bots(self, user_id: str) -> List[Dict]:
        """Get all bots for a user"""
        try:
            response = self.supabase.table('custom_bots').select('''
                id, name, description, instruments, risk_level, execution_interval,
                trailing_stop_type, trailing_stop_pips, status, is_active,
                created_at, updated_at, last_run, error_count, last_error
            ''').eq('user_id', user_id).eq('is_active', True).order('created_at', desc=True).execute()
            
            return response.data
            
        except Exception as e:
            logger.error(f"Error getting user bots: {e}")
            return []
    
    def get_bot(self, user_id: str, bot_id: str) -> Optional[Dict]:
        """Get a specific bot"""
        try:
            response = self.supabase.table('custom_bots').select('*').eq('user_id', user_id).eq('id', bot_id).single().execute()
            return response.data
        except Exception as e:
            logger.error(f"Error getting bot {bot_id}: {e}")
            return None
    
    def update_bot(self, user_id: str, bot_id: str, bot_data: Dict) -> bool:
        """Update a bot"""
        try:
            update_data = {
                'name': bot_data.get('name'),
                'description': bot_data.get('description'),
                'code': bot_data.get('code'),
                'instruments': bot_data.get('instruments'),
                'risk_level': bot_data.get('risk_level'),
                'execution_interval': bot_data.get('execution_interval'),
                'trailing_stop_type': bot_data.get('trailing_stop_type'),
                'trailing_stop_pips': bot_data.get('trailing_stop_pips'),
                'updated_at': datetime.now().isoformat()
            }
            
            # Remove None values
            update_data = {k: v for k, v in update_data.items() if v is not None}
            
            response = self.supabase.table('custom_bots').update(update_data).eq('user_id', user_id).eq('id', bot_id).execute()
            
            return len(response.data) > 0
            
        except Exception as e:
            logger.error(f"Error updating bot {bot_id}: {e}")
            return False
    
    def delete_bot(self, user_id: str, bot_id: str) -> bool:
        """Soft delete a bot"""
        try:
            response = self.supabase.table('custom_bots').update({
                'is_active': False,
                'status': 'stopped',
                'updated_at': datetime.now().isoformat()
            }).eq('user_id', user_id).eq('id', bot_id).execute()
            
            return len(response.data) > 0
            
        except Exception as e:
            logger.error(f"Error deleting bot {bot_id}: {e}")
            return False
    
    def update_bot_status(self, user_id: str, bot_id: str, status: str, error: str = None) -> bool:
        """Update bot status"""
        try:
            update_data = {
                'status': status,
                'updated_at': datetime.now().isoformat()
            }
            
            if status == 'running':
                update_data['last_run'] = datetime.now().isoformat()
            
            if error:
                update_data['last_error'] = error
                update_data['error_count'] = self.supabase.table('custom_bots').select('error_count').eq('id', bot_id).single().execute().data['error_count'] + 1
            
            response = self.supabase.table('custom_bots').update(update_data).eq('user_id', user_id).eq('id', bot_id).execute()
            
            return len(response.data) > 0
            
        except Exception as e:
            logger.error(f"Error updating bot status {bot_id}: {e}")
            return False
    
    # ========================================
    # PERFORMANCE TRACKING
    # ========================================
    
    def get_bot_performance(self, user_id: str, bot_id: str) -> Optional[Dict]:
        """Get bot performance metrics"""
        try:
            response = self.supabase.table('bot_performance').select('*').eq('user_id', user_id).eq('bot_id', bot_id).single().execute()
            return response.data
        except Exception as e:
            logger.error(f"Error getting bot performance {bot_id}: {e}")
            return None
    
    def record_trade(self, user_id: str, bot_id: str, trade_data: Dict) -> bool:
        """Record a trade"""
        try:
            trade_insert_data = {
                'bot_id': bot_id,
                'user_id': user_id,
                'symbol': trade_data['symbol'],
                'direction': trade_data['direction'],
                'entry_price': trade_data['entry_price'],
                'exit_price': trade_data.get('exit_price'),
                'position_size': trade_data['position_size'],
                'entry_time': trade_data['entry_time'],
                'exit_time': trade_data.get('exit_time'),
                'pnl': trade_data.get('pnl'),
                'reason': trade_data.get('reason'),
                'trade_status': trade_data.get('trade_status', 'open'),
                'position_id': trade_data.get('position_id')
            }
            
            response = self.supabase.table('bot_trades').insert(trade_insert_data).execute()
            return len(response.data) > 0
            
        except Exception as e:
            logger.error(f"Error recording trade: {e}")
            return False
    
    def close_trade(self, user_id: str, trade_id: str, exit_data: Dict) -> bool:
        """Close a trade"""
        try:
            update_data = {
                'exit_price': exit_data['exit_price'],
                'exit_time': exit_data['exit_time'],
                'pnl': exit_data['pnl'],
                'trade_status': 'closed',
                'reason': exit_data.get('reason', 'Manual close')
            }
            
            response = self.supabase.table('bot_trades').update(update_data).eq('user_id', user_id).eq('id', trade_id).execute()
            
            return len(response.data) > 0
            
        except Exception as e:
            logger.error(f"Error closing trade {trade_id}: {e}")
            return False
    
    def get_bot_trades(self, user_id: str, bot_id: str, limit: int = 100) -> List[Dict]:
        """Get bot trades history"""
        try:
            response = self.supabase.table('bot_trades').select('*').eq('user_id', user_id).eq('bot_id', bot_id).order('entry_time', desc=True).limit(limit).execute()
            return response.data
        except Exception as e:
            logger.error(f"Error getting bot trades: {e}")
            return []
    
    # ========================================
    # POSITION MANAGEMENT
    # ========================================
    
    def save_position(self, user_id: str, bot_id: str, position_data: Dict) -> bool:
        """Save an active position"""
        try:
            position_insert_data = {
                'bot_id': bot_id,
                'user_id': user_id,
                'symbol': position_data['symbol'],
                'position_id': position_data['position_id'],
                'direction': position_data['direction'],
                'entry_price': position_data['entry_price'],
                'current_price': position_data.get('current_price'),
                'position_size': position_data['position_size'],
                'unrealized_pnl': position_data.get('unrealized_pnl', 0),
                'entry_time': position_data['entry_time'],
                'stop_loss': position_data.get('stop_loss'),
                'take_profit': position_data.get('take_profit'),
                'trailing_stop_distance': position_data.get('trailing_stop_distance')
            }
            
            # Use upsert to handle updates
            response = self.supabase.table('bot_positions').upsert(position_insert_data, on_conflict='bot_id,symbol').execute()
            return len(response.data) > 0
            
        except Exception as e:
            logger.error(f"Error saving position: {e}")
            return False
    
    def get_bot_positions(self, user_id: str, bot_id: str) -> List[Dict]:
        """Get active positions for a bot"""
        try:
            response = self.supabase.table('bot_positions').select('*').eq('user_id', user_id).eq('bot_id', bot_id).eq('is_active', True).execute()
            return response.data
        except Exception as e:
            logger.error(f"Error getting bot positions: {e}")
            return []
    
    def close_position(self, user_id: str, bot_id: str, symbol: str) -> bool:
        """Close a position"""
        try:
            response = self.supabase.table('bot_positions').update({
                'is_active': False,
                'last_updated': datetime.now().isoformat()
            }).eq('user_id', user_id).eq('bot_id', bot_id).eq('symbol', symbol).execute()
            
            return len(response.data) > 0
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False
    
    # ========================================
    # LOGGING
    # ========================================
    
    def log_bot_activity(self, user_id: str, bot_id: str, level: str, message: str, details: Dict = None) -> bool:
        """Log bot activity"""
        try:
            log_data = {
                'bot_id': bot_id,
                'user_id': user_id,
                'log_level': level,
                'message': message,
                'details': details or {},
                'timestamp': datetime.now().isoformat()
            }
            
            response = self.supabase.table('bot_logs').insert(log_data).execute()
            return len(response.data) > 0
            
        except Exception as e:
            logger.error(f"Error logging bot activity: {e}")
            return False
    
    def get_bot_logs(self, user_id: str, bot_id: str, limit: int = 100) -> List[Dict]:
        """Get bot logs"""
        try:
            response = self.supabase.table('bot_logs').select('*').eq('user_id', user_id).eq('bot_id', bot_id).order('timestamp', desc=True).limit(limit).execute()
            return response.data
        except Exception as e:
            logger.error(f"Error getting bot logs: {e}")
            return []
    
    # ========================================
    # USER API KEYS
    # ========================================
    
    def save_user_api_key(self, user_id: str, broker_data: Dict) -> bool:
        """Save encrypted user API key"""
        try:
            api_key_data = {
                'user_id': user_id,
                'broker_name': broker_data['broker_name'],
                'api_key_encrypted': broker_data['api_key_encrypted'],
                'account_id': broker_data.get('account_id'),
                'environment': broker_data.get('environment', 'practice'),
                'is_active': True
            }
            
            response = self.supabase.table('user_api_keys').upsert(api_key_data, on_conflict='user_id,broker_name').execute()
            return len(response.data) > 0
            
        except Exception as e:
            logger.error(f"Error saving user API key: {e}")
            return False
    
    def get_user_api_key(self, user_id: str, broker_name: str) -> Optional[Dict]:
        """Get user API key"""
        try:
            response = self.supabase.table('user_api_keys').select('*').eq('user_id', user_id).eq('broker_name', broker_name).eq('is_active', True).single().execute()
            return response.data
        except Exception as e:
            logger.error(f"Error getting user API key: {e}")
            return None
    
    # ========================================
    # BOT TEMPLATES MANAGEMENT
    # ========================================
    
    def get_bot_templates(self, language: str = None, category: str = None, strategy_type: str = None) -> List[Dict]:
        """Get bot templates with optional filters"""
        try:
            query = self.supabase.table('bot_templates').select('*')
            
            if language:
                query = query.eq('language', language)
            if category:
                query = query.eq('category', category)
            if strategy_type:
                query = query.eq('strategy_type', strategy_type)
            
            response = query.eq('is_active', True).order('created_at', desc=True).execute()
            return response.data
            
        except Exception as e:
            logger.error(f"Error getting bot templates: {e}")
            return []
    
    def get_bot_template(self, template_id: str) -> Optional[Dict]:
        """Get a specific bot template"""
        try:
            response = self.supabase.table('bot_templates').select('*').eq('id', template_id).eq('is_active', True).single().execute()
            return response.data
        except Exception as e:
            logger.error(f"Error getting bot template {template_id}: {e}")
            return None
    
    def create_bot_template(self, template_data: Dict) -> Optional[str]:
        """Create a new bot template"""
        try:
            insert_data = {
                'name': template_data['name'],
                'description': template_data.get('description', ''),
                'language': template_data['language'],
                'category': template_data.get('category', 'beginner'),
                'strategy_type': template_data.get('strategy_type', 'trend_following'),
                'code': template_data['code'],
                'author': template_data.get('author', 'System'),
                'tags': template_data.get('tags', [])
            }
            
            response = self.supabase.table('bot_templates').insert(insert_data).execute()
            
            if response.data:
                template_id = response.data[0]['id']
                logger.info(f"Created bot template {template_id}")
                return template_id
            
            return None
            
        except Exception as e:
            logger.error(f"Error creating bot template: {e}")
            raise
    
    def update_bot_template(self, template_id: str, template_data: Dict) -> bool:
        """Update a bot template"""
        try:
            update_data = {
                'name': template_data.get('name'),
                'description': template_data.get('description'),
                'language': template_data.get('language'),
                'category': template_data.get('category'),
                'strategy_type': template_data.get('strategy_type'),
                'code': template_data.get('code'),
                'author': template_data.get('author'),
                'tags': template_data.get('tags'),
                'updated_at': datetime.now().isoformat()
            }
            
            # Remove None values
            update_data = {k: v for k, v in update_data.items() if v is not None}
            
            response = self.supabase.table('bot_templates').update(update_data).eq('id', template_id).execute()
            
            return len(response.data) > 0
            
        except Exception as e:
            logger.error(f"Error updating bot template {template_id}: {e}")
            return False
    
    def delete_bot_template(self, template_id: str) -> bool:
        """Soft delete a bot template"""
        try:
            response = self.supabase.table('bot_templates').update({
                'is_active': False,
                'updated_at': datetime.now().isoformat()
            }).eq('id', template_id).execute()
            
            return len(response.data) > 0
            
        except Exception as e:
            logger.error(f"Error deleting bot template {template_id}: {e}")
            return False
    
    def increment_template_usage(self, template_id: str) -> bool:
        """Increment template usage count"""
        try:
            # Get current usage count
            current = self.supabase.table('bot_templates').select('usage_count').eq('id', template_id).single().execute()
            if current.data:
                new_count = current.data['usage_count'] + 1
                response = self.supabase.table('bot_templates').update({
                    'usage_count': new_count,
                    'updated_at': datetime.now().isoformat()
                }).eq('id', template_id).execute()
                
                return len(response.data) > 0
            
            return False
            
        except Exception as e:
            logger.error(f"Error incrementing template usage {template_id}: {e}")
            return False 