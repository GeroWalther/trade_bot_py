from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class TradingState:
    def __init__(self):
        self.is_trading = False
        self.last_update = None
        self.errors = []
        self.active_symbols = ['EUR_USD', 'GBP_USD', 'USD_JPY']
        self.trading_stats = {
            'win_rate': 0,
            'winning_trades': 0,
            'losing_trades': 0
        }
        logger.info("Trading state initialized successfully")

    def start_trading(self):
        """Start trading"""
        self.is_trading = True
        self.last_update = datetime.now()
        logger.info("Trading started")

    def stop_trading(self):
        """Stop trading"""
        self.is_trading = False
        self.last_update = datetime.now()
        logger.info("Trading stopped")

    def add_error(self, error: str):
        """Add error message"""
        self.errors.append({
            'message': error,
            'timestamp': datetime.now().isoformat()
        })
        logger.error(f"Trading error: {error}")

    def clear_errors(self):
        """Clear error messages"""
        self.errors = []
        logger.info("Errors cleared")

    def update_stats(self, win: bool):
        """Update trading statistics"""
        if win:
            self.trading_stats['winning_trades'] += 1
        else:
            self.trading_stats['losing_trades'] += 1
            
        total_trades = (self.trading_stats['winning_trades'] + 
                       self.trading_stats['losing_trades'])
        
        if total_trades > 0:
            self.trading_stats['win_rate'] = round(
                (self.trading_stats['winning_trades'] / total_trades) * 100, 2
            ) 