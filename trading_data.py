class TradingDataStore:
    def __init__(self):
        self.data = {
            'cash_available': 0,
            'total_portfolio_value': 0,
            'daily_return_pct': 0,
            'positions': {},
            'trading_stats': {
                'win_rate': 0,
                'winning_trades': 0,
                'losing_trades': 0
            },
            'market_status': {
                'is_market_open': False,
                'active_symbols': []
            }
        }

    def update_data(self, new_data):
        self.data.update(new_data)

    def get_data(self):
        return self.data

# Global instance
store = TradingDataStore() 