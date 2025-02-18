class TradingDataStore:
    def __init__(self):
        self.data = {
            'account': {
                'balance': 0,
                'unrealized_pl': 0,
                'total_value': 0,
            },
            'market_prices': {
                'EUR_USD': None,
                'BTC_USD': None,
                'last_update': None
            },
            'trade_history': [],  # List of recent trades
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