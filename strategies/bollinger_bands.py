import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class BollingerBandsStrategy:
    def __init__(self, broker, parameters=None):
        self.broker = broker
        self.parameters = parameters or {
            'bb_length': 20,
            'bb_std': 2.0,
            'cash_at_risk': 0.1
        }
        self._continue = False
        self.symbol = 'EUR_USD'  # Default trading pair
        self.last_check_time = None
        self.status_updates = []  # Store recent status updates
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
        
    def should_continue(self):
        return self._continue
        
    def stop(self):
        self._continue = False
        logger.info("Strategy stopped")
        
    def start(self):
        self._continue = True
        logger.info("Strategy started")
        
    def calculate_position_size(self, cash_at_risk, current_price):
        position_size = (cash_at_risk * self.broker._cash) / current_price
        return round(position_size)
        
    def calculate_indicators(self, prices):
        df = pd.DataFrame(prices, columns=['price'])
        df['SMA'] = df['price'].rolling(window=self.parameters['bb_length']).mean()
        df['STD'] = df['price'].rolling(window=self.parameters['bb_length']).std()
        df['Upper'] = df['SMA'] + (df['STD'] * self.parameters['bb_std'])
        df['Lower'] = df['SMA'] - (df['STD'] * self.parameters['bb_std'])
        return df
        
    def check_entry_signals(self, current_price, indicators):
        if current_price > indicators['Upper'].iloc[-1]:
            return 'sell'
        elif current_price < indicators['Lower'].iloc[-1]:
            return 'buy'
        return None 
    
    def run_iteration(self):
        """Execute one iteration of the strategy"""
        try:
            current_time = datetime.now()
            current_price = self.broker.get_last_price(self.symbol)
            
            if not current_price:
                self.log_status("Could not get current price")
                return
                
            # Get historical prices for indicator calculation
            prices = []
            for i in range(self.parameters['bb_length'] + 1):
                price = self.broker.get_last_price(self.symbol)
                if price:
                    prices.append({'price': price, 'time': current_time - timedelta(minutes=i)})
            
            if len(prices) < self.parameters['bb_length']:
                self.log_status("Not enough price data")
                return
                
            # Calculate indicators
            indicators = self.calculate_indicators(prices)
            
            # Get current position
            positions = self.broker.get_tracked_positions()
            current_position = positions.get(self.symbol)
            
            # Log current market state
            self.log_status(
                f"Price: {current_price:.5f}, "
                f"Upper: {indicators['Upper'].iloc[-1]:.5f}, "
                f"Lower: {indicators['Lower'].iloc[-1]:.5f}, "
                f"Position: {current_position['quantity'] if current_position else 'None'}"
            )
            
            # Check for trading signals
            signal = self.check_entry_signals(current_price, indicators)
            
            if signal:
                if not current_position:
                    # Calculate position size
                    cash_at_risk = self.parameters['cash_at_risk']
                    position_size = self.calculate_position_size(cash_at_risk, current_price)
                    
                    # Log potential trade
                    self.log_status(
                        f"Signal: {signal.upper()}, "
                        f"Size: {position_size}, "
                        f"Price: {current_price:.5f}"
                    )
                    
                    # Submit order
                    if position_size > 0:
                        order = {
                            'symbol': self.symbol,
                            'quantity': position_size,
                            'side': signal
                        }
                        order_id = self.broker.submit_order(order)
                        if order_id:
                            self.log_status(f"Order executed: {order_id}")
                        else:
                            self.log_status("Order submission failed")
                else:
                    self.log_status("Already in position, skipping signal")
            
            self.last_check_time = current_time
            
        except Exception as e:
            self.log_status(f"Error in strategy: {str(e)}")
            logger.error(f"Strategy error: {e}", exc_info=True)
    
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
        """Update performance metrics after a trade"""
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
        self.performance['max_drawdown'] = min(self.performance['max_drawdown'], current_drawdown)
        
        # Add to trade history
        self.trades_history.append(trade_result)
        if len(self.trades_history) > 100:  # Keep last 100 trades
            self.trades_history.pop(0)

    def analyze_market(self, current_price, indicators):
        """Analyze current market conditions"""
        sma = indicators['SMA'].iloc[-1]
        upper_band = indicators['Upper'].iloc[-1]
        lower_band = indicators['Lower'].iloc[-1]
        
        # Calculate volatility and trend strength
        volatility = (upper_band - lower_band) / sma * 100
        trend_strength = abs(current_price - sma) / (upper_band - lower_band) * 100
        
        # Determine market conditions
        if current_price > upper_band:
            market_condition = "Overbought"
        elif current_price < lower_band:
            market_condition = "Oversold"
        else:
            market_condition = "Neutral"
            
        self.last_analysis = {
            'timestamp': datetime.now(),
            'price': current_price,
            'sma': sma,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'volatility': volatility,
            'trend_strength': trend_strength,
            'market_condition': market_condition
        }
        
        return self.last_analysis

    def get_status(self):
        """Enhanced status information"""
        return {
            'running': self._continue,
            'symbol': self.symbol,
            'parameters': self.parameters,
            'last_check': self.last_check_time.isoformat() if self.last_check_time else None,
            'recent_updates': self.status_updates[-10:],
            'performance': self.performance,
            'current_trade': self.current_trade,
            'market_analysis': self.last_analysis,
            'trades_history': self.trades_history[-5:],  # Last 5 trades
            'total_trades_today': len([t for t in self.trades_history if t['timestamp'].date() == datetime.now().date()]),
            'daily_profit_loss': sum(t['profit_loss'] for t in self.trades_history if t['timestamp'].date() == datetime.now().date())
        } 