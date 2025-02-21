import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import logging
from scipy import stats

logger = logging.getLogger(__name__)

class ProbabilityService:
    def __init__(self, monte_carlo_sims: int = 1000):
        self.monte_carlo_sims = monte_carlo_sims
        
    def calculate_monte_carlo(self, prices: pd.Series, days: int = 30) -> Dict:
        """Run Monte Carlo simulation for price prediction"""
        returns = prices.pct_change().dropna()
        mean, std = returns.mean(), returns.std()
        
        simulations = []
        last_price = prices.iloc[-1]
        
        for _ in range(self.monte_carlo_sims):
            prices = [last_price]
            for _ in range(days):
                price = prices[-1] * (1 + np.random.normal(mean, std))
                prices.append(price)
            simulations.append(prices)
            
        simulations = np.array(simulations)
        
        # Enhanced probability calculations
        final_prices = simulations[:, -1]
        returns_distribution = (final_prices - last_price) / last_price
        
        return {
            'upside_probability': np.mean(final_prices > last_price),
            'expected_return': np.mean(returns_distribution),
            'var_95': np.percentile(returns_distribution, 5),
            'var_99': np.percentile(returns_distribution, 1),
            'confidence_level': self._calculate_confidence(simulations),
            'price_ranges': {
                'likely_range': self._calculate_price_range(final_prices, 0.68),  # 1 std
                'possible_range': self._calculate_price_range(final_prices, 0.95), # 2 std
                'extreme_range': self._calculate_price_range(final_prices, 0.99)  # 3 std
            }
        }

    def backtest_strategy(self, 
                         prices: pd.DataFrame, 
                         strategy_params: Dict,
                         initial_capital: float = 10000.0) -> Dict:
        """Backtest a trading strategy"""
        try:
            # Initialize backtest variables
            capital = initial_capital
            position = 0
            trades = []
            
            # Calculate technical indicators
            prices['SMA20'] = prices['close'].rolling(window=20).mean()
            prices['SMA50'] = prices['close'].rolling(window=50).mean()
            prices['RSI'] = self._calculate_rsi(prices['close'])
            
            # Run backtest
            for i in range(50, len(prices)):
                date = prices.index[i]
                current_price = prices['close'].iloc[i]
                
                # Get signals based on strategy parameters
                signal = self._get_trading_signal(
                    prices.iloc[i],
                    strategy_params
                )
                
                # Execute trades
                if signal == 'BUY' and position <= 0:
                    # Calculate position size based on risk management
                    size = self._calculate_position_size(
                        capital,
                        current_price,
                        strategy_params['risk_per_trade']
                    )
                    position = size
                    trades.append({
                        'date': date,
                        'type': 'BUY',
                        'price': current_price,
                        'size': size,
                        'capital': capital
                    })
                    
                elif signal == 'SELL' and position >= 0:
                    if position > 0:
                        # Close long position
                        capital += position * (current_price - trades[-1]['price'])
                    position = 0
                    trades.append({
                        'date': date,
                        'type': 'SELL',
                        'price': current_price,
                        'size': 0,
                        'capital': capital
                    })
            
            # Calculate performance metrics
            performance = self._calculate_performance_metrics(trades, prices)
            
            return {
                'final_capital': capital,
                'total_return': (capital - initial_capital) / initial_capital * 100,
                'trades': trades,
                'metrics': performance
            }
            
        except Exception as e:
            logger.error(f"Error in backtest: {e}", exc_info=True)
            return None

    def _calculate_confidence(self, simulations: np.ndarray) -> float:
        """Calculate confidence level based on simulation consistency"""
        final_prices = simulations[:, -1]
        
        # Calculate statistical measures
        skew = stats.skew(final_prices)
        kurtosis = stats.kurtosis(final_prices)
        
        # More normal distribution = higher confidence
        normality = 1 - min(abs(skew) / 2, 0.5) - min(abs(kurtosis) / 6, 0.5)
        
        # Calculate trend consistency
        trends = np.diff(simulations, axis=1)
        trend_consistency = np.mean(np.sign(trends) == np.sign(np.mean(trends)))
        
        # Combine factors
        confidence = (normality * 0.5 + trend_consistency * 0.5) * 100
        return min(max(confidence, 0), 100)  # Ensure between 0-100

    def _calculate_price_range(self, prices: np.ndarray, confidence: float) -> Tuple[float, float]:
        """Calculate price range for a given confidence level"""
        lower = np.percentile(prices, (1 - confidence) * 100 / 2)
        upper = np.percentile(prices, (1 + confidence) * 100 / 2)
        return (lower, upper)

    def _calculate_performance_metrics(self, trades: List[Dict], prices: pd.DataFrame) -> Dict:
        """Calculate detailed performance metrics"""
        if not trades:
            return {}
            
        # Calculate basic metrics
        winning_trades = [t for t in trades if t['type'] == 'SELL' and t['capital'] > trades[trades.index(t)-1]['capital']]
        losing_trades = [t for t in trades if t['type'] == 'SELL' and t['capital'] <= trades[trades.index(t)-1]['capital']]
        
        win_rate = len(winning_trades) / (len(winning_trades) + len(losing_trades)) if trades else 0
        
        # Calculate returns
        returns = [(t['capital'] - trades[trades.index(t)-1]['capital']) / trades[trades.index(t)-1]['capital']
                  for t in trades if t['type'] == 'SELL']
        
        return {
            'win_rate': win_rate * 100,
            'avg_return': np.mean(returns) * 100 if returns else 0,
            'sharpe_ratio': self._calculate_sharpe_ratio(returns) if returns else 0,
            'max_drawdown': self._calculate_max_drawdown([t['capital'] for t in trades]),
            'profit_factor': self._calculate_profit_factor(winning_trades, losing_trades)
        }

    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe Ratio"""
        if not returns:
            return 0
        excess_returns = np.array(returns) - risk_free_rate/252  # Daily risk-free rate
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) != 0 else 0

    def _calculate_max_drawdown(self, capitals: List[float]) -> float:
        """Calculate maximum drawdown"""
        peak = capitals[0]
        max_dd = 0
        
        for capital in capitals:
            if capital > peak:
                peak = capital
            dd = (peak - capital) / peak
            max_dd = max(max_dd, dd)
            
        return max_dd * 100

    def _calculate_profit_factor(self, winning_trades: List[Dict], losing_trades: List[Dict]) -> float:
        """Calculate profit factor"""
        total_profits = sum(t['capital'] - trades[trades.index(t)-1]['capital'] for t in winning_trades)
        total_losses = sum(abs(t['capital'] - trades[trades.index(t)-1]['capital']) for t in losing_trades)
        
        return total_profits / total_losses if total_losses != 0 else float('inf') 