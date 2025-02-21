from typing import Dict, List
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class TradingStrategyService:
    def __init__(self):
        self.risk_levels = {
            'LOW': {'max_position_size': 0.02, 'stop_loss': 0.01},
            'MEDIUM': {'max_position_size': 0.05, 'stop_loss': 0.02},
            'HIGH': {'max_position_size': 0.10, 'stop_loss': 0.03}
        }

    def generate_strategy(self, 
                         analysis: Dict,
                         account_size: float,
                         risk_preference: str = 'MEDIUM') -> Dict:
        """Generate trading strategy based on analysis"""
        try:
            # Calculate position size
            position_size = self._calculate_position_size(
                account_size,
                self.risk_levels[risk_preference]['max_position_size']
            )
            
            # Calculate entry and exit points
            entry_points = self._calculate_entry_points(analysis)
            exit_points = self._calculate_exit_points(
                entry_points,
                self.risk_levels[risk_preference]['stop_loss']
            )
            
            return {
                'recommendation': self._get_recommendation(analysis),
                'position_size': position_size,
                'entry_points': entry_points,
                'exit_points': exit_points,
                'risk_level': risk_preference,
                'stop_loss': exit_points['stop_loss'],
                'take_profit': exit_points['take_profit']
            }
            
        except Exception as e:
            logger.error(f"Error generating strategy: {e}")
            return self._get_default_strategy()

    def _calculate_position_size(self, account_size: float, max_risk: float) -> float:
        """Calculate optimal position size based on risk management"""
        try:
            # Never risk more than 2% of account on a single trade
            max_loss = account_size * max_risk
            return max_loss
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0

    def _calculate_entry_points(self, analysis: Dict) -> Dict:
        """Calculate entry points based on technical levels"""
        try:
            current_price = analysis['market_data']['current_price']
            volatility = analysis['market_data']['indicators']['volatility']
            
            return {
                'primary': current_price,
                'secondary': [
                    current_price * (1 - volatility * 0.1),  # Lower entry
                    current_price * (1 + volatility * 0.1)   # Higher entry
                ],
                'type': 'LIMIT',
                'rationale': self._generate_entry_rationale(analysis)
            }
        except Exception as e:
            logger.error(f"Error calculating entry points: {e}")
            return self._get_default_entry_points()

    def _calculate_exit_points(self, entry_points: Dict, stop_loss_pct: float) -> Dict:
        """Calculate stop loss and take profit levels"""
        try:
            entry_price = entry_points['primary']
            return {
                'stop_loss': entry_price * (1 - stop_loss_pct),
                'take_profit': [
                    entry_price * 1.02,  # First target
                    entry_price * 1.05   # Second target
                ],
                'trailing_stop': {
                    'initial': stop_loss_pct,
                    'step': stop_loss_pct / 2
                }
            }
        except Exception as e:
            logger.error(f"Error calculating exit points: {e}")
            return self._get_default_exit_points() 