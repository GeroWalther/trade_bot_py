from typing import Dict

class RiskManagementService:
    def __init__(self):
        self.max_daily_risk = 0.05  # 5% max daily risk
        self.max_position_risk = 0.02  # 2% max per position
        self.correlation_threshold = 0.7

    def validate_trade(self, trade: Dict, portfolio: Dict) -> Dict:
        """Validate trade against risk management rules"""
        try:
            checks = {
                'position_size': self._check_position_size(trade, portfolio),
                'correlation': self._check_correlation(trade, portfolio),
                'exposure': self._check_total_exposure(trade, portfolio),
                'drawdown': self._check_drawdown(portfolio)
            }
            
            return {
                'valid': all(check['passed'] for check in checks.values()),
                'checks': checks,
                'recommendations': self._generate_risk_recommendations(checks)
            }
        except Exception as e:
            logger.error(f"Error validating trade: {e}")
            return {'valid': False, 'error': str(e)} 