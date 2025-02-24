from quart import Blueprint, jsonify, request, current_app
from services.market_intelligence_service import MarketIntelligenceService
from services.ai_analysis_service import AIAnalysisService
from services.trading_strategy_service import TradingStrategyService
from services.risk_management_service import RiskManagementService
from config import OANDA_CREDS
import logging

logger = logging.getLogger(__name__)
trading_bp = Blueprint('trading', __name__)

# Initialize services
market_intelligence = MarketIntelligenceService()
ai_analysis = AIAnalysisService(OANDA_CREDS['ACCESS_TOKEN'])
trading_strategy = TradingStrategyService()
risk_management = RiskManagementService()

@trading_bp.route('/api/analyze-asset', methods=['POST'])
async def analyze_asset():
    try:
        data = await request.get_json()
        logger.info(f"Received analysis request: {data}")
        
        symbol = data.get('asset')
        timeframe = data.get('timeframe', 'SWING')
        risk_level = data.get('risk_level', 'MEDIUM')
        
        # Get market data and analysis
        analysis_result = await market_intelligence.analyze_market(
            symbol=symbol,
            timeframe=timeframe,
            risk_level=risk_level
        )
        
        return jsonify(analysis_result)

    except Exception as e:
        logger.error(f"Error in analyze_asset: {e}", exc_info=True)
        return jsonify(market_intelligence._get_default_analysis())

@trading_bp.route('/health')
async def health_check():
    return jsonify({'status': 'healthy'})

@trading_bp.route('/trading-status')
async def get_trading_status():
    try:
        trading_state = current_app.trading_state
        return {
            'status': 'OK',
            'isTrading': trading_state.is_trading,
            'lastUpdate': trading_state.last_update.isoformat() if trading_state.last_update else None,
            'errors': trading_state.errors
        }
    except Exception as e:
        logger.error(f"Error getting trading status: {e}")
        return {
            'status': 'ERROR',
            'message': str(e)
        }, 500 