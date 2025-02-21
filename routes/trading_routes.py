from quart import Blueprint, jsonify, request
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
        
        asset = data.get('asset')
        timeframe = data.get('timeframe', 'SWING')
        risk_level = data.get('risk_level', 'MEDIUM')
        account_size = data.get('account_size', 10000)

        # Get market data - pass the timeframe parameter
        market_data = await market_intelligence.get_asset_analysis(
            asset=asset,
            timeframe=timeframe  # Add this parameter
        )
        
        # Get AI analysis
        ai_analysis_result = await ai_analysis.generate_market_analysis(
            asset=asset,
            market_data=market_data['market_data'],
            economic_data=market_data['economic_data'],
            news_data=market_data['news'],
            sentiment_score=market_data['sentiment']['score']
        )

        return jsonify({
            'status': 'success',
            'data': {
                'analysis': ai_analysis_result,
                'macro': {
                    'aiAnalysis': {
                        'summary': ai_analysis_result['summary'],
                        'keyFactors': ai_analysis_result['key_factors'],
                        'recommendedStrategy': ai_analysis_result['trading_strategy']
                    }
                },
                'news': market_data['news']
            }
        })

    except Exception as e:
        logger.error(f"Error in analyze_asset: {e}", exc_info=True)
        # Return more detailed error message
        return jsonify({
            'status': 'error',
            'message': str(e),
            'details': {
                'asset': asset,
                'timeframe': timeframe,
                'error_type': type(e).__name__
            }
        }), 500 