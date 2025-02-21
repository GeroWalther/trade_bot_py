from quart import Blueprint, jsonify, request
from services.market_intelligence_service import MarketIntelligenceService
from services.ai_analysis_service import AIAnalysisService
from config import OANDA_CREDS
import logging
import os

logger = logging.getLogger(__name__)
analysis_bp = Blueprint('analysis', __name__, url_prefix='/api')

# Initialize services
market_intelligence = MarketIntelligenceService()
ai_analysis = AIAnalysisService(os.getenv('ALPHA_VANTAGE_KEY'))

@analysis_bp.route('/analyze-asset', methods=['POST'])
async def analyze_asset():
    try:
        data = await request.get_json()
        logger.info(f"Received analysis request: {data}")
        
        asset = data.get('asset')
        timeframe = data.get('timeframe', 'SWING')
        
        if not asset:
            return jsonify({
                'status': 'error',
                'message': 'Asset symbol is required'
            }), 400
        
        logger.info(f"Starting analysis for {asset} with timeframe {timeframe}")
        
        try:
            market_data = await market_intelligence.get_asset_analysis(
                asset=asset,
                timeframe=timeframe
            )
            
            logger.info(f"Market data received for {asset}")
            
            ai_analysis_result = await ai_analysis.generate_market_analysis(
                asset=asset,
                market_data=market_data['market_data'],
                economic_data=market_data['economic_data'],
                news_data=market_data['news'],
                sentiment_score=market_data['sentiment']['score']
            )
            
            logger.info(f"AI analysis completed for {asset}")
            
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
            logger.error(f"Error processing analysis: {e}", exc_info=True)
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
            
    except Exception as e:
        logger.error(f"Error parsing request: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': 'Invalid request format'
        }), 400 