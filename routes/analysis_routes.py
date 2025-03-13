from quart import Blueprint, jsonify, request, current_app
from services.market_analyzer import MarketAnalyzer
# from services.ai_analysis_service_new import AIAnalysisService  # ML service removed
import logging
import traceback

logger = logging.getLogger(__name__)
analysis_bp = Blueprint('analysis', __name__, url_prefix='/api')

# @analysis_bp.route('/analyze-asset', methods=['POST'])  # ML route removed
# async def analyze_asset():
#     try:
#         # Initialize MarketAnalyzer with the API key from app config
#         market_analyzer = MarketAnalyzer(
#             alpha_vantage_key=current_app.market_data.alpha_vantage_key
#         )
#         
#         data = await request.get_json()
#         asset = data.get('asset')
#         timeframe = data.get('timeframe', 'MEDIUM_TERM')
#         risk_level = data.get('risk_level', 'MEDIUM')
#         
#         analysis = await market_analyzer.analyze_market(
#             symbol=asset,
#             timeframe=timeframe,
#             risk_level=risk_level
#         )
#         
#         return jsonify({
#             'status': 'success',
#             'data': {
#                 'analysis': analysis
#             }
#         })
#     except Exception as e:
#         logger.error(f"Error processing analysis: {e}")
#         return jsonify({
#             'status': 'error',
#             'message': str(e)
#         }), 500

@analysis_bp.route('/test-connection', methods=['GET'])
async def test_connection():
    try:
        market_analyzer = MarketAnalyzer(
            alpha_vantage_key=current_app.market_data.alpha_vantage_key
        )
        
        from oandapyV20.endpoints.accounts import AccountSummary
        
        r = AccountSummary(current_app.config['OANDA_CREDS']['ACCOUNT_ID'])
        market_analyzer.client.request(r)
        
        return jsonify({
            'status': 'success',
            'message': 'OANDA connection successful',
            'account_info': r.response
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500

@analysis_bp.route('/test-indicators/<symbol>', methods=['GET'])
async def test_indicators(symbol):
    try:
        # Use the app's initialized service instead of creating a new one
        ai_service = current_app.ai_analysis
        if not ai_service:
            return jsonify({
                'status': 'error',
                'message': 'AI Analysis service not initialized'
            }), 500
        
        # Get technical analysis
        result = await ai_service.get_technical_analysis(symbol)
        logger.info(f"Technical analysis result: {result}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in test indicators: {e}", exc_info=True)  # Add full traceback
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500 