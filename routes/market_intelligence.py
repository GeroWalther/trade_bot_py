from quart import Blueprint, jsonify, request
from services.market_intelligence_service import MarketIntelligenceService
from services.market_data_service import MarketDataService
from services.ai_analysis_service_new import AIAnalysisService
import os
import logging
from functools import wraps
import time
from collections import defaultdict

# Configure logger
logger = logging.getLogger(__name__)

market_bp = Blueprint('market_intelligence', __name__)

# Initialize service at module level
market_service = MarketIntelligenceService()
market_data_service = MarketDataService(alpha_vantage_key=os.getenv('ALPHA_VANTAGE_API_KEY'))
ai_analysis_service = AIAnalysisService()

# Simple rate limiting
request_counts = defaultdict(lambda: {'count': 0, 'reset_time': 0})

def rate_limit(requests_per_minute=30):
    def decorator(f):
        @wraps(f)
        async def wrapped(*args, **kwargs):
            now = time.time()
            key = f.__name__
            
            # Reset counter if minute has passed
            if now > request_counts[key]['reset_time']:
                request_counts[key] = {
                    'count': 0,
                    'reset_time': now + 60
                }
            
            # Check rate limit
            if request_counts[key]['count'] >= requests_per_minute:
                return jsonify({'error': 'Rate limit exceeded'}), 429
                
            request_counts[key]['count'] += 1
            return await f(*args, **kwargs)
        return wrapped
    return decorator

@market_bp.route('/api/market-intelligence')
async def get_market_intelligence():
    try:
        analysis = await market_service.get_market_analysis()
        return jsonify(analysis)
    except Exception as e:
        return jsonify({
            'error': True,
            'message': str(e)
        }), 500

@market_bp.route('/api/economic-indicators')
async def get_economic_indicators():
    """Get economic indicators from FRED"""
    try:
        indicators = await market_service.get_economic_indicators()
        logger.info(f"All indicators: {indicators}")  # Log everything
        return jsonify(indicators)
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

@market_bp.route('/clear-cache', methods=['POST'])
async def clear_cache():
    """Clear all caches"""
    try:
        success = market_service.clear_cache()
        if success:
            return {'message': 'Cache cleared successfully'}, 200
        else:
            return {'error': 'Failed to clear cache'}, 500
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return {'error': str(e)}, 500

@market_bp.route('/api/historical-prices/<symbol>')
@rate_limit(requests_per_minute=30)
async def get_historical_prices(symbol):
    try:
        # Get timeframe from query parameters, default to Intraday
        timeframe = request.args.get('timeframe', 'Intraday')
        logger.info(f"Price request for {symbol} with {timeframe} timeframe")
        
        prices = await market_data_service.get_historical_prices(symbol, timeframe)
        return jsonify(prices)
    except Exception as e:
        logger.error(f"Error in route: {e}")
        return jsonify([])

@market_bp.route('/api/test-indicators/<symbol>')
@rate_limit(requests_per_minute=30)
async def test_indicators(symbol):
    try:
        # Log all request parameters
        logger.info(f"Request args: {dict(request.args)}")
        timeframe = request.args.get('timeframe', 'SWING')
        logger.info(f"Processing test_indicators request: symbol={symbol}, timeframe={timeframe}")
        
        analysis = await ai_analysis_service.get_technical_analysis(symbol, timeframe)
        logger.info(f"Analysis completed. Config used: timeframe={timeframe}, status={analysis.get('status')}")
        
        return jsonify({
            'status': 'success',
            'data': analysis
        })
    except Exception as e:
        logger.error(f"Error in test_indicators: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }) 