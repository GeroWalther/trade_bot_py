from flask import Blueprint, jsonify, request
from services.market_intelligence_service import MarketIntelligenceService
import os
import asyncio
import logging

# Configure logger
logger = logging.getLogger(__name__)

market_bp = Blueprint('market_intelligence', __name__)
market_service = MarketIntelligenceService(
    alpha_vantage_key=os.getenv('ALPHA_VANTAGE_KEY'),
    news_api_key=os.getenv('NEWS_API_KEY')
)

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
def get_economic_indicators():
    try:
        category = request.args.get('category')
        asset_symbol = request.args.get('symbol')
        
        logger.info(f"Received request with category: {category}, symbol: {asset_symbol}")
        
        if not category:
            logger.error("No category provided in request")
            return jsonify({
                'economic_indicators': [],
                'message': 'Category parameter is required'
            }), 200  # Return 200 with empty data instead of error
            
        indicators = market_service.get_economic_indicators(category, asset_symbol)
        return jsonify(indicators)
        
    except Exception as e:
        logger.error(f"Error getting economic indicators: {e}")
        return jsonify({
            'economic_indicators': [],
            'message': str(e)
        }), 200  # Return 200 with empty data and message 