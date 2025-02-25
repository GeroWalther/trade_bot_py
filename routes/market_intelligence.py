from quart import Blueprint, jsonify, request
from services.market_intelligence_service import MarketIntelligenceService
import os
import logging

# Configure logger
logger = logging.getLogger(__name__)

market_bp = Blueprint('market_intelligence', __name__)

# Initialize service at module level
market_service = MarketIntelligenceService()

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