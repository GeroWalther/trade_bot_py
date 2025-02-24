from quart import Blueprint, jsonify, request
from services.market_intelligence_service import MarketIntelligenceService
import os
import logging

# Configure logger
logger = logging.getLogger(__name__)

market_bp = Blueprint('market_intelligence', __name__)
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
        logger.info("Fetching economic indicators from FRED")
        indicators = await market_service.get_economic_indicators()
        
        if not indicators:
            return jsonify({
                'status': 'error',
                'message': 'Failed to fetch economic indicators'
            }), 500
            
        logger.info(f"Successfully retrieved economic indicators: {indicators}")
        return jsonify(indicators)
        
    except Exception as e:
        logger.error(f"Error getting economic indicators: {e}")
        return jsonify({'error': str(e)}), 500 