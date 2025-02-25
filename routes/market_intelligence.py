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
        # Get parameters from request
        asset_type = request.args.get('asset_type')
        core_only = request.args.get('core', 'false').lower() == 'true'
        
        logger.info(f"Fetching indicators - asset_type: {asset_type}, core_only: {core_only}")
        
        # Initialize service if not already done
        if not hasattr(market_bp, 'market_intelligence'):
            market_bp.market_intelligence = MarketIntelligenceService()
        
        # Get indicators
        indicators = await market_bp.market_intelligence.get_economic_indicators(
            asset_type=asset_type if not core_only else None,
            core_only=core_only
        )
        
        if not indicators:
            logger.warning("No indicators returned")
            return jsonify({
                'status': 'error',
                'message': 'No indicators found'
            }), 404
            
        logger.info(f"Successfully retrieved {len(indicators)} indicators")
        return jsonify(indicators)
        
    except Exception as e:
        logger.error(f"Error getting economic indicators: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500 