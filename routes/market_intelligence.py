from flask import Blueprint, jsonify
from services.market_intelligence_service import MarketIntelligenceService
import os
import asyncio

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