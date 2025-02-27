from quart import Blueprint, jsonify, request
from services.ai_analysis_service import AIAnalysisService
import logging
import traceback
from anthropic import Anthropic

logger = logging.getLogger(__name__)
ai_analysis_bp = Blueprint('ai_analysis', __name__)

# Initialize Anthropic client
anthropic = Anthropic()

@ai_analysis_bp.route('/api/ai-analysis', methods=['POST'])
async def analyze_market():
    try:
        data = await request.get_json()
        logger.info(f"Received AI analysis request for asset: {data.get('asset')}")

        # Extract data from request
        asset = data.get('asset')
        market_data = data.get('marketData')
        risk_level = data.get('riskLevel')
        timeframe = data.get('timeframe')

        if not all([asset, market_data, risk_level, timeframe]):
            logger.error("Missing required parameters")
            return jsonify({
                'status': 'error',
                'message': 'Missing required parameters'
            }), 400

        # Prepare context for AI analysis
        context = f"""
        Please analyze {asset} and provide a detailed trading strategy.

        Market Data:
        - Asset: {asset}
        - Timeframe: {timeframe}
        - Risk Level: {risk_level}
        - Current Price Data: {market_data.get('historicalPrices', {})}
        - Technical Indicators: {market_data.get('technicalIndicators', {})}
        - News: {market_data.get('news', {})}
        - Macro Indicators: {market_data.get('macroIndicators', {})}
        
        Please provide a comprehensive analysis including:
        1. Market Summary and Key Drivers
        2. Technical Analysis
        3. Risk Assessment
        4. Trading Strategy Recommendations
        5. Entry/Exit Points
        6. Risk Management Guidelines
        """

        try:
            # Create message with Anthropic's Claude
            response = anthropic.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=4000,
                temperature=0.1,
                system="You are an expert forex and financial markets trader. Analyze the provided market data and generate detailed trading recommendations.",
                messages=[{
                    "role": "user",
                    "content": context
                }]
            )

            logger.info("Received response from Claude")
            analysis_text = str(response.content)

            # Structure the response
            analysis_result = {
                'summary': analysis_text,
                'key_factors': [
                    "Technical Indicators Analysis",
                    "Economic Data Impact",
                    "Market Sentiment Analysis",
                    "Risk Assessment"
                ],
                'trading_strategy': {
                    'direction': market_data.get('trend', 'NEUTRAL'),
                    'entry': {
                        'price': market_data.get('current_price', 0),
                        'rationale': 'Based on technical analysis and market sentiment'
                    },
                    'stopLoss': {
                        'price': market_data.get('support_level', 0),
                        'rationale': 'Based on volatility and risk management'
                    }
                }
            }

            return jsonify({
                'status': 'success',
                'data': analysis_result
            })

        except Exception as e:
            logger.error(f"Claude API error: {str(e)}\n{traceback.format_exc()}")
            return jsonify({
                'status': 'error',
                'message': 'AI analysis service temporarily unavailable'
            }), 503

    except Exception as e:
        logger.error(f"Request processing error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'status': 'error',
            'message': 'Internal server error'
        }), 500 