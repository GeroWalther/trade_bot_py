from quart import Blueprint, jsonify, request
from services.ai_analysis_service import AIAnalysisService
import logging
import traceback
from anthropic import Anthropic
import os
import openai
import json
import aiohttp
import asyncio

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ai_analysis_bp = Blueprint('ai_analysis', __name__)

# Initialize Anthropic client
anthropic = Anthropic()

# Initialize OpenAI client with better error handling
openai_api_key = os.getenv('OPENAI_API_KEY')
openai_client = None

if not openai_api_key:
    logger.error("OPENAI_API_KEY not found in environment variables")
else:
    try:
        logger.info(f"Initializing OpenAI client with API key: {openai_api_key[:8]}...")
        openai_client = openai.OpenAI(api_key=openai_api_key)
        logger.info("OpenAI client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {str(e)}")

# Function to fetch real-time market data from Yahoo Finance
async def fetch_market_data(symbol):
    try:
        # For Nasdaq, we use the ^IXIC symbol
        yahoo_symbol = "%5EIXIC" if symbol == "Nasdaq" else symbol
        
        async with aiohttp.ClientSession() as session:
            async with session.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}", timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('chart', {}).get('result'):
                        result = data['chart']['result'][0]
                        meta = result.get('meta', {})
                        
                        # Extract current price and other relevant data
                        current_price = meta.get('regularMarketPrice')
                        previous_close = meta.get('previousClose')
                        day_high = meta.get('dayHigh')
                        day_low = meta.get('dayLow')
                        
                        return {
                            'current_price': current_price,
                            'previous_close': previous_close,
                            'day_high': day_high,
                            'day_low': day_low,
                            'symbol': meta.get('symbol'),
                            'exchange': meta.get('exchangeName')
                        }
        
        logger.error(f"Failed to fetch market data for {symbol}")
        return None
    except Exception as e:
        logger.error(f"Error fetching market data: {str(e)}")
        return None

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
            logger.info("Sending request to Claude API...")
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
            logger.error(f"Claude API error: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': 'AI analysis service temporarily unavailable'
            }), 503

    except Exception as e:
        logger.error(f"Request processing error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Internal server error'
        }), 500

@ai_analysis_bp.route('/api/advanced-market-analysis', methods=['POST'])
async def advanced_market_analysis():
    try:
        data = await request.get_json()
        logger.info(f"Received advanced AI analysis request for asset: {data.get('asset')}")

        # Extract data from request
        asset = data.get('asset')
        term = data.get('term')
        risk_level = data.get('riskLevel')

        if not all([asset, term, risk_level]):
            logger.error("Missing required parameters")
            return jsonify({
                'status': 'error',
                'message': 'Missing required parameters: asset, term, and riskLevel are required'
            }), 400

        # Always fetch market data from Yahoo Finance
        logger.info(f"Fetching current market data for {asset} from Yahoo Finance")
        market_data = await fetch_market_data(asset)
        
        if not market_data or 'current_price' not in market_data:
            logger.warning(f"Failed to fetch market data for {asset}, proceeding with limited data")
            # Create a minimal market data object if fetch failed
            market_data = {
                'current_price': None,
                'symbol': asset,
                'exchange': 'Unknown'
            }
        else:
            logger.info(f"Successfully fetched market data. Current price: {market_data['current_price']}")
        
        # Check if OpenAI client is initialized
        if not openai_client:
            logger.error("OpenAI client not initialized. Check your API key.")
            
            # For testing purposes, return a mock response instead of an error
            # This allows frontend testing without a valid OpenAI API key
            mock_data = {
                "market_summary": "This is a mock response for testing purposes. The OpenAI API key is not configured correctly.",
                "key_drivers": ["Mock driver 1", "Mock driver 2", "Mock driver 3"],
                "technical_analysis": "Mock technical analysis for testing purposes.",
                "risk_assessment": "Mock risk assessment for testing purposes.",
                "trading_strategy": {
                    "direction": "LONG",
                    "rationale": "This is a mock strategy for testing purposes.",
                    "entry": {
                        "price": "14,500",
                        "rationale": "Mock entry rationale"
                    },
                    "stop_loss": {
                        "price": "14,000",
                        "rationale": "Mock stop loss rationale"
                    },
                    "take_profit_1": {
                        "price": "15,000",
                        "rationale": "Mock TP1 rationale"
                    },
                    "take_profit_2": {
                        "price": "15,500",
                        "rationale": "Mock TP2 rationale"
                    }
                }
            }
            
            # Add current market price to the mock response if available
            if market_data and market_data.get('current_price'):
                mock_data['current_market_price'] = market_data['current_price']
            
            return jsonify({
                'status': 'success',
                'data': mock_data,
                'mock': True
            })

        # Prepare system message for ChatGPT
        system_message = """
        You are an expert financial analyst and trader with deep knowledge of markets, technical analysis, and trading strategies.
        Your task is to analyze the requested asset and provide a comprehensive trading strategy.
        
        You should:
        1. Research and analyze the current market conditions for the asset
        2. Consider relevant news, Twitter/social media sentiment, historical price data, technical indicators, and macroeconomic factors
        3. Provide a detailed analysis with clear trading recommendations
        4. Format your response as a JSON object with the following structure:
        
        {
            "market_summary": "Comprehensive summary of current market conditions",
            "key_drivers": ["List of key market drivers and factors"],
            "technical_analysis": "Detailed technical analysis with key indicators",
            "risk_assessment": "Assessment of market risks",
            "trading_strategy": {
                "direction": "LONG or SHORT",
                "rationale": "Explanation of the strategy direction",
                "entry": {
                    "price": "Recommended entry price or range",
                    "rationale": "Rationale for entry point"
                },
                "stop_loss": {
                    "price": "Recommended stop loss price",
                    "rationale": "Rationale for stop loss placement"
                },
                "take_profit_1": {
                    "price": "First take profit target",
                    "rationale": "Rationale for TP1"
                },
                "take_profit_2": {
                    "price": "Second take profit target",
                    "rationale": "Rationale for TP2"
                }
            }
        }
        
        Your response MUST be a valid JSON object with this exact structure.
        
        IMPORTANT: Make sure your price targets are realistic and close to the current market price. For swing trades, entry points should typically be within 5% of the current price, not requiring massive market moves to trigger.
        """

        # Prepare user message with real-time market data
        current_price_info = ""
        if market_data and market_data.get('current_price'):
            current_price = market_data.get('current_price')
            current_price_info = f"""
            Current Market Data:
            - Current Price: {current_price}
            """
            
            # Add additional market data if available
            if market_data.get('previous_close'):
                current_price_info += f"- Previous Close: {market_data['previous_close']}\n"
            if market_data.get('day_high'):
                current_price_info += f"- Day High: {market_data['day_high']}\n"
            if market_data.get('day_low'):
                current_price_info += f"- Day Low: {market_data['day_low']}\n"
            if market_data.get('exchange'):
                current_price_info += f"- Exchange: {market_data['exchange']}\n"

        user_message = f"""
        Please provide an advanced market analysis and trading strategy for:
        
        Asset: {asset}
        Trading Term: {term}
        Risk Level: {risk_level}
        
        {current_price_info}
        
        I need a comprehensive analysis that includes:
        - Current market conditions and sentiment
        - Technical analysis with key indicators
        - Relevant news and social media sentiment
        - Macroeconomic factors affecting the asset
        - A detailed trading strategy with entry, stop loss, and take profit levels
        
        IMPORTANT: Make sure your price targets are realistic and close to the current market price. For swing trades, entry points should typically be within 5% of the current price, not requiring massive market moves to trigger.
        
        Please format your response as a JSON object as specified in your instructions.
        """

        try:
            # Call OpenAI API
            logger.info("Sending request to OpenAI API...")
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.2,
                max_tokens=1000,
                timeout=25
            )

            logger.info("Received response from ChatGPT")
            
            # Parse the JSON response
            try:
                analysis_json = json.loads(response.choices[0].message.content)
                logger.info("Successfully parsed JSON response")
                
                # Add current market price to the response
                if market_data and market_data.get('current_price'):
                    analysis_json['current_market_price'] = market_data['current_price']
                
                return jsonify({
                    'status': 'success',
                    'data': analysis_json
                })
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                return jsonify({
                    'status': 'error',
                    'message': 'Failed to parse AI response'
                }), 500

        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            logger.error(traceback.format_exc())
            
            # For testing purposes, return a mock response instead of an error
            mock_data = {
                "market_summary": f"Mock response due to API error: {str(e)}",
                "key_drivers": ["API Error", "Mock data for testing", "Check server logs"],
                "technical_analysis": "Mock technical analysis for testing purposes.",
                "risk_assessment": "Mock risk assessment for testing purposes.",
                "trading_strategy": {
                    "direction": "NEUTRAL",
                    "rationale": "This is a mock strategy due to an API error.",
                    "entry": {
                        "price": "14,500",
                        "rationale": "Mock entry rationale"
                    },
                    "stop_loss": {
                        "price": "14,000",
                        "rationale": "Mock stop loss rationale"
                    },
                    "take_profit_1": {
                        "price": "15,000",
                        "rationale": "Mock TP1 rationale"
                    },
                    "take_profit_2": {
                        "price": "15,500",
                        "rationale": "Mock TP2 rationale"
                    }
                }
            }
            
            # Add current market price to the mock response if available
            if market_data and market_data.get('current_price'):
                mock_data['current_market_price'] = market_data['current_price']
            
            return jsonify({
                'status': 'success',
                'data': mock_data,
                'mock': True,
                'error': str(e)
            })

    except Exception as e:
        logger.error(f"Request processing error: {str(e)}")
        logger.error(traceback.format_exc())
        
        # For testing purposes, return a mock response instead of an error
        mock_data = {
            "market_summary": f"Mock response due to processing error: {str(e)}",
            "key_drivers": ["Processing Error", "Mock data for testing", "Check server logs"],
            "technical_analysis": "Mock technical analysis for testing purposes.",
            "risk_assessment": "Mock risk assessment for testing purposes.",
            "trading_strategy": {
                "direction": "NEUTRAL",
                "rationale": "This is a mock strategy due to a processing error.",
                "entry": {
                    "price": "14,500",
                    "rationale": "Mock entry rationale"
                },
                "stop_loss": {
                    "price": "14,000",
                    "rationale": "Mock stop loss rationale"
                },
                "take_profit_1": {
                    "price": "15,000",
                    "rationale": "Mock TP1 rationale"
                },
                "take_profit_2": {
                    "price": "15,500",
                    "rationale": "Mock TP2 rationale"
                }
            }
        }
        
        return jsonify({
            'status': 'success',
            'data': mock_data,
            'mock': True,
            'error': str(e)
        }) 