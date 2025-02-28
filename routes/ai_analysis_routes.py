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
import time

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

# Simple cache for market data
market_data_cache = {}
CACHE_EXPIRY = 60  # Cache expiry in seconds

# Function to fetch real-time market data from Yahoo Finance
async def fetch_market_data(symbol):
    # Check cache first
    current_time = time.time()
    cache_key = symbol.lower()
    
    if cache_key in market_data_cache:
        cache_entry = market_data_cache[cache_key]
        # If cache is still valid (less than CACHE_EXPIRY seconds old)
        if current_time - cache_entry['timestamp'] < CACHE_EXPIRY:
            logger.info(f"Using cached market data for {symbol}, age: {current_time - cache_entry['timestamp']:.1f} seconds")
            return cache_entry['data']
        else:
            logger.info(f"Cache expired for {symbol}, fetching fresh data")
    
    try:
        # For Nasdaq, we use the ^IXIC symbol
        if symbol.lower() == "nasdaq":
            primary_symbol = "%5EIXIC"  # URL encoded ^IXIC
            fallback_symbol = "QQQ"     # QQQ ETF as fallback
            logger.info(f"Using Yahoo Finance symbol %5EIXIC for Nasdaq with QQQ as fallback")
        else:
            primary_symbol = symbol
            fallback_symbol = None
            
        # Try primary symbol first
        market_data = await _fetch_from_yahoo(primary_symbol)
        
        # If primary symbol fails and we have a fallback, try that
        if market_data.get('error') and fallback_symbol:
            logger.info(f"Primary symbol failed, trying fallback symbol: {fallback_symbol}")
            market_data = await _fetch_from_yahoo(fallback_symbol)
            
            # If fallback succeeds, adjust the data to represent the original symbol
            if not market_data.get('error'):
                market_data['symbol'] = symbol
                market_data['note'] = f"Data from related symbol: {fallback_symbol}"
        
        # Cache successful results
        if not market_data.get('error'):
            market_data_cache[cache_key] = {
                'data': market_data,
                'timestamp': current_time
            }
            
        return market_data
    except Exception as e:
        logger.error(f"Error fetching market data: {str(e)}")
        return {
            'error': True,
            'message': f"Error fetching market data: {str(e)}",
            'status_code': 500
        }

async def _fetch_from_yahoo(yahoo_symbol):
    """Helper function to fetch data from Yahoo Finance for a specific symbol"""
    try:
        logger.info(f"Fetching market data using Yahoo symbol: {yahoo_symbol}")
        
        async with aiohttp.ClientSession() as session:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}"
            logger.info(f"Making request to: {url}")
            
            # Add a random delay between 0.5 and 1.5 seconds to avoid rate limiting
            await asyncio.sleep(0.5 + (time.time() % 1))
            
            async with session.get(url, timeout=5, 
                                  headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Received response from Yahoo Finance")
                    
                    if data.get('chart', {}).get('result'):
                        result = data['chart']['result'][0]
                        meta = result.get('meta', {})
                        
                        # Extract current price and other relevant data
                        current_price = meta.get('regularMarketPrice')
                        previous_close = meta.get('previousClose')
                        day_high = meta.get('dayHigh')
                        day_low = meta.get('dayLow')
                        
                        logger.info(f"Current price for {yahoo_symbol}: {current_price}")
                        
                        return {
                            'current_price': current_price,
                            'previous_close': previous_close,
                            'day_high': day_high,
                            'day_low': day_low,
                            'symbol': meta.get('symbol'),
                            'exchange': meta.get('exchangeName')
                        }
                    else:
                        logger.error(f"No result data in Yahoo Finance response for {yahoo_symbol}")
                        return {
                            'error': True,
                            'message': f"No data available for {yahoo_symbol}",
                            'status_code': response.status
                        }
                else:
                    error_msg = f"Yahoo Finance API returned status code {response.status}"
                    if response.status == 429:
                        error_msg = "Rate limit exceeded when accessing market data. Please try again later."
                    logger.error(error_msg)
                    return {
                        'error': True,
                        'message': error_msg,
                        'status_code': response.status
                    }
    except Exception as e:
        logger.error(f"Error fetching from Yahoo for {yahoo_symbol}: {str(e)}")
        return {
            'error': True,
            'message': f"Error fetching market data: {str(e)}",
            'status_code': 500
        }

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
        
        # Check if there was an error fetching market data
        if market_data.get('error'):
            logger.error(f"Error fetching market data: {market_data.get('message')}")
            return jsonify({
                'status': 'error',
                'message': market_data.get('message', 'Failed to fetch current market data')
            }), market_data.get('status_code', 500)
        
        logger.info(f"Successfully fetched market data. Current price: {market_data['current_price']}")
        
        # Check if OpenAI client is initialized
        if not openai_client:
            logger.error("OpenAI client not initialized. Check your API key.")
            return jsonify({
                'status': 'error',
                'message': 'AI service is not properly configured. Please check your API key.'
            }), 500

        # Prepare system message for ChatGPT
        system_message = """
        You are an expert financial analyst and trader with deep knowledge of markets, technical analysis, and trading strategies.
        Your task is to analyze the requested asset and provide a comprehensive trading strategy.
        
        You have the ability to search the web for current market data if needed. For Nasdaq data, you can check https://finance.yahoo.com/quote/%5EIXIC/
        
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
        
        If you need current market data, you can search for it at https://finance.yahoo.com/quote/%5EIXIC/ for Nasdaq.
        
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
            
            return jsonify({
                'status': 'error',
                'message': f"AI analysis service error: {str(e)}"
            }), 500

    except Exception as e:
        logger.error(f"Request processing error: {str(e)}")
        logger.error(traceback.format_exc())
        
        return jsonify({
            'status': 'error',
            'message': f"Server error: {str(e)}"
        }), 500 