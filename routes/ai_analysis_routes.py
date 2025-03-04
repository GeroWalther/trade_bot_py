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
        # Map asset names to Yahoo Finance symbols
        yahoo_symbols = {
            "nasdaq": "%5EIXIC",  # ^IXIC
            "s&p500": "%5EGSPC",  # ^GSPC
            "gold": "GC%3DF",     # GC=F
            "usd/jpy": "JPY=X",   # JPY=X
            "btcusd": "BTC-USD",  # BTC-USD
            "eur/usd": "EUR=X"    # EUR=X
        }
        
        symbol_lower = symbol.lower()
        if symbol_lower in yahoo_symbols:
            yahoo_symbol = yahoo_symbols[symbol_lower]
            logger.info(f"Using Yahoo Finance symbol {yahoo_symbol} for {symbol}")
        else:
            # Default to using the symbol as-is if not in our mapping
            yahoo_symbol = symbol
            logger.info(f"No mapping found for {symbol}, using as-is")
        
        # Fetch from Yahoo Finance
        market_data = await _fetch_from_yahoo(yahoo_symbol)
        
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

        # Map asset names to their Yahoo Finance and TradingView URLs
        asset_urls = {
            "nasdaq": {
                "yahoo": "https://finance.yahoo.com/quote/%5EIXIC/",
                "tradingview": "https://www.tradingview.com/symbols/NASDAQ-IXIC/technicals/"
            },
            "s&p500": {
                "yahoo": "https://finance.yahoo.com/quote/%5EGSPC/",
                "tradingview": "https://www.tradingview.com/symbols/SPX/technicals/"
            },
            "gold": {
                "yahoo": "https://finance.yahoo.com/quote/GC%3DF/",
                "tradingview": "https://www.tradingview.com/symbols/COMEX-GC1!/technicals/"
            },
            "usd/jpy": {
                "yahoo": "https://finance.yahoo.com/quote/JPY=X/",
                "tradingview": "https://www.tradingview.com/symbols/USDJPY/technicals/"
            },
            "btcusd": {
                "yahoo": "https://finance.yahoo.com/quote/BTC-USD/",
                "tradingview": "https://www.tradingview.com/symbols/BTCUSD/technicals/"
            },
            "eur/usd": {
                "yahoo": "https://finance.yahoo.com/quote/EURUSD=X/",
                "tradingview": "https://www.tradingview.com/symbols/EURUSD/technicals/"
            }
        }
        
        # Get URLs for the requested asset
        asset_lower = asset.lower()
        yahoo_url = asset_urls.get(asset_lower, {}).get("yahoo", "https://finance.yahoo.com/")
        tradingview_url = asset_urls.get(asset_lower, {}).get("tradingview", "https://www.tradingview.com/")
        
        # Prepare system message for ChatGPT
        system_message = f"""
        You are an expert financial analyst and trader with deep knowledge of markets, technical analysis, and trading strategies.
        Your task is to analyze {asset} and provide a comprehensive trading strategy for a {term} with {risk_level} risk level.
        
        IMPORTANT: You should actively search for current information about {asset}, including:
        
        1. CURRENT MARKET DATA:
           - Look up the latest price data on Yahoo Finance: {yahoo_url}
           - Find current price, volume, and basic indicators
        
        2. TECHNICAL ANALYSIS:
           - Check TradingView: {tradingview_url}
           - Research current RSI, MACD, Moving Averages, and support/resistance levels
        
        3. RECENT NEWS:
           - Search for latest news about the asset on financial news sites
           - Look for news from the past 7 days that could impact the asset
        
        4. MACROECONOMIC DATA:
           - Check for recent economic data releases
           - Research Fed announcements, inflation data, employment reports, etc.
        
        5. MARKET SENTIMENT:
           - Look for recent analyst opinions and market sentiment
           - Check Twitter/X for market commentary from respected analysts
        
        After gathering this information, analyze it and provide a comprehensive trading strategy.
        Format your response as a JSON object with the following structure:
        
        {{
            "market_summary": "Comprehensive summary of current market conditions based on your research",
            "key_drivers": ["List of key market drivers and factors from your research"],
            "technical_analysis": "Detailed technical analysis with key indicators you found",
            "risk_assessment": "Assessment of market risks based on current data",
            "trading_strategy": {{
                "direction": "LONG or SHORT",
                "rationale": "Explanation of the strategy direction based on your research",
                "entry": {{
                    "price": "Recommended entry price or range",
                    "rationale": "Rationale for entry point"
                }},
                "stop_loss": {{
                    "price": "Recommended stop loss price",
                    "rationale": "Rationale for stop loss placement"
                }},
                "take_profit_1": {{
                    "price": "First take profit target",
                    "rationale": "Rationale for TP1"
                }},
                "take_profit_2": {{
                    "price": "Second take profit target",
                    "rationale": "Rationale for TP2"
                }}
            }}
        }}
        
        Your response MUST be a valid JSON object with this exact structure.
        
        IMPORTANT: 
        - Make sure your price targets are realistic and close to the current market price. For Day trades, entry points should be extremely close (within 0.5-1%) to the current price for immediate execution. For swing trades, entry points should be within 1-3% of the current price at key technical levels. For position trades, entry points can be within 3-7% of the current price with focus on major support/resistance zones. Always prioritize high-probability technical setups over arbitrary percentage ranges.
        - The stop loss placement and risk-reward ratio should be determined by the risk level: For Conservative risk, use tighter stop losses with a minimum risk-reward ratio of 1:1.2. For Moderate risk, allow for wider stops with a minimum risk-reward ratio of 1:1.8. For Aggressive risk, use the widest acceptable stops with a minimum risk-reward ratio of 1:2.8. Always align stop losses with key technical levels (place them below for long positions and above for short positions) while maintaining these minimum ratios for the selected risk level.
        - Include specific data points and findings from your research in your analysis.
        - Cite specific news events, economic data, or technical indicators that inform your strategy.
        - If the current price is provided in the user message, use it as a reference but still verify it with your own research.
        
        - CRITICAL REQUIREMENT: Entry points MUST be within the specified percentage range of the current price:
          - Day trades: Entry MUST be within 0.5-1% of current price
          - Swing trades: Entry MUST be within 1-3% of current price
          - Position trades: Entry MUST be within 3-7% of current price
        Find a reasonable entry based on technical levels within this range.
        If you cannot find a suitable technical level within these ranges, you must still provide an entry within the specified percentage range and explain why this entry point is reasonable.

        DO NOT suggest entries outside these ranges under any circumstances.
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

        # Define trading term descriptions
        term_descriptions = {
            "day trade": "very short-term (1-2 days) with entries within 0.5-1% of current price",
            "swing trade": "short to medium-term (1-2 weeks) with entries within 1-3% of current price",
            "position trade": "medium to long-term (1-3 months) with entries within 3-7% of current price"
        }
        
        # Define risk level descriptions
        risk_level_descriptions = {
            "conservative": "conservative (prioritizing capital preservation with modest returns) with a risk to reward ratio of 1:1.2",
            "moderate": "moderate (balanced approach between risk and reward) with a risk to reward ratio of 1:1.8",
            "aggressive": "aggressive (higher risk tolerance for potentially higher returns) with a risk to reward ratio of 1:2.8"
        }
        
        # Get descriptions based on selected options
        term_desc = term_descriptions.get(term.lower(), term)
        risk_desc = risk_level_descriptions.get(risk_level.lower(), risk_level)

        user_message = f"""
        Please provide an advanced market analysis and trading strategy for:
        
        Asset: {asset}
        Trading Term: {term} ({term_desc})
        Risk Level: {risk_level} ({risk_desc})
        
        {current_price_info}
        
        I need you to search for and provide a comprehensive analysis that includes:
        
        1. CURRENT MARKET CONDITIONS:
           - Look up the latest price data and verify the current price for {asset}
           - Compare current price to recent trends
        
        2. TECHNICAL ANALYSIS:
           - Research current technical indicators specific to {asset}
           - Identify key support/resistance levels
           - Analyze momentum indicators (RSI, MACD, etc.)
        
        3. RECENT NEWS AND EVENTS:
           - Find and analyze news from the past week that impacts {asset}
           - Look for earnings reports, sector news, or market-moving events
        
        4. MACROECONOMIC FACTORS:
           - Check recent economic data releases that might affect {asset}
           - Consider how current Fed policy affects {asset}
           - Look for upcoming economic events that might impact the market
        
        5. MARKET SENTIMENT:
           - Research current analyst opinions about {asset}
           - Look for institutional positioning data if available
        
        Based on this research, develop a detailed {term_desc} trading strategy with {risk_desc} risk profile.
        Include specific entry, stop-loss, and take-profit levels appropriate for {asset}.
        
        IMPORTANT: Make sure your price targets are realistic and close to the current market price. For {term} strategies, entry points should typically be within 5% of the current price.
        
        Please format your response as a JSON object as specified in your instructions.
        """

        try:
            # Call OpenAI API with web browsing capability
            logger.info("Sending request to OpenAI API...")
            response = openai_client.chat.completions.create(
                model="gpt-4o",  # Use GPT-4o for advanced analysis capabilities
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.2,
                max_tokens=1500,
                timeout=45  # Increased timeout for web browsing
            )

            logger.info("Received response from ChatGPT")
            
            # Parse the JSON response
            try:
                analysis_json = json.loads(response.choices[0].message.content)
                logger.info("Successfully parsed JSON response")
                
                # Add current market price to the response
                if market_data and market_data.get('current_price'):
                    analysis_json['current_market_price'] = market_data['current_price']
                
                # Add metadata about the analysis
                analysis_json['meta'] = {
                    'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'model': 'gpt-4o',
                    'asset': asset,
                    'term': term,
                    'risk_level': risk_level,
                    'note': 'This analysis includes information the model has searched for about current market conditions, technical indicators, recent news, and market sentiment.'
                }
                
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