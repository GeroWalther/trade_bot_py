import pandas as pd
from typing import Dict, Any, Optional, List
import logging
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
from config import OANDA_CREDS, ALPHA_VANTAGE_API_KEY
from datetime import datetime, timedelta
import yfinance as yf
from textblob import TextBlob  # For basic sentiment analysis
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
import asyncio
import os
import finnhub
import requests
from bs4 import BeautifulSoup
import aiohttp

logger = logging.getLogger(__name__)

class MarketAnalyzer:
    def __init__(self, alpha_vantage_key: str = None):
        """Initialize the market analyzer"""
        self.alpha_vantage_key = alpha_vantage_key
        self.news_api_key = os.getenv('NEWS_API_KEY')  # Add back NewsAPI key
        
        if alpha_vantage_key:
            self._init_alpha_vantage(alpha_vantage_key)
            
        self._init_news_service()
        
        self.data_sources = ['oanda', 'alpha_vantage']
        self.cache_duration = 300  # 5 minutes cache
        self.cached_data = {}
        
        # Timeframe configuration for data fetching
        self.timeframe_config = {
            'INTRADAY': {
                'description': '1-8 hours',
                'granularity': 'M30',
                'candles': 16,
            },
            'SWING': {
                'description': '2-5 days',
                'granularity': 'H4',
                'candles': 30,
            },
            'MEDIUM_TERM': {
                'description': '1-4 weeks',
                'granularity': 'D',
                'candles': 28,
            },
            'LONG_TERM': {
                'description': '1-6 months',
                'granularity': 'W',
                'candles': 26,
            }
        }
        
        # Initialize services
        self._init_oanda()
        self._init_ai_service()

    def _init_oanda(self):
        """Initialize OANDA connection"""
        try:
            self.client = oandapyV20.API(
                access_token=OANDA_CREDS['ACCESS_TOKEN'],
                environment=OANDA_CREDS.get('ENVIRONMENT', 'practice')
            )
            logger.info("OANDA client initialized successfully")
            
            # Fix the instruments initialization
            try:
                from oandapyV20.endpoints.instruments import InstrumentsCandles
                # We'll get instruments list from actual requests instead
                self.available_instruments = ['EUR_USD', 'GBP_USD', 'USD_JPY']  # Default list
                logger.info(f"Using default instruments list: {self.available_instruments}")
                
                # Test connection with account info
                from oandapyV20.endpoints.accounts import AccountSummary
                r = AccountSummary(OANDA_CREDS['ACCOUNT_ID'])
                self.client.request(r)
                logger.info("OANDA connection test successful")
                
            except Exception as e:
                logger.error(f"Failed to fetch OANDA instruments: {e}")
                self.available_instruments = []
                
        except Exception as e:
            logger.error(f"Failed to initialize OANDA: {e}")
            raise

    async def _fetch_market_data(self, symbol: str, timeframe: str = 'SWING') -> pd.DataFrame:
        """Fetch market data with fallback"""
        logger.info(f"Starting market data fetch for {symbol} with timeframe {timeframe}")
        try:
            # Map timeframe to appropriate granularity
            granularity_map = {
                'INTRADAY': 'M15',    # 15-minute candles for intraday
                'SWING': 'H1',        # 1-hour candles for swing
                'MEDIUM_TERM': 'H4',  # 4-hour candles for medium term
                'LONG_TERM': 'D'      # Daily candles for long term
            }
            
            # Get appropriate granularity and count
            granularity = granularity_map.get(timeframe.upper(), 'H1')
            count_map = {
                'M15': 1000,  # More candles for shorter timeframes
                'H1': 500,
                'H4': 250,
                'D': 200
            }
            count = count_map.get(granularity, 500)
            
            logger.info(f"Using granularity: {granularity} with {count} candles for {timeframe} analysis")

            # Try OANDA first
            params = {
                "granularity": granularity,
                "count": count,
                "price": "M"
            }
            
            request = instruments.InstrumentsCandles(
                instrument=symbol,
                params=params
            )
            
            logger.info("Making OANDA API request...")
            response = self.client.request(request)
            logger.info("OANDA request completed")
            
            if response and 'candles' in response:
                candles = response['candles']
                logger.info(f"Received {len(candles)} candles from OANDA")
                
                # Convert to DataFrame synchronously
                logger.info("Converting OANDA data to DataFrame...")
                data = []
                for candle in candles:
                    if candle['complete']:
                        data.append({
                            'timestamp': pd.to_datetime(candle['time']),
                            'open': float(candle['mid']['o']),
                            'high': float(candle['mid']['h']),
                            'low': float(candle['mid']['l']),
                            'close': float(candle['mid']['c']),
                            'volume': float(candle['volume'])
                        })
                
                df = pd.DataFrame(data)
                if not df.empty:
                    logger.info(f"Successfully created DataFrame with {len(df)} rows")
                    return df
                else:
                    logger.warning("OANDA data conversion resulted in empty DataFrame")

            else:
                logger.warning("OANDA response missing candles data")

            # If OANDA fails or returns empty data, try Alpha Vantage
            logger.info("Attempting Alpha Vantage fallback...")
            try:
                av_symbol = symbol.replace('_', '')
                logger.info(f"Requesting Alpha Vantage data for {av_symbol}")
                data, _ = self.alpha_vantage.get_intraday(
                    symbol=av_symbol,
                    interval='60min',
                    outputsize='full'
                )
                df = pd.DataFrame(data).astype(float)
                if not df.empty:
                    logger.info(f"Successfully fetched Alpha Vantage data with {len(df)} rows")
                    return df
                else:
                    logger.warning("Alpha Vantage returned empty DataFrame")
            except Exception as e:
                logger.error(f"Alpha Vantage fallback failed: {str(e)}")

            logger.error("No data available from any source")
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error in market data fetch: {str(e)}", exc_info=True)
            return pd.DataFrame()

    def _format_symbol(self, symbol: str) -> str:
        """Format symbol to OANDA format (EURUSD -> EUR_USD)"""
        if '_' not in symbol and len(symbol) == 6:
            return f"{symbol[:3]}_{symbol[3:]}"
        return symbol

    async def analyze_market(self, symbol: str, timeframe: str = 'SWING', risk_level: str = 'MEDIUM') -> Dict:
        """Get market data and let AI analyze it"""
        logger.info(f"Starting market analysis for {symbol} with {timeframe} timeframe and {risk_level} risk")
        try:
            # Format symbol for OANDA (EURUSD -> EUR_USD)
            oanda_symbol = self._format_symbol(symbol)
            
            # Collect all relevant data
            market_data = {
                'price_data': await self._fetch_market_data(oanda_symbol, timeframe),
                'technical_data': await self._get_alpha_vantage_indicators(symbol),  # Keep original format for AV
                'news_data': await self._fetch_news_data(symbol),
                'economic_data': await self._fetch_economic_data(symbol)
            }

            # Initialize AI service if needed
            if not self.ai_service:
                from services.ai_analysis_service import AIAnalysisService
                self.ai_service = AIAnalysisService()
                logger.info("AI service initialized on demand")

            # Let AI analyze everything
            analysis_input = {
                'symbol': symbol,
                'timeframe': timeframe,
                'risk_level': risk_level,
                'market_data': market_data,
                'request_type': 'full_analysis'
            }

            # Get comprehensive AI analysis
            analysis = await self.ai_service.generate_analysis(analysis_input)
            
            if not analysis:
                return self._get_default_analysis(symbol, timeframe)
                
            return analysis

        except Exception as e:
            logger.error(f"Error in market analysis: {e}", exc_info=True)
            return self._get_default_analysis(symbol, timeframe)

    async def _get_alpha_vantage_indicators(self, symbol: str) -> Dict:
        """Get technical indicators from Alpha Vantage"""
        try:
            formatted_symbol = symbol.replace('_', '')
            
            # Get multiple indicators in parallel
            indicators = await asyncio.gather(
                asyncio.to_thread(self.ti.get_rsi, symbol=formatted_symbol),
                asyncio.to_thread(self.ti.get_macd, symbol=formatted_symbol),
                asyncio.to_thread(self.ti.get_bbands, symbol=formatted_symbol),
                asyncio.to_thread(self.ti.get_ema, symbol=formatted_symbol),
                return_exceptions=True
            )
            
            return {
                'rsi': indicators[0][0] if not isinstance(indicators[0], Exception) else None,
                'macd': indicators[1][0] if not isinstance(indicators[1], Exception) else None,
                'bbands': indicators[2][0] if not isinstance(indicators[2], Exception) else None,
                'ema': indicators[3][0] if not isinstance(indicators[3], Exception) else None,
            }
            
        except Exception as e:
            logger.error(f"Error getting Alpha Vantage indicators: {e}")
            return {}

    def _get_default_analysis(self, symbol: str, timeframe: str) -> Dict:
        """Return default analysis structure when errors occur"""
        return {
            'symbol': symbol,
            'timeframe': {
                'value': timeframe,
                'description': self.timeframe_config.get(timeframe, {}).get('description', 'unknown')
            },
            'price': {
                'current': 0.0,
                'change': 0.0
            },
            'technical_analysis': {
                'rsi': {'value': 50, 'status': 'NEUTRAL'},
                'ema50': {'value': 0, 'trend': 'NEUTRAL'}
            },
            'sentiment': {
                'overall': 'NEUTRAL',
                'score': 50,
                'confidence': 0
            },
            'risk_level': 'MEDIUM',
            'timestamp': datetime.now().isoformat(),
            'error': True
        }

    async def _fetch_news_data(self, symbol: str) -> List[Dict]:
        """Fetch news data from both Finnhub and NewsAPI"""
        try:
            all_articles = []
            seen_headlines = set()

            # Try Finnhub first
            try:
                finnhub_articles = await self._fetch_finnhub_news(symbol)
                logger.info(f"Got {len(finnhub_articles)} articles from Finnhub for {symbol}")
                for article in finnhub_articles:
                    if article['title'] not in seen_headlines:
                        seen_headlines.add(article['title'])
                        # Ensure published_at is a timestamp
                        published_at = float(article['published_at'])
                        article['formatted_date'] = datetime.fromtimestamp(published_at).strftime('%Y-%m-%d %H:%M')
                        article['published_at'] = published_at
                        all_articles.append(article)
            except Exception as e:
                logger.error(f"Finnhub fetch error: {e}")

            # Then try NewsAPI
            try:
                if hasattr(self, 'news_api'):
                    newsapi_articles = await self._fetch_newsapi_news(symbol)
                    logger.info(f"Got {len(newsapi_articles)} articles from NewsAPI for {symbol}")
                    for article in newsapi_articles:
                        if article['title'] not in seen_headlines:
                            seen_headlines.add(article['title'])
                            # NewsAPI articles already have timestamp as published_at
                            article['formatted_date'] = datetime.fromtimestamp(article['published_at']).strftime('%Y-%m-%d %H:%M')
                            all_articles.append(article)
            except Exception as e:
                logger.error(f"NewsAPI fetch error: {e}")

            if not all_articles:
                logger.warning(f"No news found from any source for {symbol}")
                return []

            # Sort by date (newest first)
            all_articles.sort(key=lambda x: float(x['published_at']), reverse=True)
            logger.info(f"Found total of {len(all_articles)} combined articles")

            return all_articles[:30]  # Return 30 newest articles

        except Exception as e:
            logger.error(f"Error fetching news: {e}", exc_info=True)
            return []

    def _get_broad_search_term(self, symbol: str) -> str:
        """Get broader search terms for fallback"""
        broad_terms = {
            'XAU_USD': 'gold OR precious metals OR bullion',
            'XAG_USD': 'silver OR precious metals OR bullion',
            'BCO_USD': 'oil OR crude OR energy market',
            'SPX500_USD': 'stock market OR S&P OR wall street',
            'NAS100_USD': 'nasdaq OR tech stocks OR technology sector',
            'JP225_USD': 'nikkei OR japanese market OR asian stocks',
            'DE30_EUR': 'dax OR german market OR european stocks'
        }
        
        if symbol in broad_terms:
            return broad_terms[symbol]
        elif '_' in symbol:  # Forex pairs
            base, quote = symbol.split('_')
            return f"forex OR currency OR {base} OR {quote}"
        return symbol

    async def _fetch_finnhub_news(self, symbol: str) -> List[Dict]:
        """Fetch news from Finnhub"""
        try:
            # Map symbols to categories and search terms
            categories = {
                # Commodities
                'XAU': 'general',  # Changed from forex to general for better coverage
                'XAG': 'general',
                'BCO': 'general',
                # Indices
                'SPX500': 'general',
                'NAS100': 'general',
                'JP225': 'general',
                'DE30': 'general'
            }
            
            # Get the appropriate category
            if '_' in symbol:
                if any(symbol.startswith(prefix) for prefix in categories.keys()):
                    category = categories[symbol.split('_')[0]]
                else:
                    category = 'forex'  # Default for forex pairs
            else:
                category = 'general'

            logger.info(f"Fetching news for {symbol} with category: {category}")
            
            try:
                # Get news from both general and forex categories for better coverage
                general_news = await asyncio.to_thread(
                    self.finnhub_client.general_news,
                    'general'
                )
                
                forex_news = await asyncio.to_thread(
                    self.finnhub_client.general_news,
                    'forex'
                )
                
                # Combine news from both sources
                news = general_news + forex_news if forex_news else general_news
                
                if not news:
                    logger.warning(f"No news found for {symbol}")
                    return []
                    
                logger.info(f"Got {len(news)} raw news items from Finnhub")
                
                # Enhanced search terms for better matching
                search_terms = {
                    # Forex pairs
                    'EUR': ['euro', 'eur', 'european currency', 'eurozone'],
                    'USD': ['dollar', 'usd', 'us currency', 'greenback', 'federal reserve', 'fed'],
                    'GBP': ['pound', 'sterling', 'gbp', 'british currency', 'bank of england', 'boe'],
                    'JPY': ['yen', 'jpy', 'japanese currency', 'bank of japan', 'boj'],
                    'CHF': ['franc', 'chf', 'swiss', 'snb', 'swiss national bank'],
                    'AUD': ['aussie', 'aud', 'australian dollar', 'rba', 'reserve bank of australia'],
                    'CAD': ['loonie', 'cad', 'canadian dollar', 'bank of canada', 'boc'],
                    'NZD': ['kiwi', 'nzd', 'new zealand dollar', 'rbnz'],
                    # Commodities
                    'XAU': ['gold', 'xau', 'bullion', 'precious metal', 'gold price', 'gold market'],
                    'XAG': ['silver', 'xag', 'precious metal', 'silver price', 'silver market'],
                    'BCO': ['brent', 'crude oil', 'oil price', 'petroleum', 'energy market'],
                    # Indices
                    'SPX500': ['s&p', 'sp500', 's&p 500', 'spx', 'us stocks', 'wall street'],
                    'NAS100': ['nasdaq', 'tech stocks', 'nasdaq 100', 'technology sector'],
                    'JP225': ['nikkei', 'japanese stocks', 'nikkei 225', 'japan market'],
                    'DE30': ['dax', 'german stocks', 'dax 40', 'german market', 'european stocks']
                }
                
                # Get relevant search terms
                base_curr = symbol[:3] if '_' in symbol else symbol
                quote_curr = symbol[4:] if '_' in symbol else None
                
                relevant_terms = search_terms.get(base_curr, [base_curr.lower()])
                if quote_curr:
                    relevant_terms.extend(search_terms.get(quote_curr, [quote_curr.lower()]))
                
                articles = []
                seen_headlines = set()
                
                for article in news:
                    if not article.get('headline') or not article.get('summary'):
                        continue
                        
                    text = (article['headline'] + ' ' + article['summary']).lower()
                    
                    # Check if article is relevant
                    if not any(term.lower() in text for term in relevant_terms):
                        continue
                        
                    if article['headline'] in seen_headlines:
                        continue
                        
                    seen_headlines.add(article['headline'])
                    
                    articles.append({
                        'title': article['headline'],
                        'description': article.get('summary', ''),
                        'sentiment': self._analyze_text_sentiment(
                            article['headline'] + ' ' + article.get('summary', '')
                        ),
                        'published_at': article['datetime'],
                        'url': article.get('url', ''),
                        'source': article.get('source', 'Finnhub'),
                        'category': category
                    })
                
                # Sort by date (newest first)
                articles.sort(key=lambda x: x['published_at'], reverse=True)
                
                logger.info(f"Found {len(articles)} relevant articles for {symbol}")
                
                return articles[:30]  # Limit to 30 most recent articles
                
            except Exception as api_error:
                logger.error(f"Finnhub API error: {api_error}")
                raise
                
        except Exception as e:
            logger.error(f"Error fetching Finnhub news: {e}")
            return []

    def _analyze_text_sentiment(self, text: str) -> float:
        """Analyze text sentiment using TextBlob"""
        try:
            analysis = TextBlob(text)
            return analysis.sentiment.polarity  # Returns value between -1 and 1
        except Exception:
            return 0.0
            
    async def _fetch_economic_data(self, symbol: str) -> Dict:
        """Fetch relevant economic indicators"""
        try:
            # For now, return basic structure
            return {
                'gdp_growth': None,
                'interest_rate': None,
                'inflation_rate': None,
                'unemployment': None
            }
        except Exception as e:
            logger.error(f"Error fetching economic data: {e}")
            return {}

    def _init_alpha_vantage(self, key: str):
        """Initialize Alpha Vantage clients"""
        self.alpha_vantage = TimeSeries(key=key, output_format='pandas')
        self.ti = TechIndicators(key=key, output_format='pandas')
        
    def _init_news_service(self):
        """Initialize news API clients"""
        try:
            # Initialize both Finnhub and NewsAPI clients
            finnhub_key = os.getenv('FINNHUB_API_KEY')
            if not finnhub_key:
                raise ValueError("FINNHUB_API_KEY not found in environment")
            self.finnhub_client = finnhub.Client(api_key=finnhub_key)
            
            if self.news_api_key:
                from newsapi import NewsApiClient
                self.news_api = NewsApiClient(api_key=self.news_api_key)
                logger.info("NewsAPI initialized successfully")
            
            # Test the connections
            test_news_finnhub = self.finnhub_client.general_news('general')  # Changed from 'forex' to 'general'
            if test_news_finnhub is not None:
                logger.info("Finnhub connection test successful")
            
            if hasattr(self, 'news_api'):
                test_news_api = self.news_api.get_everything(
                    q='market',  # More general search term for testing
                    language='en',
                    page_size=1
                )
                if test_news_api:
                    logger.info("NewsAPI connection test successful")
            
        except Exception as e:
            logger.error(f"News service initialization error: {e}")
            raise ValueError(f"News service initialization failed: {e}")

    def _init_ai_service(self):
        """Initialize AI analysis service"""
        try:
            from services.ai_analysis_service import AIAnalysisService
            self.ai_service = AIAnalysisService()
            logger.info("AI service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AI service: {e}")
            self.ai_service = None

    def _generate_analysis_explanation(self, technical, sentiment, signals, timeframe) -> str:
        """Generate human-readable analysis explanation"""
        try:
            trend = technical['ema50']['trend']
            rsi_status = technical['rsi']['status']
            timeframe_desc = self.timeframe_config[timeframe]['description']
            
            explanation = f"Analysis for {timeframe_desc} timeframe shows a {trend.lower()} trend "
            explanation += f"with RSI indicating {rsi_status.lower()} conditions. "
            
            if sentiment['overall'] != 'NEUTRAL':
                explanation += f"Market sentiment is {sentiment['overall'].lower()} "
                explanation += f"with {sentiment['confidence']}% confidence. "
            
            if signals:
                explanation += "Key signals include: " + ", ".join(signals).lower() + "."
                
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return "Unable to generate analysis explanation."

    def _get_search_term_for_symbol(self, symbol: str) -> str:
        """Get appropriate search term for the symbol"""
        search_terms = {
            # Indices
            'SPX500_USD': '(S&P 500) OR SP500 OR (US stock market)',
            'NAS100_USD': 'Nasdaq OR (Nasdaq 100) OR (US tech stocks)',
            'JP225_USD': 'Nikkei OR (Nikkei 225) OR (Japanese stocks)',
            'DE30_EUR': 'DAX OR (DAX 40) OR (German stocks)',
            # Commodities with broader terms
            'XAU_USD': '(gold price) OR (gold market) OR (precious metals) OR (gold trading)',
            'XAG_USD': '(silver price) OR (silver market) OR (precious metals) OR (silver trading) OR (commodity silver)',
            'BCO_USD': '(brent crude) OR (oil price) OR (crude oil) OR (oil market)'
        }
        
        # If it's a commodity, add more general market terms
        if symbol in ['XAU_USD', 'XAG_USD']:
            base_term = search_terms.get(symbol, '')
            return f"{base_term} OR (commodity market) OR (metals trading)"
        
        return search_terms.get(symbol, symbol.replace('_', '/'))

    async def _fetch_newsapi_news(self, symbol: str) -> List[Dict]:
        """Fetch news from NewsAPI"""
        try:
            search_term = self._get_search_term_for_symbol(symbol)
            from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
            # Add more financial news sources
            domains = (
                'reuters.com,bloomberg.com,cnbc.com,finance.yahoo.com,marketwatch.com,'
                'kitco.com,mining.com,bullionvault.com,investing.com,ft.com'
            )
            
            # Try multiple searches for better coverage
            all_articles = []
            
            # First try with specific search
            news = await asyncio.to_thread(
                self.news_api.get_everything,
                q=search_term,
                language='en',
                sort_by='publishedAt',
                from_param=from_date,
                page_size=30,
                domains=domains
            )
            
            all_articles.extend(news.get('articles', []))
            
            # For commodities, try an additional broader search
            if symbol in ['XAU_USD', 'XAG_USD']:
                broad_term = 'precious metals market' if symbol == 'XAG_USD' else 'gold market'
                more_news = await asyncio.to_thread(
                    self.news_api.get_everything,
                    q=broad_term,
                    language='en',
                    sort_by='publishedAt',
                    from_param=from_date,
                    page_size=15,
                    domains=domains
                )
                all_articles.extend(more_news.get('articles', []))
            
            # Process and format articles
            processed_articles = []
            seen_headlines = set()
            
            for article in all_articles:
                if not article['title'] or not article['description']:
                    continue
                    
                if article['title'] in seen_headlines:
                    continue
                    
                seen_headlines.add(article['title'])
                
                processed_articles.append({
                    'title': article['title'],
                    'description': article['description'],
                    'sentiment': self._analyze_text_sentiment(
                        article['title'] + ' ' + article['description']
                    ),
                    'published_at': article['publishedAt'],
                    'url': article['url'],
                    'source': article['source']['name'],
                    'category': 'general'
                })
            
            # Sort by date and return
            processed_articles.sort(key=lambda x: x['published_at'], reverse=True)
            return processed_articles[:30]
            
        except Exception as e:
            logger.error(f"NewsAPI fetch error: {e}")
            return [] 