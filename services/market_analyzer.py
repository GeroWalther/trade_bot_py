import numpy as np
import pandas as pd
import talib
from typing import Dict, Any, Optional, List
import logging
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
from config import OANDA_CREDS, NEWS_API_KEY, ALPHA_VANTAGE_API_KEY
from datetime import datetime, timedelta
import yfinance as yf
from newsapi import NewsApiClient
from textblob import TextBlob  # For basic sentiment analysis
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
import asyncio

logger = logging.getLogger(__name__)

class MarketAnalyzer:
    def __init__(self, alpha_vantage_key: str):
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
        self._init_alpha_vantage(alpha_vantage_key)
        self._init_news_service()
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
        """Fetch news data for the symbol"""
        try:
            # Convert forex pair to searchable terms
            search_term = symbol.replace('_', '/') if '_' in symbol else symbol
            
            news = await asyncio.to_thread(
                self.news_api.get_everything,
                q=search_term,
                language='en',
                sort_by='relevancy',
                page_size=10
            )
            
            return [{
                'title': article['title'],
                'description': article['description'],
                'sentiment': self._analyze_text_sentiment(article['title'] + ' ' + article['description']),
                'published_at': article['publishedAt']
            } for article in news['articles']]

        except Exception as e:
            logger.error(f"Error fetching news data: {e}")
            return []
            
    def _analyze_text_sentiment(self, text: str) -> float:
        """Analyze text sentiment using TextBlob"""
        try:
            analysis = TextBlob(text)
            return analysis.sentiment.polarity
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
        """Initialize news API client"""
        self.news_api = NewsApiClient(api_key=NEWS_API_KEY)

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