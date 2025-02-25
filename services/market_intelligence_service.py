import logging
import asyncio
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles
import pandas as pd
from config import OANDA_CREDS
from newsapi import NewsApiClient
from textblob import TextBlob
import requests
import os
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.timeseries import TimeSeries
from fredapi import Fred
from services.ai_analysis_service import AIAnalysisService
import aiohttp
from functools import wraps
import time

logger = logging.getLogger(__name__)

def rate_limit_decorator(max_requests: int, window: int):
    """Rate limiting decorator for API calls"""
    calls = []
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            now = time.time()
            calls_in_window = [call for call in calls if call > now - window]
            if len(calls_in_window) >= max_requests:
                wait_time = calls[0] - (now - window)
                logger.warning(f"Rate limit hit, waiting {wait_time:.2f}s")
                time.sleep(wait_time)
                calls.clear()
            calls.append(now)
            return await func(*args, **kwargs)
        return wrapper
    return decorator

class MarketIntelligenceService:
    def __init__(self):
        self.base_url = 'https://api.stlouisfed.org/fred/series'
        self.fred_api_key = os.getenv('FRED_API_KEY')
        
        if not self.fred_api_key:
            logger.error("FRED API key not found")
            raise ValueError("FRED API key is required")
        
        # Core indicators (7 that we know work)
        self.series_ids = {
            'cpi': 'CPIAUCSL',            # CPI
            'core_cpi': 'CPILFESL',       # Core CPI
            'fed_rate': 'FEDFUNDS',        # Fed Funds Rate
            'unemployment': 'UNRATE',      # Unemployment Rate
            'nfp': 'PAYEMS',              # Non-Farm Payrolls
            'consumer_conf': 'UMCSENT',    # Consumer Confidence
            'repo_liquidity': 'RRPONTSYD', # Reverse Repo Operations
        }

        # Units map for each series
        self.units_map = {
            'CPIAUCSL': 'pc1',      # Percent change
            'CPILFESL': 'pc1',      # Percent change
            'FEDFUNDS': 'lin',      # Linear (actual rate)
            'UNRATE': 'lin',        # Linear (actual rate)
            'PAYEMS': 'pc1',        # Percent change
            'UMCSENT': 'lin',       # Linear (actual index)
            'RRPONTSYD': 'pc1',     # Percent change
        }

        # Initialize cache
        self.cache = {
            'data': None,
            'timestamp': None,
            'expiry': 3600  # Cache expires after 1 hour
        }

        # Asset-specific indicators
        self.asset_specific_indicators = {
            'forex': {
                'euro_sentiment': 'EUSSI',      # Euro Area Sentiment
                'uk_rate': 'BOERUKM',          # UK Interest Rate
                'swiss_rate': 'SNBCHM',        # Swiss Interest Rate
                'us_rate': 'DFF'               # US Federal Funds Rate
            },
            'commodities': {
                'oil_supply': 'WCESTUS1',      # US Oil Supply
                'gold_demand': 'GOLDAMGBD228NLBM'  # Gold Fixing Price
            }
        }

        logger.info("Market Intelligence Service initialized successfully")

    @rate_limit_decorator(max_requests=5, window=1)
    async def _fetch_fred_data(self, series_id: str, start_date: datetime, end_date: datetime) -> Optional[Dict]:
        """Fetch data from FRED API with rate limiting"""
        try:
            url = f"{self.base_url}/observations"
            
            params = {
                'series_id': series_id,
                'api_key': self.fred_api_key,
                'file_type': 'json',
                'sort_order': 'desc',
                'observation_start': start_date.strftime('%Y-%m-%d'),
                'observation_end': end_date.strftime('%Y-%m-%d'),
                'units': self.units_map.get(series_id, 'lin'),
                'frequency': 'm',  # Monthly frequency
                'aggregation_method': 'avg'  # Average for the period
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 429:  # Too Many Requests
                        logger.warning(f"Rate limit hit for {series_id}, will retry later")
                        return None
                    response.raise_for_status()
                    return await response.json()

        except Exception as e:
            logger.error(f"Error fetching FRED data for {series_id}: {e}")
            return None

    async def get_economic_indicators(self, asset_type=None, core_only=False):
        """Get economic indicators with caching"""
        try:
            # Check cache first
            if self.cache['data'] and self.cache['timestamp']:
                if (datetime.now() - self.cache['timestamp']).seconds < self.cache['expiry']:
                    if core_only:
                        return {k: v for k, v in self.cache['data'].items() if k in self.series_ids}
                    if asset_type:
                        return {k: v for k, v in self.cache['data'].items() 
                                if k in self.asset_specific_indicators.get(asset_type, [])}
                    return self.cache['data']

            # If not in cache, fetch fresh data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            indicators = {}
            
            # Always fetch core indicators
            if core_only or not asset_type:
                for series_id, fred_id in self.series_ids.items():
                    data = await self._fetch_fred_data(fred_id, start_date, end_date)
                    if data:
                        indicators[series_id] = self._process_indicator_data(data, series_id)

            # Fetch asset-specific indicators if requested
            if asset_type and not core_only:
                asset_indicators = self.asset_specific_indicators.get(asset_type, {})
                for indicator_id, fred_id in asset_indicators.items():
                    data = await self._fetch_fred_data(fred_id, start_date, end_date)
                    if data:
                        indicators[indicator_id] = self._process_indicator_data(data, indicator_id)

            # Update cache
            self.cache['data'] = indicators
            self.cache['timestamp'] = datetime.now()
            
            return indicators

        except Exception as e:
            logger.error(f"Error fetching economic indicators: {e}")
            raise

    async def get_comprehensive_analysis(self, asset: str, timeframe: str) -> Dict:
        """Get complete market analysis including probabilities"""
        try:
            # Gather all data concurrently
            market_data, economic_data, news_data, technical_data = await asyncio.gather(
                self._get_market_data(asset),
                self._get_economic_indicators(asset),
                self._get_market_news(asset),
                self._get_technical_analysis(asset)
            )
            
            # Calculate probabilities using Monte Carlo
            probability_analysis = self._calculate_probabilities(
                technical_data, 
                timeframe
            )
            
            # Get AI analysis
            ai_analysis = await self.ai_analysis.generate_market_analysis(
                asset=asset,
                market_data=market_data,
                economic_data=economic_data,
                news_data=news_data,
                technical_data=technical_data,
                probability_analysis=probability_analysis
            )
            
            return {
                'market_data': market_data,
                'economic_data': economic_data,
                'news': news_data,
                'technical_analysis': technical_data,
                'probability_analysis': probability_analysis,
                'ai_analysis': ai_analysis
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}", exc_info=True)
            raise

    async def get_asset_analysis(self, asset: str, timeframe: str) -> Dict:
        """Get comprehensive asset analysis"""
        try:
            logger.info(f"Getting analysis for {asset} with timeframe {timeframe}")
            
            # Gather all data concurrently for efficiency
            market_data, economic_data, news_data = await asyncio.gather(
                self._get_market_data(asset),
                self._get_economic_indicators(asset),
                self._get_market_news(asset)
            )

            # Get relevant economic indicators for this asset
            relevant_indicators = self.asset_specific_indicators.get(asset, {}).get('additional', [])
            filtered_economic_data = {
                k: v for k, v in economic_data.items() 
                if k in relevant_indicators
            }

            return {
                'market_data': market_data,
                'economic_data': filtered_economic_data,
                'news': news_data,
                'sentiment': {
                    'score': self._calculate_sentiment(news_data) if news_data else 0.5,
                    'details': [n.get('sentiment', 0) for n in news_data] if news_data else []
                },
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in asset analysis: {e}", exc_info=True)
            raise

    async def _get_oanda_data(self, symbol: str) -> pd.DataFrame:
        """Get OANDA candle data"""
        try:
            logger.info(f"Fetching OANDA data for {symbol}")
            
            # Create request
            params = {
                "count": "100",
                "granularity": "H1",
                "price": "M"
            }
            
            request = InstrumentsCandles(
                instrument=symbol,
                params=params
            )
            
            try:
                # Make API request
                response = self.api.request(request)
                logger.info(f"OANDA response received for {symbol}")
                
                if 'candles' not in response:
                    logger.error(f"Invalid OANDA response: {response}")
                    return pd.DataFrame()
                    
                candles = response['candles']
                if not candles:
                    logger.warning(f"No candles returned for {symbol}")
                    return pd.DataFrame()
                
                # Create DataFrame
                data = []
                for candle in candles:
                    if candle['complete']:  # Only use complete candles
                        data.append({
                            'time': pd.to_datetime(candle['time']),
                            'open': float(candle['mid']['o']),
                            'high': float(candle['mid']['h']),
                            'low': float(candle['mid']['l']),
                            'close': float(candle['mid']['c']),
                            'volume': float(candle.get('volume', 0))
                        })
                
                df = pd.DataFrame(data)
                df.set_index('time', inplace=True)
                return df
                
            except Exception as e:
                logger.error(f"OANDA API request failed: {str(e)}")
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error in _get_oanda_data: {str(e)}")
            return pd.DataFrame()

    async def _analyze_forex(self) -> Dict:
        """Analyze forex pairs"""
        results = {}
        for symbol in self.symbols['forex']:
            try:
                df = await self._get_oanda_data(symbol)
                
                # Calculate basic indicators
                sma_20 = df['close'].rolling(window=20).mean().iloc[-1]
                sma_50 = df['close'].rolling(window=50).mean().iloc[-1]
                rsi = self._calculate_rsi(df['close'])
                
                current_price = df['close'].iloc[-1]
                prev_price = df['close'].iloc[-2]
                
                results[symbol] = {
                    'symbol': symbol,
                    'price': current_price,
                    'change_percent': ((current_price - prev_price) / prev_price) * 100,
                    'indicators': {
                        'sma_20': sma_20,
                        'sma_50': sma_50,
                        'rsi': rsi
                    },
                    'signals': {
                        'trend': {
                            'primary': 'BULLISH' if sma_20 > sma_50 else 'BEARISH',
                            'strength': 'STRONG' if abs(sma_20 - sma_50) / sma_50 > 0.001 else 'MODERATE'
                        }
                    },
                    'probability_up': 65 if sma_20 > sma_50 and rsi < 70 else 35,
                    'confidence': 80 if abs(sma_20 - sma_50) / sma_50 > 0.002 else 60
                }

            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                results[symbol] = self._get_error_symbol_response(symbol)
                
        return results

    async def _get_market_news(self, asset: str) -> Dict:
        """Get relevant market news"""
        try:
            articles = self.news_client.get_everything(
                q=f'{asset} OR "{asset} trading" OR "foreign exchange"',
                language='en',
                sort_by='relevancy',
                page_size=10
            )
            
            processed_articles = []
            for article in articles['articles']:
                blob = TextBlob(article['title'] + ' ' + (article['description'] or ''))
                processed_articles.append({
                    'title': article['title'],
                    'description': article['description'],
                    'url': article['url'],
                    'source': article['source']['name'],
                    'sentiment': blob.sentiment.polarity,
                    'published_at': article['publishedAt']
                })
            
            return processed_articles
            
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return []

    def _calculate_rsi(self, prices, periods=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs.iloc[-1]))

    def _get_error_response(self) -> Dict:
        """Generate error response"""
        return {
            'error': True,
            'message': 'Error fetching market analysis',
            'timestamp': datetime.now().isoformat()
        }

    def _get_error_symbol_response(self, symbol) -> Dict:
        """Generate error response for a specific symbol"""
        return {
            'error': True,
            'symbol': symbol,
            'message': f'Error analyzing {symbol}',
            'timestamp': datetime.now().isoformat()
        }

    def _get_next_release_date(self, indicator: str) -> str:
        """Get next release date from FRED releases API"""
        if indicator not in self.release_ids:
            return None

        url = 'https://api.stlouisfed.org/fred/release/dates'
        params = {
            'release_id': self.release_ids[indicator],
            'api_key': self.fred_api_key,
            'file_type': 'json',
            'sort_order': 'asc',  # Get earliest (next) dates first
            'limit': 1,  # Only get next release
            'include_release_dates_with_no_data': False
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data and 'release_dates' in data and data['release_dates']:
                return data['release_dates'][0]['date']
            
            return None
        except Exception as e:
            logger.error(f"Error fetching release date for {indicator}: {e}")
            return None

    def _calculate_sentiment(self, news_data):
        """Calculate sentiment score from news data"""
        sentiment_scores = [n['sentiment'] for n in news_data]
        average_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        return average_sentiment

    async def _get_market_data(self, asset: str) -> Dict:
        """Get market data including Alpha Vantage indicators"""
        try:
            logger.info(f"Getting market data for {asset}")
            
            # Get OANDA data
            oanda_symbol = asset.replace('USD', '_USD')
            df = await self._get_oanda_data(oanda_symbol)
            
            if df.empty:
                logger.warning(f"No OANDA data available for {asset}, using default data")
                return self._get_default_market_data()
            
            # Get indicators
            try:
                current_price = df['close'].iloc[-1]
                prev_price = df['close'].iloc[-2]
                volume = int(df['volume'].iloc[-1]) if 'volume' in df else 0
                
                sma20 = df['close'].rolling(window=20).mean().iloc[-1]
                sma50 = df['close'].rolling(window=50).mean().iloc[-1]
                rsi = self._calculate_rsi(df['close'])
                
                return {
                    'current_price': current_price,
                    'change_percent': ((current_price - prev_price) / prev_price) * 100,
                    'volume': volume,
                    'indicators': {
                        'rsi': rsi,
                        'sma': {
                            'sma20': sma20,
                            'sma50': sma50
                        }
                    },
                    'signals': {
                        'trend': {
                            'primary': 'BULLISH' if sma20 > sma50 else 'BEARISH',
                            'strength': 'STRONG' if abs(sma20 - sma50) / sma50 > 0.001 else 'MODERATE'
                        }
                    }
                }
                
            except Exception as e:
                logger.error(f"Error calculating indicators: {str(e)}")
                return self._get_default_market_data()
            
        except Exception as e:
            logger.error(f"Error in _get_market_data: {str(e)}")
            return self._get_default_market_data()

    async def _get_tradingview_strategies(self, asset: str) -> List[Dict]:
        """Get popular TradingView strategies for asset"""
        try:
            # You would implement TradingView API call here
            # For now, returning example strategies
            return [
                {
                    'name': 'BB + RSI Strategy',
                    'timeframe': '1h',
                    'signal': 'LONG' if self._should_go_long(asset) else 'SHORT',
                    'confidence': 75
                },
                {
                    'name': 'MACD Crossover',
                    'timeframe': '4h',
                    'signal': 'NEUTRAL',
                    'confidence': 65
                }
            ]
        except Exception as e:
            logger.error(f"Error fetching TradingView strategies: {e}")
            return [] 

    def _get_default_market_data(self) -> Dict:
        """Return default market data when real data cannot be fetched"""
        return {
            'current_price': 0.0,
            'change_percent': 0.0,
            'volume': 0,
            'indicators': {
                'rsi': 50,
                'macd': {
                    'macd': 0,
                    'signal': 0,
                    'hist': 0
                },
                'bbands': {
                    'upper': 0,
                    'middle': 0,
                    'lower': 0
                },
                'sma': {
                    'sma20': 0,
                    'sma50': 0
                }
            },
            'signals': {
                'trend': {
                    'primary': 'NEUTRAL',
                    'strength': 'WEAK'
                }
            }
        }

    def _process_indicator_data(self, data: Dict, indicator_id: str) -> Dict:
        """Process raw FRED data into formatted indicator data"""
        try:
            if not data or not data.get('observations'):
                return None

            # Core indicators metadata
            indicators_metadata = {
                'cpi': {
                    'name': 'CPI (YoY)',
                    'description': 'Consumer Price Index',
                    'importance': 96,
                    'correlation': 92
                },
                'core_cpi': {
                    'name': 'Core CPI (YoY)',
                    'description': 'CPI excluding food and energy',
                    'importance': 95,
                    'correlation': 90
                },
                'fed_rate': {
                    'name': 'Fed Funds Rate',
                    'description': 'Federal Funds Rate',
                    'importance': 96,
                    'correlation': 92
                },
                'unemployment': {
                    'name': 'Unemployment Rate',
                    'description': 'U.S. Unemployment Rate',
                    'importance': 94,
                    'correlation': 88
                },
                'nfp': {
                    'name': 'Non-Farm Payrolls',
                    'description': 'U.S. Jobs Data',
                    'importance': 96,
                    'correlation': 91
                },
                'consumer_conf': {
                    'name': 'Consumer Confidence',
                    'description': 'Consumer Confidence Index',
                    'importance': 93,
                    'correlation': 87
                },
                'repo_liquidity': {
                    'name': 'Repo Market Liquidity',
                    'description': 'Reverse Repo Operations',
                    'importance': 94,
                    'correlation': 88
                }
            }

            metadata = indicators_metadata.get(indicator_id, {
                'name': indicator_id.replace('_', ' ').title(),
                'description': 'Economic Indicator',
                'importance': 90,
                'correlation': 85
            })

            # Handle missing or invalid values
            latest_value = data['observations'][0]['value']
            latest_value = 0.0 if latest_value == '.' else float(latest_value)
            
            prev_value = data['observations'][1]['value'] if len(data['observations']) > 1 else latest_value
            prev_value = 0.0 if prev_value == '.' else float(prev_value)
            
            historical_data = [
                {
                    'value': 0.0 if obs['value'] == '.' else float(obs['value']),
                    'date': obs['date']
                }
                for obs in data['observations'][:3]
            ]

            return {
                'name': metadata['name'],
                'description': metadata['description'],
                'value': f"{latest_value:.2f}%",
                'trend': 'up' if latest_value > prev_value else 'down',
                'importance': metadata['importance'],
                'correlation': metadata['correlation'],
                'latest_release': data['observations'][0]['date'],
                'historical_data': historical_data
            }

        except Exception as e:
            logger.error(f"Error processing indicator {indicator_id}: {e}")
            return None 