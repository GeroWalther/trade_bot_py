import logging
import asyncio
from typing import Dict
from datetime import datetime, timedelta
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import pandas as pd
from config import OANDA_CREDS
from newsapi import NewsApiClient
from textblob import TextBlob
import requests
import os

logger = logging.getLogger(__name__)

class MarketIntelligenceService:
    def __init__(self):
        self.api = API(access_token=OANDA_CREDS["ACCESS_TOKEN"])
        self.news_client = NewsApiClient(api_key='52deeb5fce334ae2b127e2b08efa8335')
        self.fred_api_key = os.getenv('FRED_API_KEY')
        if not self.fred_api_key:
            logger.error("FRED API key not found")
            raise ValueError("FRED API key is required")
        
        self.symbols = {
            'forex': ['EUR_USD', 'GBP_USD', 'USD_JPY', 'AUD_USD'],
            'commodities': ['XAU_USD', 'BCO_USD'],
            'indices': ['SPX500_USD', 'NAS100_USD', 'DE30_EUR']
        }
        
        # Updated economic indicators with real FRED series IDs
        self.indicators = {
            'US CPI (YoY)': 'CPIAUCSL',          # Consumer Price Index
            'US PPI (YoY)': 'PPIACO',            # Producer Price Index
            'Fed Interest Rate': 'FEDFUNDS',      # Federal Funds Rate
            'US 10Y Treasury': 'DGS10',          # 10-Year Treasury Rate
            'ECB Interest Rate': 'ECBDFR',       # ECB Deposit Facility Rate
            'EU CPI (YoY)': 'CP0000EZ19M086NEST',# EU Harmonized CPI
            'US Unemployment': 'UNRATE',          # Unemployment Rate
            'US GDP Growth': 'GDP',              # Gross Domestic Product
            'Industrial Production': 'INDPRO',    # Industrial Production Index
            'US Core PCE': 'PCEPILFE'            # Core Personal Consumption Expenditures
        }

        # Impact levels based on market significance
        self.impact_levels = {
            'US CPI (YoY)': 'HIGH',
            'Fed Interest Rate': 'HIGH',
            'US 10Y Treasury': 'HIGH',
            'ECB Interest Rate': 'HIGH',
            'EU CPI (YoY)': 'HIGH',
            'US Core PCE': 'HIGH',
            'US PPI (YoY)': 'MEDIUM',
            'US Unemployment': 'MEDIUM',
            'US GDP Growth': 'MEDIUM',
            'Industrial Production': 'MEDIUM'
        }

        # Separate cache for economic indicators (1 day) and market data (15 min)
        self.market_cache = {
            'data': None,
            'timestamp': None,
            'duration': timedelta(minutes=15)
        }
        self.economic_cache = {
            'data': None,
            'timestamp': None,
            'duration': timedelta(days=1)  # Cache economic data for 1 day
        }

        # Add release IDs for our indicators
        self.release_ids = {
            'US CPI (YoY)': '10', # Consumer Price Index (CPI)
            'US PPI (YoY)': '13', # Producer Price Index (PPI)
            'Fed Interest Rate': '18', # Federal Funds Rate (FOMC)
            'US Unemployment': '8', # Employment Situation
            'US GDP Growth': '3', # Gross Domestic Product
            'US Core PCE': '21', # Personal Income and Outlays
            'Industrial Production': '14' # Industrial Production and Capacity Utilization
        }

    async def get_market_analysis(self) -> Dict:
        """Get comprehensive market analysis"""
        try:
            current_time = datetime.now()
            
            if self.cache and (current_time - self.cache['timestamp'] < self.cache_duration):
                return self.cache['data']

            # Get technical analysis and news concurrently
            forex_analysis, news_data = await asyncio.gather(
                self._analyze_forex(),
                self._get_market_news()
            )

            analysis = {
                'forex': forex_analysis,
                'news': news_data,
                'timestamp': current_time.isoformat(),
                'next_update': (current_time + self.cache_duration).isoformat()
            }
            
            self.cache = {
                'data': analysis,
                'timestamp': current_time
            }
            
            return analysis

        except Exception as e:
            logger.error(f"Error in market analysis: {e}")
            return self._get_error_response()

    async def _get_oanda_data(self, symbol: str) -> pd.DataFrame:
        """Get OANDA candle data"""
        params = {
            "count": 100,
            "granularity": "H1"
        }
        r = instruments.Candles(instrument=symbol, params=params)
        response = self.api.request(r)
        
        candles = response['candles']
        df = pd.DataFrame([{
            'close': float(c['mid']['c']),
            'open': float(c['mid']['o']),
            'high': float(c['mid']['h']),
            'low': float(c['mid']['l']),
            'volume': int(c['volume']),
            'time': c['time']
        } for c in candles])
        
        return df

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

    async def _get_market_news(self) -> Dict:
        """Get relevant market news"""
        try:
            articles = self.news_client.get_everything(
                q='forex OR "currency trading" OR "foreign exchange"',
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

    def get_economic_indicators(self) -> Dict:
        """Get economic indicators from FRED"""
        try:
            current_time = datetime.now()
            
            # Check cache
            if (self.economic_cache['data'] and 
                self.economic_cache['timestamp'] and 
                (current_time - self.economic_cache['timestamp'] < self.economic_cache['duration'])):
                return self.economic_cache['data']

            indicators_data = []
            for name, series_id in self.indicators.items():
                try:
                    data = self._fetch_fred_data(series_id)
                    if data and 'observations' in data:
                        obs = data['observations']
                        
                        # Get values and dates
                        current_value = float(obs[0]['value'])
                        current_date = obs[0]['date']
                        previous_values = [(float(o['value']), o['date']) for o in obs[1:4]]
                        
                        # Format values based on type
                        if series_id in ['FEDFUNDS', 'DGS10', 'ECBDFR']:
                            value_format = lambda x: f"{x:.2f}%"
                        else:
                            value_format = lambda x: f"{x:.1f}%"
                        
                        # Get next release date
                        next_release = self._get_next_release_date(name)
                        
                        indicator_data = {
                            'name': name,
                            'current': {
                                'value': value_format(current_value),
                                'date': current_date
                            },
                            'previous': [
                                {
                                    'value': value_format(value),
                                    'date': date
                                } for value, date in previous_values
                            ],
                            'trend': self._calculate_trend([current_value] + [v for v, _ in previous_values]),
                            'impact': self._get_impact_level(name),
                            'nextRelease': next_release
                        }
                        
                        indicators_data.append(indicator_data)
                        
                except Exception as e:
                    logger.error(f"Error processing {name}: {e}")
                    continue

            result = {
                'economic_indicators': indicators_data,
                'timestamp': current_time.isoformat(),
                'next_update': (current_time + self.economic_cache['duration']).isoformat()
            }

            # Update cache
            self.economic_cache.update({
                'data': result,
                'timestamp': current_time
            })

            return result

        except Exception as e:
            logger.error(f"Error fetching economic indicators: {e}")
            return {
                'error': True,
                'message': str(e)
            }

    def _fetch_fred_data(self, series_id: str) -> Dict:
        """Helper method to fetch data from FRED API"""
        url = 'https://api.stlouisfed.org/fred/series/observations'
        params = {
            'series_id': series_id,
            'api_key': self.fred_api_key,
            'file_type': 'json',
            'sort_order': 'desc',
            'limit': 6,  # Get last 6 observations (more history)
            'frequency': 'm'  # Monthly frequency
        }
        
        if series_id in ['FEDFUNDS', 'DGS10', 'ECBDFR']:
            params['units'] = 'lin'  # Linear units (actual values)
        else:
            params['units'] = 'pc1'  # Percent change from year ago

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Get release date info
            release_url = 'https://api.stlouisfed.org/fred/release/dates'
            release_params = {
                'release_id': self.release_ids.get(series_id, ''),
                'api_key': self.fred_api_key,
                'file_type': 'json',
                'sort_order': 'desc',  # Get most recent first
                'limit': 1
            }
            
            release_response = requests.get(release_url, release_params)
            if release_response.status_code == 200:
                release_data = release_response.json()
                if release_data.get('release_dates'):
                    data['last_release'] = release_data['release_dates'][0]['date']
            
            return data
            
        except Exception as e:
            logger.error(f"FRED API Error for {series_id}: {str(e)}")
            return None

    def _calculate_trend(self, values):
        """Calculate trend from values"""
        if len(values) < 2:
            return 'NEUTRAL'
        diff = values[0] - values[1]
        if abs(diff) < 0.1:  # Less than 0.1% change
            return 'NEUTRAL'
        return 'INCREASING' if diff > 0 else 'DECREASING'

    def _get_impact_level(self, indicator):
        """Get impact level for indicator"""
        return self.impact_levels.get(indicator, "MEDIUM") 