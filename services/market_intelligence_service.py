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

logger = logging.getLogger(__name__)

class MarketIntelligenceService:
    def __init__(self):
        self.api = API(access_token=OANDA_CREDS["ACCESS_TOKEN"])
        self.news_client = NewsApiClient(api_key='52deeb5fce334ae2b127e2b08efa8335')
        
        self.symbols = {
            'forex': ['EUR_USD', 'GBP_USD', 'USD_JPY', 'AUD_USD'],
            'commodities': ['XAU_USD', 'BCO_USD'],
            'indices': ['SPX500_USD', 'NAS100_USD', 'DE30_EUR']
        }
        
        self.cache = {}
        self.cache_duration = timedelta(minutes=15)

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