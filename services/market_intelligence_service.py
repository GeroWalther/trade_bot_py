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
from dotenv import load_dotenv
import json

# Force reload of environment variables
load_dotenv(override=True)

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
        # Debug environment variables
        logger.info("Environment variables:")
        logger.info(f"FRED_API_KEY: {os.getenv('FRED_API_KEY')}")
        logger.info(f"EIA_API_KEY: {os.getenv('EIA_API_KEY')}")
        
        self.base_url = 'https://api.stlouisfed.org/fred/series'
        self.fred_api_key = os.getenv('FRED_API_KEY')
        
        # Add debug logging to check API key
        logger.info(f"Initializing with FRED API key: {self.fred_api_key[:5]}...")
        
        if not self.fred_api_key:
            logger.error("FRED API key not found")
            raise ValueError("FRED API key is required")
        elif self.fred_api_key == 'your_fred_key':
            logger.error("FRED API key not properly set")
            raise ValueError("FRED API key is set to default value")
        
        # Initialize indicators cache
        self.indicators_cache = {}
        
        # Initialize cache for all data sources
        self.cache = {
            'fred': {'data': None, 'timestamp': None},
            'boe': {'data': None, 'timestamp': None},
            'boj': {'data': None, 'timestamp': None},
            'eia': {'data': None, 'timestamp': None},
            'last_cleared': datetime.now()
        }

        # Try to load persisted cache
        self._load_cache()

        # Update series IDs with correct ones
        self.series_ids = {
            # Core Economic - Verified working series
            'cpi': 'CPIAUCSL',            # CPI
            'core_cpi': 'CPILFESL',       # Core CPI
            'fed_rate': 'FEDFUNDS',       # Fed Funds Rate
            'unemployment': 'UNRATE',      # Unemployment Rate
            'nfp': 'PAYEMS',              # Non-Farm Payrolls
            'consumer_conf': 'UMCSENT',    # Consumer Confidence
            'repo_liquidity': 'RRPONTSYD', # Reverse Repo Operations
            'gdp': 'GDP',                  # Gross Domestic Product
            'retail_sales': 'RSAFS',       # Retail Sales
            'industrial_prod': 'INDPRO',   # Industrial Production
            
            # Interest Rates & Bonds - Verified working series
            'treasury_10y': 'GS10',        # 10-Year Treasury Rate (changed from DGS10)
            'treasury_2y': 'GS2',          # 2-Year Treasury Rate (changed from DGS2)
            'yield_curve': 'T10Y2Y',       # 10Y-2Y Spread
            'mortgage_rate': 'MORTGAGE30US', # 30-Year Mortgage Rate
            
            # Money & Banking - Verified working series
            'money_supply': 'M2SL',        # M2 Money Supply
            'bank_credit': 'TOTBKCR',      # Total Bank Credit
            
            # International - Verified working series
            'dollar_index': 'DTWEXBGS',    # Dollar Index
            'trade_balance': 'BOPGSTB',    # Trade Balance
            'foreign_reserves': 'TRESEUL',  # Updated series for Foreign Holdings
            
            # Market Related - Verified working series
            'sp500': 'SP500',              # S&P 500
            'vix': 'VIXCLS',               # VIX Volatility Index
            'credit_spread': 'BAA10Y',     # Corporate Bond Spread
            'commodities': 'PPIACO',       # Producer Price Index
            'oil_price': 'DCOILWTICO',     # WTI Crude Oil Price
            
            # Real Rates - Verified working series
            'real_rate_5y': 'DFII5',       # 5-Year TIPS Rate
            'real_rate_10y': 'DFII10',     # 10-Year TIPS Rate
            'breakeven_5y': 'T5YIE',       # 5-Year Breakeven Rate
            'breakeven_10y': 'T10YIE',     # 10-Year Breakeven Rate
            
            # Corporate Bonds - Verified working series
            'corp_aaa': 'AAA',             # AAA Corporate Bond Yield
            'corp_baa': 'BAA',             # BAA Corporate Bond Yield
            
            # Exchange Rates - Verified working series
            'dollar_major': 'DTWEXM',      # Dollar vs Major Currencies
            'reer_eur': 'RBXMBIS',         # Real Broad Euro
            'reer_jpy': 'RBJPBIS',         # Real Broad Yen
        }

        # Remove series that require alternative data sources
        self.series_ids.pop('high_yield_spread', None)  # Need alternative source
        self.series_ids.pop('inv_grade_spread', None)   # Need alternative source
        self.series_ids.pop('nasdaq_vol', None)         # Need alternative source
        self.series_ids.pop('semi_index', None)         # Need alternative source
        self.series_ids.pop('adv_decline', None)        # Need alternative source
        self.series_ids.pop('new_highs', None)          # Need alternative source
        self.series_ids.pop('sp500_pe', None)           # Need alternative source
        self.series_ids.pop('sp500_pb', None)           # Need alternative source
        self.series_ids.pop('put_call_ratio', None)     # Need alternative source
        self.series_ids.pop('margin_debt', None)        # Need alternative source

        # Comprehensive list of important indicators
        self.series_ids.update({
            # Market Liquidity
            'm2_growth': 'M2SL',          # M2 Money Supply Growth
            'repo_volume': 'RRPONTSYD',   # Repo Market Volume
            
            # Corporate Bond Metrics
            'high_yield_spread': 'BAMLH0A0HYM2',  # High Yield Spread
            'inv_grade_spread': 'BAMLC0A0CM',     # Investment Grade Spread
            
            # Stock Market Metrics
            'adv_decline': 'ADVN',        # NYSE Advance-Decline Line
            'new_highs': 'NHNL',          # New Highs minus New Lows
            
            # Valuation Metrics
            'sp500_pe': 'SP500_PE_RATIO_MONTH',  # S&P 500 P/E Ratio
            'sp500_pb': 'SP500_PB_RATIO',        # S&P 500 P/B Ratio
            
            # Sentiment Indicators
            'put_call_ratio': 'PCALLS',   # Put/Call Ratio
            'margin_debt': 'BOGZ1FL663067003Q',  # Margin Debt
            
            # Additional Market Health
            'institutional_flows': 'BOGZ1FL563064105Q',  # Institutional Investment Flows
            'earnings_growth': 'SP500_EARNINGS_GROWTH',  # S&P 500 Earnings Growth Rate
        })

        # Add forex-specific indicators
        self.series_ids.update({
            # Interest Rate Differentials
            'ecb_rate': 'ECBDFR',          # ECB Deposit Facility Rate
            'boe_rate': 'BOERUKM',         # Bank of England Rate
            'boj_rate': 'INTDSRJPM193N',   # Bank of Japan Rate
            
            # Central Bank Balance Sheets
            'fed_assets': 'WALCL',         # Federal Reserve Total Assets
            'ecb_assets': 'ECBASSETS',     # ECB Total Assets
            'boj_assets': 'BOJASSETS',     # BOJ Total Assets
            
            # Exchange Rate Metrics
            'reer_usd': 'RBXRBIS',        # Real Broad Effective Exchange Rate for United States
            'trade_weighted_usd': 'DTWEXB', # Trade Weighted U.S. Dollar Index
            
            # Capital Flows
            'tic_holdings': 'WMTSLA',      # Foreign Holdings of Treasury Securities
            'foreign_flows': 'NETFLI',     # Net Foreign Security Purchases
            
            # Risk & Sentiment
            'geopolitical_risk': 'GPRI',   # Geopolitical Risk Index
            'policy_uncertainty': 'USEPUINDXD', # Economic Policy Uncertainty Index
            'vix_currency': 'VXEFXCLS',    # Currency Volatility Index
            
            # Positioning & Flows
            'cot_euro': 'EUCFTC',          # EUR Commitment of Traders
            'cot_yen': 'JPYCFTC',          # JPY Commitment of Traders
            'cot_gbp': 'GBPCFTC',          # GBP Commitment of Traders
        })

        # Add commodity-specific indicators
        self.series_ids.update({
            # Oil & Energy
            'crude_inventories': 'WTTSTUS1',    # US Crude Oil Stocks
            'oil_production': 'PROMUS',         # US Oil Production
            'rig_count': 'RIGCOUNT',           # US Oil Rig Count (Baker Hughes)
            'gasoline_stocks': 'WGRSTUS1',     # US Gasoline Stocks
            
            # Precious Metals
            'gold_price': 'GOLDAMGBD228NLBM',  # London Gold Fixing Price
            'silver_price': 'SLVPRUSD',        # Silver Price
            'gold_etf': 'GLDTONS',             # Gold ETF Holdings
            'silver_etf': 'SLVTONS',           # Silver ETF Holdings
            'cb_gold_reserves': 'GOLDRESERVES', # Central Bank Gold Reserves
            
            # Interest Rates & Inflation
            'tips_spread': 'TIPSSPREAD',       # TIPS Spread
            
            # Dollar & Currency
            'dxy_index': 'DTWEXBGS',          # Dollar Index (Broad)
            
            # Global Demand
            'china_imports': 'XTIMVA01CNM657S', # China Import Value
            'india_imports': 'XTIMVA01INM657S', # India Import Value
            'global_trade': 'WTRADE',          # World Trade Volume
        })

        # Remove all category definitions
        self.indicator_categories = {}

        # Add metadata for better display
        self.indicators_metadata = {
            'fed_rate': {
                'name': 'Federal Reserve Rate',
                'description': 'US Federal Funds Rate',
                'importance': 98,
                'correlation': 95
            },
            'ecb_rate': {
                'name': 'ECB Rate',
                'description': 'European Central Bank Deposit Rate',
                'importance': 95,
                'correlation': 90
            },
            'crude_inventories': {
                'name': 'US Crude Inventories',
                'description': 'Weekly US commercial crude oil inventories',
                'importance': 92,
                'correlation': 85
            },
            'gold_price': {
                'name': 'Gold Price',
                'description': 'London Gold Fixing Price USD/oz',
                'importance': 95,
                'correlation': 88
            },
            'real_rate_10y': {
                'name': '10Y Real Rate',
                'description': '10-Year Treasury Inflation-Protected Security Rate',
                'importance': 94,
                'correlation': 90
            },
            'dxy_index': {
                'name': 'Dollar Index',
                'description': 'Trade Weighted US Dollar Index',
                'importance': 96,
                'correlation': 92
            }
        }

        # Add new API endpoints
        self.boj_api_url = 'https://www.stat-search.boj.or.jp/api/v1'
        self.eia_api_url = 'https://api.eia.gov/v2'
        self.eia_api_key = os.getenv('EIA_API_KEY')
        
        if not self.eia_api_key:
            logger.error("EIA API key not found")
            raise ValueError("EIA API key is required")

        # Add BOJ and EIA specific series
        self.boj_series = {
            'policy_rate': 'STSBDR01',          # Policy Rate
            'total_assets': 'MDBSAM1',          # Total Assets
            'monetary_base': 'MDMABS1'          # Monetary Base
        }

        # Initialize EIA series
        self.eia_series = {
            'wti_crude': 'PET.RWTC.W',           # WTI Crude Oil Price (weekly)
            'brent_crude': 'PET.RBRTE.W',        # Brent Crude Oil Price (weekly)
            'crude_stocks': 'PET.WCESTUS1.W',    # US Crude Oil Stocks
            'gasoline_stocks': 'PET.WGTSTUS1.W', # US Gasoline Stocks
            'distillate_stocks': 'PET.WDISTUS1.W', # Distillate Fuel Oil Stocks
            'crude_production': 'PET.WCRFPUS2.W', # US Crude Oil Production
            'refinery_runs': 'PET.WCRRIUS2.W'    # Gross Input to Refineries
        }

        # Remove BOJ series from FRED series_ids since we're using direct API
        self.series_ids.pop('boj_rate', None)
        self.series_ids.pop('boj_assets', None)

        # Add BOE API configuration
        self.boe_api_url = 'https://api.bankofengland.co.uk/v1'
        self.boe_series = {
            'bank_rate': 'IUDBEDR',              # Official Bank Rate
            'total_assets': 'RPQB3BM',           # Total Assets
            'reserves': 'RPQB3DM'                # Reserve Balances
        }

        # Update indicator categories
        self.indicator_categories = {
            'energy_markets': [
                'wti_crude', 'brent_crude', 'crude_stocks', 'gasoline_stocks',
                'distillate_stocks', 'crude_production', 'refinery_runs',
                'oil_imports', 'oil_exports'
            ],
            'central_banks': [
                'fed_rate', 'fed_assets', 'boe_bank_rate', 'boe_total_assets',
                'boe_reserves', 'boj_policy_rate', 'boj_total_assets',
                'boj_monetary_base'
            ],
            # ... other categories ...
        }

        # Remove BOE from FRED series since we're using direct API
        self.series_ids.pop('boe_rate', None)
        self.series_ids.pop('boe_assets', None)

    def _should_refresh_cache(self, source: str) -> bool:
        """Check if cache should be refreshed"""
        if not self.cache[source]['timestamp']:
            return True
            
        current_date = datetime.now()
        cache_date = self.cache[source]['timestamp']
        cache_age = current_date - cache_date
        
        # Only refresh if:
        # 1. Cache is older than 4 weeks
        # 2. Cache was explicitly cleared
        weeks_old = cache_age.days / 7
        return weeks_old > 4

    def clear_cache(self):
        """Explicitly clear all caches and indicators"""
        try:
            # Clear all caches
            self.cache = {
                'fred': {'data': None, 'timestamp': None},
                'boe': {'data': None, 'timestamp': None},
                'boj': {'data': None, 'timestamp': None},
                'eia': {'data': None, 'timestamp': None},
                'last_cleared': datetime.now()
            }
            
            # Clear indicators cache
            self.indicators_cache = {}
            
            # Force reload of environment variables
            load_dotenv(override=True)
            
            # Re-initialize API keys
            self.fred_api_key = os.getenv('FRED_API_KEY')
            self.eia_api_key = os.getenv('EIA_API_KEY')
            
            # Save cleared state
            self._save_cache()
            
            logger.info("All caches and indicators cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False

    def _save_cache(self):
        """Save cache to disk"""
        try:
            cache_file = 'market_intelligence_cache.json'
            cache_data = {
                'fred': self.cache['fred'],
                'boe': self.cache['boe'],
                'boj': self.cache['boj'],
                'eia': self.cache['eia'],
                'last_cleared': self.cache['last_cleared'].isoformat()
            }
            
            # Convert datetime objects to strings
            for source in ['fred', 'boe', 'boj', 'eia']:
                if cache_data[source]['timestamp']:
                    cache_data[source]['timestamp'] = cache_data[source]['timestamp'].isoformat()

            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
            logger.info("Cache saved to disk")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")

    def _load_cache(self):
        """Load cache from disk"""
        try:
            cache_file = 'market_intelligence_cache.json'
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                # Convert string dates back to datetime
                for source in ['fred', 'boe', 'boj', 'eia']:
                    if cache_data[source]['timestamp']:
                        cache_data[source]['timestamp'] = datetime.fromisoformat(cache_data[source]['timestamp'])
                
                cache_data['last_cleared'] = datetime.fromisoformat(cache_data['last_cleared'])
                self.cache = cache_data
                logger.info("Cache loaded from disk")
        except Exception as e:
            logger.error(f"Error loading cache: {e}")

    async def _fetch_fred_data(self, series_id: str) -> Optional[Dict]:
        """Simple FRED API fetch"""
        try:
            url = f"{self.base_url}/observations"
            params = {
                'series_id': series_id,
                'api_key': self.fred_api_key,
                'file_type': 'json',
                'sort_order': 'desc',
                'limit': 3  # Changed from 1 to 3 to get last 3 observations
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    response.raise_for_status()
                    return await response.json()

        except Exception as e:
            logger.error(f"Error fetching FRED data for {series_id}: {e}")
            return None

    async def get_economic_indicators(self):
        """Get all economic indicators"""
        try:
            indicators = {}
            
            # Fetch FRED data
            for indicator_id, series_id in self.series_ids.items():
                try:
                    data = await self._fetch_fred_data(series_id)
                    if data and data.get('observations'):
                        latest = data['observations'][0]
                        previous = data['observations'][1] if len(data['observations']) > 1 else None
                        
                        # Calculate trend
                        trend = 'neutral'
                        if previous and latest['value'] != '.' and previous['value'] != '.':
                            current_val = float(latest['value'])
                            prev_val = float(previous['value'])
                            trend = 'up' if current_val > prev_val else 'down' if current_val < prev_val else 'neutral'

                        indicators[indicator_id] = {
                            'name': indicator_id.replace('_', ' ').title(),
                            'value': latest['value'],
                            'date': latest['date'],
                            'category': 'other',  # All indicators in one category
                            'trend': trend,
                            'historical_data': [
                                {'value': obs['value'], 'date': obs['date']}
                                for obs in data['observations'][:3]
                            ]
                        }
                except Exception as e:
                    logger.error(f"Error fetching {indicator_id}: {e}")
                    continue

            return indicators

        except Exception as e:
            logger.error(f"Error in get_economic_indicators: {str(e)}")
            return {}

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

    def _process_indicator_data(self, data: Dict, indicator_id: str, category: str = None) -> Dict:
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
                'historical_data': historical_data,
                'category': category
            }

        except Exception as e:
            logger.error(f"Error processing indicator {indicator_id}: {e}")
            return None 

    async def verify_series_ids(self):
        """Verify all series IDs exist in FRED and log results"""
        logger.info("Verifying FRED series IDs...")
        invalid_series = []
        valid_series = []
        
        for indicator_id, series_id in self.series_ids.items():
            try:
                url = f"{self.base_url}/series"
                params = {
                    'series_id': series_id,
                    'api_key': self.fred_api_key,
                    'file_type': 'json'
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            valid_series.append((indicator_id, series_id))
                            logger.info(f"✓ Valid series: {indicator_id} ({series_id})")
                        else:
                            invalid_series.append((indicator_id, series_id))
                            logger.error(f"✗ Invalid series: {indicator_id} ({series_id})")
            
            except Exception as e:
                invalid_series.append((indicator_id, series_id))
                logger.error(f"Error verifying {indicator_id} ({series_id}): {e}")
        
        # Log summary
        logger.info(f"\nVerification Summary:")
        logger.info(f"Valid series: {len(valid_series)}")
        logger.info(f"Invalid series: {len(invalid_series)}")
        
        if invalid_series:
            logger.warning("\nInvalid Series List:")
            for indicator_id, series_id in invalid_series:
                logger.warning(f"- {indicator_id}: {series_id}")
        
        return {
            'valid': valid_series,
            'invalid': invalid_series
        } 

    async def fetch_boj_data(self) -> Dict:
        """Fetch BOJ data"""
        try:
            boj_data = {}
            
            for indicator_id, series_id in self.boj_series.items():
                url = f"{self.boj_api_url}/series/{series_id}"  # Updated endpoint
                params = {
                    'frequency': 'monthly',
                    'limit': 3
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data.get('data'):
                                latest = data['data'][0]
                                boj_data[f'boj_{indicator_id}'] = {
                                    'name': f'BOJ {indicator_id.replace("_", " ").title()}',
                                    'value': latest['value'],
                                    'date': latest['date'],
                                    'category': 'central_banks',
                                    'trend': self._calculate_trend(float(latest['value']), data['data']),
                                    'historical_data': [
                                        {'value': str(d['value']), 'date': d['date']}
                                        for d in data['data'][:3]
                                    ]
                                }
            
            return boj_data
            
        except Exception as e:
            logger.error(f"Error fetching BOJ data: {e}")
            return {}

    async def fetch_eia_data(self) -> Dict:
        """Fetch EIA oil inventory and production data"""
        try:
            if not self.eia_api_key:
                logger.error("EIA API key not found")
                return {}

            logger.info("Fetching EIA data...")
            
            # Updated EIA series with correct IDs
            self.eia_series = {
                'wti_crude': 'PET.RWTC.W',           # WTI Crude Oil Price (weekly)
                'brent_crude': 'PET.RBRTE.W',        # Brent Crude Oil Price (weekly)
                'crude_stocks': 'PET.WCESTUS1.W',    # US Crude Oil Stocks
                'gasoline_stocks': 'PET.WGTSTUS1.W', # US Gasoline Stocks
                'distillate_stocks': 'PET.WDISTUS1.W', # Distillate Fuel Oil Stocks
                'crude_production': 'PET.WCRFPUS2.W', # US Crude Oil Production
                'refinery_runs': 'PET.WCRRIUS2.W'    # Gross Input to Refineries
            }

            eia_data = {}
            for indicator_id, series_id in self.eia_series.items():
                url = "https://api.eia.gov/v2/seriesid"
                params = {
                    'api_key': self.eia_api_key,
                    'series_id': series_id,
                    'length': 3
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data.get('response', {}).get('data'):
                                series_data = data['response']['data']
                                latest = series_data[0]
                                
                                eia_data[indicator_id] = {
                                    'name': indicator_id.replace('_', ' ').title(),
                                    'value': str(latest['value']),
                                    'date': latest['period'],
                                    'category': 'energy_markets',
                                    'trend': self._calculate_trend(float(latest['value']), series_data),
                                    'historical_data': series_data[:3]
                                }
                                logger.info(f"Fetched EIA {indicator_id}: {latest['value']}")

            return eia_data

        except Exception as e:
            logger.error(f"Error fetching EIA data: {e}")
            return {}

    async def fetch_boe_data(self) -> Dict:
        """Fetch BOE data"""
        try:
            # Updated BOE endpoint and authentication
            headers = {
                'Accept': 'application/json',
                'Version': '1.0'
            }
            
            boe_data = {}
            for indicator_id, series_id in self.boe_series.items():
                url = f"{self.boe_api_url}/statistics/{series_id}/data"
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=headers) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data.get('data'):
                                latest = data['data'][0]
                                boe_data[f'boe_{indicator_id}'] = {
                                    'name': f'BOE {indicator_id.replace("_", " ").title()}',
                                    'value': str(latest['value']),
                                    'date': latest['date'],
                                    'category': 'central_banks',
                                    'trend': self._calculate_trend(float(latest['value']), data['data']),
                                    'historical_data': data['data'][:3]
                                }
                                logger.info(f"Fetched BOE {indicator_id}: {latest['value']}")

            return boe_data
            
        except Exception as e:
            logger.error(f"Error fetching BOE data: {e}")
            return {}

    async def get_commodity_data(self):
        """Get commodity-specific data with additional sources"""
        try:
            # Get FRED data first
            indicators = await self.get_economic_indicators()
            if not indicators:
                indicators = {}
            logger.info(f"FRED indicators count: {len(indicators)}")
            
            # Add BOE data
            try:
                boe_data = await self.fetch_boe_data()
                logger.info(f"BOE data: {json.dumps(boe_data, indent=2)}")
                indicators.update(boe_data or {})
            except Exception as e:
                logger.error(f"Error fetching BOE data: {e}")
            
            # Add BOJ data
            try:
                boj_data = await self.fetch_boj_data()
                logger.info(f"BOJ data: {json.dumps(boj_data, indent=2)}")
                indicators.update(boj_data or {})
            except Exception as e:
                logger.error(f"Error fetching BOJ data: {e}")
            
            # Add EIA data
            try:
                eia_data = await self.fetch_eia_data()
                logger.info(f"EIA data: {json.dumps(eia_data, indent=2)}")
                indicators.update(eia_data or {})
            except Exception as e:
                logger.error(f"Error fetching EIA data: {e}")

            return indicators

        except Exception as e:
            logger.error(f"Error in get_commodity_data: {str(e)}")
            return {} 