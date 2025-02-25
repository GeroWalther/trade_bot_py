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
        
        # Initialize cache
        self.cache = {
            'data': None,
            'timestamp': None
        }

        # Comprehensive list of important indicators
        self.series_ids = {
            # Core Economic
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
            
            # Interest Rates & Bonds
            'treasury_10y': 'DGS10',       # 10-Year Treasury Rate
            'treasury_2y': 'DGS2',         # 2-Year Treasury Rate
            'yield_curve': 'T10Y2Y',       # 10Y-2Y Spread
            'mortgage_rate': 'MORTGAGE30US', # 30-Year Mortgage Rate
            
            # Money & Banking
            'money_supply': 'M2SL',        # M2 Money Supply
            'bank_credit': 'TOTBKCR',      # Total Bank Credit
            
            # International
            'dollar_index': 'DTWEXBGS',    # Dollar Index
            'trade_balance': 'BOPGSTB',    # Trade Balance
            'foreign_reserves': 'TRESEGUSM052N', # Foreign Holdings of Treasuries
            
            # Market Related
            'sp500': 'SP500',              # S&P 500
            'vix': 'VIXCLS',               # VIX Volatility Index
            'credit_spread': 'BAA10Y',     # Corporate Bond Spread
            'commodities': 'PPIACO',       # Producer Price Index
            'oil_price': 'DCOILWTICO'      # WTI Crude Oil Price
        }

        # Add new indicators to series_ids
        self.series_ids.update({
            # Market Liquidity
            'm2_growth': 'M2SL',          # M2 Money Supply Growth
            'repo_volume': 'RRPONTSYD',   # Repo Market Volume
            
            # Corporate Bond Metrics
            'corp_aaa': 'AAA',            # AAA Corporate Bond Yield
            'corp_baa': 'BAA',            # BAA Corporate Bond Yield
            'high_yield_spread': 'BAMLH0A0HYM2',  # High Yield Spread
            'inv_grade_spread': 'BAMLC0A0CM',     # Investment Grade Spread
            
            # Stock Market Metrics
            'nasdaq_vol': 'VXNCLS',       # NASDAQ Volatility Index
            'semi_index': 'SOX',          # Philadelphia Semiconductor Index
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
            'fed_rate': 'FEDFUNDS',        # Federal Reserve Rate
            'ecb_rate': 'ECBDFR',          # ECB Deposit Facility Rate
            'boe_rate': 'BOERUKM',         # Bank of England Rate
            'boj_rate': 'INTDSRJPM193N',   # Bank of Japan Rate
            
            # Central Bank Balance Sheets
            'fed_assets': 'WALCL',         # Federal Reserve Total Assets
            'ecb_assets': 'ECBASSETS',     # ECB Total Assets
            'boj_assets': 'BOJASSETS',     # BOJ Total Assets
            
            # Exchange Rate Metrics
            'reer_usd': 'RBXRBIS',        # Real Broad Effective Exchange Rate for United States
            'reer_eur': 'RBXMBIS',        # Real Broad Effective Exchange Rate for Euro Area
            'reer_jpy': 'RBJPBIS',        # Real Broad Effective Exchange Rate for Japan
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
            'real_rate_5y': 'DFII5',           # 5-Year TIPS Rate
            'real_rate_10y': 'DFII10',         # 10-Year TIPS Rate
            'breakeven_5y': 'T5YIE',           # 5-Year Breakeven Inflation Rate
            'breakeven_10y': 'T10YIE',         # 10-Year Breakeven Inflation Rate
            'tips_spread': 'TIPSSPREAD',       # TIPS Spread
            
            # Dollar & Currency
            'dxy_index': 'DTWEXBGS',          # Dollar Index (Broad)
            'dollar_major': 'DTWEXM',         # Dollar vs Major Currencies
            
            # Global Demand
            'china_imports': 'XTIMVA01CNM657S', # China Import Value
            'india_imports': 'XTIMVA01INM657S', # India Import Value
            'global_trade': 'WTRADE',          # World Trade Volume
        })

        # Group indicators by category
        self.indicator_categories = {
            'market_liquidity': ['m2_growth', 'repo_volume'],
            'credit_markets': ['corp_aaa', 'corp_baa', 'high_yield_spread', 'inv_grade_spread'],
            'market_breadth': ['adv_decline', 'new_highs', 'nasdaq_vol'],
            'valuations': ['sp500_pe', 'sp500_pb', 'earnings_growth'],
            'sentiment': ['put_call_ratio', 'margin_debt', 'institutional_flows']
        }

        # Add new category for forex indicators
        self.indicator_categories.update({
            'forex_rates': [
                'fed_rate', 'ecb_rate', 'boe_rate', 'boj_rate'
            ],
            'central_banks': [
                'fed_assets', 'ecb_assets', 'boj_assets'
            ],
            'exchange_rates': [
                'reer_usd', 'reer_eur', 'reer_jpy', 'trade_weighted_usd'
            ],
            'capital_flows': [
                'tic_holdings', 'foreign_flows'
            ],
            'forex_risk': [
                'geopolitical_risk', 'policy_uncertainty', 'vix_currency'
            ],
            'positioning': [
                'cot_euro', 'cot_yen', 'cot_gbp'
            ]
        })

        # Add new categories
        self.indicator_categories.update({
            'energy_markets': [
                'crude_inventories', 'oil_production', 'rig_count', 
                'gasoline_stocks'
            ],
            'precious_metals': [
                'gold_price', 'silver_price', 'gold_etf', 'silver_etf',
                'cb_gold_reserves'
            ],
            'real_rates': [
                'real_rate_5y', 'real_rate_10y', 'breakeven_5y',
                'breakeven_10y', 'tips_spread'
            ],
            'currency_strength': [
                'dxy_index', 'dollar_major'
            ],
            'global_demand': [
                'china_imports', 'india_imports', 'global_trade'
            ]
        })

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
            'total_assets': 'BJ01MDAB11',      # BOJ Total Assets (fixed syntax)
            'monetary_base': 'BJ01MABS1A',      # Monetary Base
            'bond_holdings': 'BJ01MABA1A',      # JGB Holdings
            'etf_holdings': 'BJ01MABA2A',       # ETF Holdings
        }

        # Update EIA series with correct API v2 format based on documentation
        self.eia_series = {
            # Weekly Petroleum Status Report series
            'crude_stocks': 'PET.WCESTUS1.W',        # Weekly Crude Oil Stocks
            'gasoline_stocks': 'PET.WGTSTUS1.W',     # Weekly Gasoline Stocks
            'distillate_stocks': 'PET.WDISTUS1.W',   # Weekly Distillate Stocks
            'crude_production': 'PET.WCRFPUS2.W',    # Weekly Crude Production
            'refinery_input': 'PET.WCRRIUS2.W',      # Weekly Refinery Input
            'crude_imports': 'PET.WCEIMUS2.W',       # Weekly Crude Imports
            
            # Additional oil market indicators
            'crude_price': 'PET.RWTC.D',            # WTI Crude Oil Price
            'gasoline_price': 'PET.EMM_EPM0_PTE_NUS_DPG.W',  # Weekly Retail Gasoline Price
            'refinery_capacity': 'PET.WPULEUS3.W',  # Weekly Refinery Utilization
            'crude_exports': 'PET.WCREXUS2.W',      # Weekly Crude Exports
        }

    def _should_refresh_cache(self) -> bool:
        """Check if cache should be refreshed based on current date"""
        if not self.cache['timestamp']:
            return True
            
        current_date = datetime.now()
        cache_date = self.cache['timestamp']
        
        # Refresh if:
        # 1. It's a new month (1st day)
        # 2. We're in a different month than the cache
        # 3. Cache is from previous year
        return (current_date.day == 1 or 
                current_date.month != cache_date.month or 
                current_date.year != cache_date.year)

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
        """Get all economic indicators with monthly caching"""
        try:
            # Check if we should use cache
            if self.cache['data'] and not self._should_refresh_cache():
                logger.info("Using cached indicators from: " + 
                          self.cache['timestamp'].strftime('%Y-%m-%d'))
                return self.cache['data']

            logger.info("Fetching fresh indicators data")
            indicators = {}
            
            for indicator_id, series_id in self.series_ids.items():
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
                    
                    # Find category
                    category = 'other'
                    for cat, ids in self.indicator_categories.items():
                        if indicator_id in ids:
                            category = cat
                            break

                    historical = [
                        {
                            'value': obs['value'],
                            'date': obs['date']
                        }
                        for obs in data['observations'][:3]
                    ]
                    
                    indicators[indicator_id] = {
                        'name': indicator_id.replace('_', ' ').title(),
                        'value': latest['value'],
                        'date': latest['date'],
                        'category': category,
                        'trend': trend,
                        'historical_data': historical
                    }
                    logger.info(f"Fetched {indicator_id}: {indicators[indicator_id]}")

            # Update cache with new data
            self.cache['data'] = indicators
            self.cache['timestamp'] = datetime.now()
            
            return indicators

        except Exception as e:
            logger.error(f"Error fetching indicators: {e}")
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
        """Fetch BOJ balance sheet and monetary data"""
        try:
            boj_data = {}
            
            for indicator_id, series_code in self.boj_series.items():
                url = f"{self.boj_api_url}/stats"
                params = {
                    'code': series_code,
                    'lang': 'en',
                    'format': 'json',
                    'limit': 3
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            # Process BOJ data format
                            latest = data['observations'][0]
                            previous = data['observations'][1] if len(data['observations']) > 1 else None
                            
                            trend = 'neutral'
                            if previous:
                                current_val = float(latest['value'])
                                prev_val = float(previous['value'])
                                trend = 'up' if current_val > prev_val else 'down'
                            
                            boj_data[indicator_id] = {
                                'name': indicator_id.replace('_', ' ').title(),
                                'value': latest['value'],
                                'date': latest['date'],
                                'category': 'central_banks',
                                'trend': trend,
                                'historical_data': [
                                    {'value': obs['value'], 'date': obs['date']}
                                    for obs in data['observations'][:3]
                                ]
                            }
                            
                            logger.info(f"Fetched BOJ {indicator_id}: {boj_data[indicator_id]}")
            
            return boj_data
            
        except Exception as e:
            logger.error(f"Error fetching BOJ data: {e}")
            return {}

    async def fetch_eia_data(self) -> Dict:
        """Fetch EIA oil inventory and production data"""
        try:
            eia_data = {}
            
            for indicator_id, series_id in self.eia_series.items():
                url = f"{self.eia_api_url}/series/{series_id}/data/"
                params = {
                    'api_key': self.eia_api_key,
                    'frequency': 'weekly',
                    'data[0]': 'value',
                    'data[1]': 'period',
                    'sort[0][column]': 'period',
                    'sort[0][direction]': 'desc',
                    'length': 3  # Get last 3 data points
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            # Process EIA data format
                            series_data = data['response']['data']
                            if not series_data:
                                continue
                                
                            latest = series_data[0]
                            previous = series_data[1] if len(series_data) > 1 else None
                            
                            trend = 'neutral'
                            if previous:
                                current_val = float(latest['value'])
                                prev_val = float(previous['value'])
                                trend = 'up' if current_val > prev_val else 'down'
                            
                            eia_data[indicator_id] = {
                                'name': indicator_id.replace('_', ' ').title(),
                                'value': str(latest['value']),
                                'date': latest['period'],
                                'category': 'energy_markets',
                                'trend': trend,
                                'historical_data': [
                                    {'value': str(point['value']), 'date': point['period']}
                                    for point in series_data[:3]
                                ]
                            }
                            
                            logger.info(f"Fetched EIA {indicator_id}: {eia_data[indicator_id]}")
                        else:
                            logger.error(f"Error fetching {indicator_id}: {response.status}")
            
            return eia_data
            
        except Exception as e:
            logger.error(f"Error fetching EIA data: {e}")
            return {}

    async def get_commodity_data(self):
        """Get commodity-specific data with additional sources"""
        try:
            # Get FRED data first
            indicators = await self.get_economic_indicators()
            
            # Add BOJ data
            boj_data = await self.fetch_boj_data()
            indicators.update(boj_data)
            
            # Add EIA data
            eia_data = await self.fetch_eia_data()
            indicators.update(eia_data)
            
            # Add calculated real rates
            if 'fed_rate' in indicators and 'cpi' in indicators:
                fed_rate = float(indicators['fed_rate']['value'])
                cpi = float(indicators['cpi']['value'])
                real_rate = fed_rate - cpi
                
                indicators['real_fed_rate'] = {
                    'name': 'Real Fed Rate',
                    'value': f"{real_rate:.2f}",
                    'trend': 'up' if real_rate > 0 else 'down',
                    'category': 'real_rates',
                    'date': indicators['fed_rate']['date']
                }

            return indicators

        except Exception as e:
            logger.error(f"Error fetching commodity data: {e}")
            raise 