import logging
import asyncio
from typing import Dict, List
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

logger = logging.getLogger(__name__)

class MarketIntelligenceService:
    def __init__(self):
        self.api = API(access_token=OANDA_CREDS["ACCESS_TOKEN"])
        self.news_client = NewsApiClient(api_key=os.getenv('NEWS_API_KEY'))
        self.fred_api_key = os.getenv('FRED_API_KEY')
        if not self.fred_api_key:
            logger.error("FRED API key not found")
            raise ValueError("FRED API key is required")
        
        self.symbols = {
            'forex': ['EUR_USD', 'GBP_USD', 'USD_JPY', 'AUD_USD'],
            'commodities': ['XAU_USD', 'BCO_USD'],
            'indices': ['SPX500_USD', 'NAS100_USD', 'DE30_EUR']
        }
        
        # Map frontend assets to their specific indicators
        self.asset_specific_indicators = {
            # Forex Pairs
            'EURUSD': {
                'primary': 'Forex',
                'additional': [
                    'ECB Interest Rate',
                    'EU CPI (YoY)',
                    'EU GDP Growth',
                    'EU Unemployment',
                    'EU Manufacturing PMI'
                ]
            },
            'GBPUSD': {
                'primary': 'Forex',
                'additional': [
                    'BOE Interest Rate',
                    'UK CPI (YoY)',
                    'UK GDP Growth',
                    'UK Manufacturing PMI'
                ]
            },
            'AUDUSD': {
                'primary': 'Forex',
                'additional': [
                    'RBA Interest Rate',
                    'AU CPI (YoY)',
                    'AU GDP Growth',
                    'AU Trade Balance',
                    'China GDP Growth'  # Important for AUD
                ]
            },
            'USDJPY': {
                'primary': 'Forex',
                'additional': [
                    'BOJ Interest Rate',
                    'JP CPI (YoY)',
                    'JP GDP Growth',
                    'JP Trade Balance'
                ]
            },
            # Commodities
            'XAUUSD': {
                'primary': 'Gold',
                'additional': ['VIX Index', 'Global Economic Policy Uncertainty']
            },
            'XAGUSD': {
                'primary': 'Gold',
                'additional': [
                    'Industrial Production',  # Silver has industrial uses
                    'Global Manufacturing PMI'
                ]
            },
            'WTIUSD': {
                'primary': 'Oil',
                'additional': [
                    'EIA Crude Stocks',
                    'US Gasoline Demand',
                    'OPEC Production',
                    'Global Oil Demand'
                ]
            },
            # Indices
            'SPX': {
                'primary': 'Indices',
                'additional': [
                    'US Retail Sales',
                    'US Consumer Confidence',
                    'US Manufacturing PMI'
                ]
            },
            'NDX': {
                'primary': 'Indices',
                'additional': [
                    'US Tech Sector Growth',
                    'Semiconductor Index',
                    'Cloud Computing Index'
                ]
            },
            'DE30': {
                'primary': 'Indices',
                'additional': [
                    'EU Manufacturing PMI',
                    'German GDP Growth',
                    'German Industrial Production',
                    'ECB Interest Rate'
                ]
            },
            'N225': {
                'primary': 'Indices',
                'additional': [
                    'JP Manufacturing PMI',
                    'BOJ Interest Rate',
                    'JP Industrial Production'
                ]
            }
        }

        # Expanded indicators dictionary with all indicators
        self.asset_indicators = {
            'Economic': {
                'Fed Interest Rate': {
                    'series_id': 'FEDFUNDS',
                    'correlation': 0.92,
                    'importance': 98,
                    'description': 'Federal Reserve benchmark rate'
                },
                'US CPI (YoY)': {
                    'series_id': 'CPIAUCSL',
                    'correlation': 0.85,
                    'importance': 95,
                    'description': 'Key inflation indicator'
                },
                'US GDP Growth': {
                    'series_id': 'GDP',
                    'correlation': 0.82,
                    'importance': 95,
                    'description': 'Economic growth indicator'
                },
                'US Unemployment': {
                    'series_id': 'UNRATE',
                    'correlation': -0.75,
                    'importance': 90,
                    'description': 'Employment market health'
                },
                'Industrial Production': {
                    'series_id': 'INDPRO',
                    'correlation': 0.78,
                    'importance': 88,
                    'description': 'Manufacturing sector activity'
                },
                'Retail Sales': {
                    'series_id': 'RSAFS',
                    'correlation': 0.72,
                    'importance': 85,
                    'description': 'Consumer spending indicator'
                },
                'Consumer Confidence': {
                    'series_id': 'UMCSENT',
                    'correlation': 0.68,
                    'importance': 85,
                    'description': 'Consumer sentiment index'
                },
                'Housing Starts': {
                    'series_id': 'HOUST',
                    'correlation': 0.65,
                    'importance': 82,
                    'description': 'Real estate market activity'
                },
                'Building Permits': {
                    'series_id': 'PERMIT',
                    'correlation': 0.63,
                    'importance': 80,
                    'description': 'Future construction indicator'
                },
                'US PPI (YoY)': {
                    'series_id': 'PPIACO',
                    'correlation': 0.80,
                    'importance': 92,
                    'description': 'Producer Price Index - wholesale inflation'
                },
                'Houses Sold': {
                    'series_id': 'HSN1F',
                    'correlation': 0.65,
                    'importance': 82,
                    'description': 'New Single-Family Houses Sold'
                },
                'TIPS Spread': {
                    'series_id': 'T5YIE',
                    'correlation': 0.75,
                    'importance': 88,
                    'description': '5-Year Forward Inflation Expectation Rate'
                }
            },
            'Financial': {
                'US 10Y Treasury': {
                    'series_id': 'DGS10',
                    'correlation': -0.82,
                    'importance': 95,
                    'description': 'Benchmark government bond yield'
                },
                'M2 Money Supply': {
                    'series_id': 'M2SL',
                    'correlation': -0.72,
                    'importance': 85,
                    'description': 'Money supply growth'
                },
                'Corporate Profits': {
                    'series_id': 'CP',
                    'correlation': 0.75,
                    'importance': 88,
                    'description': 'Business profitability'
                },
                'VIX Index': {
                    'series_id': 'VIXCLS',
                    'correlation': -0.80,
                    'importance': 90,
                    'description': 'Market volatility indicator'
                },
                'Bank Credit': {
                    'series_id': 'TOTBKCR',
                    'correlation': 0.65,
                    'importance': 82,
                    'description': 'Bank lending activity'
                }
            },
            'International': {
                'US Trade Balance': {
                    'series_id': 'BOPGSTB',
                    'correlation': 0.75,
                    'importance': 85,
                    'description': 'Trade deficit/surplus'
                },
                'Dollar Index': {
                    'series_id': 'DTWEXB',
                    'correlation': -0.85,
                    'importance': 92,
                    'description': 'USD strength indicator'
                },
                'Global Economic Policy': {
                    'series_id': 'GEPUCURRENT',
                    'correlation': 0.75,
                    'importance': 88,
                    'description': 'Policy uncertainty index'
                },
                'Current Account': {
                    'series_id': 'BOPCA',
                    'correlation': 0.70,
                    'importance': 85,
                    'description': 'International payments balance'
                }
            },
            'Commodities': {
                'Oil Production': {
                    'series_id': 'IPG211111CN',
                    'correlation': -0.85,
                    'importance': 90,
                    'description': 'US crude oil production'
                },
                'Natural Gas Storage': {
                    'series_id': 'NGTS',
                    'correlation': -0.75,
                    'importance': 85,
                    'description': 'Natural gas inventories'
                },
                'Industrial Materials': {
                    'series_id': 'RIMP2',
                    'correlation': 0.72,
                    'importance': 82,
                    'description': 'Raw materials price index'
                }
            },
            'Manufacturing': {
                'ISM Manufacturing': {
                    'series_id': 'NAPM',
                    'correlation': 0.78,
                    'importance': 88,
                    'description': 'Manufacturing PMI'
                },
                'Capacity Utilization': {
                    'series_id': 'TCU',
                    'correlation': 0.72,
                    'importance': 85,
                    'description': 'Industrial capacity usage'
                },
                'Durable Goods': {
                    'series_id': 'DGORDER',
                    'correlation': 0.68,
                    'importance': 82,
                    'description': 'Manufacturing orders'
                },
                'Factory Orders': {
                    'series_id': 'AMTMNO',
                    'correlation': 0.65,
                    'importance': 80,
                    'description': 'Manufacturing demand'
                }
            },
            'Gold': {
                'Real Interest Rate': {
                    'series_id': 'DFII10',
                    'correlation': -0.88,
                    'importance': 95,
                    'description': 'Real 10-Year Treasury Yield'
                },
                'Gold ETF Holdings': {
                    'series_id': 'GLDTONS',
                    'correlation': 0.82,
                    'importance': 90,
                    'description': 'Total Known ETF Holdings of Gold'
                },
                'Central Bank Gold': {
                    'series_id': 'GOLDRESERVES',
                    'correlation': 0.75,
                    'importance': 88,
                    'description': 'Global Central Bank Gold Reserves'
                }
            },
            'Oil': {
                'US Oil Inventories': {
                    'series_id': 'WCESTUS1',
                    'correlation': -0.82,
                    'importance': 92,
                    'description': 'Weekly US Crude Oil Stocks'
                },
                'Oil Rig Count': {
                    'series_id': 'RIGCOUNT',
                    'correlation': 0.70,
                    'importance': 85,
                    'description': 'Baker Hughes US Oil Rig Count'
                },
                'OPEC Production': {
                    'series_id': 'OPECCRUDE',
                    'correlation': -0.85,
                    'importance': 90,
                    'description': 'OPEC Crude Oil Production'
                },
                'Global Oil Demand': {
                    'series_id': 'GLOWOILDEMAND',
                    'correlation': 0.88,
                    'importance': 93,
                    'description': 'Global Oil Consumption'
                }
            },
            'Forex': {
                'Trade Balance': {
                    'series_id': 'BOPGSTB',
                    'correlation': 0.75,
                    'importance': 88,
                    'description': 'Trade Balance - Goods and Services'
                },
                'Government Debt': {
                    'series_id': 'GFDEGDQ188S',
                    'correlation': -0.65,
                    'importance': 85,
                    'description': 'Federal Debt to GDP'
                },
                'Current Account': {
                    'series_id': 'BOPCA',
                    'correlation': 0.72,
                    'importance': 85,
                    'description': 'Current Account Balance'
                }
            },
            'Crypto': {
                'Tech Sector': {
                    'series_id': 'NASDAQ100',
                    'correlation': 0.85,
                    'importance': 90,
                    'description': 'Technology Sector Performance'
                },
                'Risk Sentiment': {
                    'series_id': 'VIXCLS',
                    'correlation': -0.75,
                    'importance': 88,
                    'description': 'Market Volatility Index'
                },
                'Monetary Base': {
                    'series_id': 'BOGMBASE',
                    'correlation': 0.70,
                    'importance': 85,
                    'description': 'Monetary Base Growth'
                }
            },
            'Market': {
                'Market Volatility': {
                    'series_id': 'VIXCLS',
                    'correlation': -0.80,
                    'importance': 90,
                    'description': 'VIX - Market Fear Index'
                },
                'Credit Spreads': {
                    'series_id': 'BAA10Y',
                    'correlation': -0.75,
                    'importance': 88,
                    'description': 'Corporate Bond Spread'
                },
                'Margin Debt': {
                    'series_id': 'BOGZ1FL663067003Q',
                    'correlation': 0.70,
                    'importance': 85,
                    'description': 'NYSE Margin Debt'
                }
            },
            'EU': {
                'ECB Interest Rate': {
                    'series_id': 'ECBDFR',  # ECB Deposit Facility Rate
                    'correlation': 0.88,
                    'importance': 95,
                    'description': 'European Central Bank benchmark rate'
                },
                'EU CPI (YoY)': {
                    'series_id': 'CP0000EZ19M086NEST',  # Euro Area HICP
                    'correlation': 0.82,
                    'importance': 92,
                    'description': 'Eurozone inflation rate'
                },
                'EU GDP Growth': {
                    'series_id': 'CLVMNACSCAB1GQEA19',  # Euro Area Real GDP
                    'correlation': 0.78,
                    'importance': 90,
                    'description': 'Eurozone economic growth'
                },
                'EU Unemployment': {
                    'series_id': 'LRHUTTTTEZM156S',  # Euro Area Unemployment Rate
                    'correlation': -0.72,
                    'importance': 85,
                    'description': 'Eurozone unemployment rate'
                },
                'EU Manufacturing PMI': {
                    'series_id': 'MPMUEUP',  # Markit Eurozone Manufacturing PMI
                    'correlation': 0.75,
                    'importance': 88,
                    'description': 'Eurozone manufacturing activity'
                }
            },
            'Japan': {
                'BOJ Interest Rate': {
                    'series_id': 'IRSTCI01JPM156N',  # Japan Policy Rate
                    'correlation': 0.85,
                    'importance': 92,
                    'description': 'Bank of Japan policy rate'
                },
                'Japan CPI (YoY)': {
                    'series_id': 'JPNCPIALLMINMEI',  # Japan CPI
                    'correlation': 0.75,
                    'importance': 88,
                    'description': 'Japan inflation rate'
                },
                'Japan GDP Growth': {
                    'series_id': 'JPNRGDPEXP',  # Japan Real GDP
                    'correlation': 0.72,
                    'importance': 85,
                    'description': 'Japan economic growth'
                },
                'Japan Industrial Production': {
                    'series_id': 'JPNPROINDMISMEI',  # Japan Industrial Production
                    'correlation': 0.70,
                    'importance': 85,
                    'description': 'Japan manufacturing output'
                },
                'Japan Trade Balance': {
                    'series_id': 'JPTBALA',  # Japan Trade Balance
                    'correlation': 0.68,
                    'importance': 82,
                    'description': 'Japan trade balance'
                }
            },
            'UK': {
                'BOE Interest Rate': {
                    'series_id': 'BOEBR',  # Bank of England Base Rate
                    'correlation': 0.88,
                    'importance': 95,
                    'description': 'Bank of England benchmark rate'
                },
                'UK CPI (YoY)': {
                    'series_id': 'GBRCPIALLMINMEI',  # UK CPI
                    'correlation': 0.82,
                    'importance': 90,
                    'description': 'UK inflation rate'
                },
                'UK GDP Growth': {
                    'series_id': 'GBRGDPEXP',  # UK Real GDP
                    'correlation': 0.75,
                    'importance': 88,
                    'description': 'UK economic growth'
                },
                'UK Manufacturing PMI': {
                    'series_id': 'GBMPMI',  # UK Manufacturing PMI
                    'correlation': 0.72,
                    'importance': 85,
                    'description': 'UK manufacturing activity'
                }
            },
            'China': {
                'China GDP Growth': {
                    'series_id': 'CHNGDPEXP',  # China Real GDP
                    'correlation': 0.85,
                    'importance': 92,
                    'description': 'China economic growth'
                },
                'China Manufacturing PMI': {
                    'series_id': 'CHNPMICN',  # China Manufacturing PMI
                    'correlation': 0.78,
                    'importance': 88,
                    'description': 'China manufacturing activity'
                },
                'China Trade Balance': {
                    'series_id': 'CHNTRBAL',  # China Trade Balance
                    'correlation': 0.75,
                    'importance': 85,
                    'description': 'China trade balance'
                }
            },
            'Australia': {
                'RBA Interest Rate': {
                    'series_id': 'RBATCTR',  # RBA Target Cash Rate
                    'correlation': 0.85,
                    'importance': 92,
                    'description': 'Reserve Bank of Australia rate'
                },
                'Australia CPI (YoY)': {
                    'series_id': 'AUSCPIALLQINMEI',  # Australia CPI
                    'correlation': 0.78,
                    'importance': 88,
                    'description': 'Australia inflation rate'
                },
                'Australia GDP Growth': {
                    'series_id': 'AUSGDPRPCAPCHPT',  # Australia Real GDP
                    'correlation': 0.75,
                    'importance': 85,
                    'description': 'Australia economic growth'
                }
            },
            'Inflation': {
                'US CPI (YoY)': {
                    'series_id': 'CPIAUCNS',  # Updated: Not seasonally adjusted CPI
                    'correlation': 0.85,
                    'importance': 95,
                    'description': 'US Consumer Price Index Year over Year'
                },
                'US Core CPI (YoY)': {
                    'series_id': 'CPILFESL',  # Core CPI (Less Food and Energy)
                    'correlation': 0.82,
                    'importance': 93,
                    'description': 'US Core inflation YoY excluding food and energy'
                },
                'US PCE (YoY)': {
                    'series_id': 'PCEPI',  # Updated: Personal Consumption Expenditures
                    'correlation': 0.88,
                    'importance': 96,
                    'description': 'Fed preferred inflation measure YoY'
                },
                'EU HICP (YoY)': {
                    'series_id': 'EA19CPALTT01GYM',  # Updated: Euro Area HICP
                    'correlation': 0.82,
                    'importance': 92,
                    'description': 'Eurozone Harmonized inflation rate YoY'
                },
                'UK CPI (YoY)': {
                    'series_id': 'GBRCPIALLMINMEI',  # UK CPI
                    'correlation': 0.80,
                    'importance': 90,
                    'description': 'UK inflation rate YoY'
                }
            }
        }

        # Release IDs for FRED API
        self.release_ids = {
            'US CPI (YoY)': '10',      # Consumer Price Index
            'US PPI (YoY)': '13',      # Producer Price Index
            'Fed Interest Rate': '18',  # Federal Funds Rate
            'US Unemployment': '8',     # Employment Situation
            'US GDP Growth': '3',       # Gross Domestic Product
            'US Core PCE': '21',        # Personal Income and Outlays
            'Industrial Production': '14', # Industrial Production
            'Manufacturing PMI': '33',   # ISM Manufacturing
            'Consumer Confidence': '55', # Consumer Sentiment
            'Corporate Profits': '112',  # Corporate Profits
            'Oil Inventories': '139',   # Petroleum Status Report
            'Real Interest Rate': '82'  # Treasury Real Yield Curve
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

        self.ai_analysis = AIAnalysisService(os.getenv('ALPHA_VANTAGE_KEY'))

        # Add more data sources
        self.alpha_vantage = TimeSeries(key=os.getenv('ALPHA_VANTAGE_KEY'))
        self.fred = Fred(api_key=os.getenv('FRED_API_KEY'))
        
        # Add probability calculation
        self.monte_carlo_sims = 1000

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

    def _fetch_fred_data(self, series_id: str) -> Dict:
        """Helper method to fetch data from FRED API"""
        url = 'https://api.stlouisfed.org/fred/series/observations'
        
        # Calculate dates to ensure we get recent data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # Get last year of data
        
        params = {
            'series_id': series_id,
            'api_key': self.fred_api_key,
            'file_type': 'json',
            'sort_order': 'desc',
            'observation_start': start_date.strftime('%Y-%m-%d'),
            'observation_end': end_date.strftime('%Y-%m-%d'),
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
            
            if not data.get('observations'):
                logger.error(f"No data returned for {series_id}")
                return None
                
            # Verify data is recent (not older than 6 months)
            latest_date = datetime.strptime(data['observations'][0]['date'], '%Y-%m-%d')
            if (end_date - latest_date).days > 180:
                logger.warning(f"Data for {series_id} is older than 6 months. Latest date: {latest_date.date()}")
            
            return data
            
        except Exception as e:
            logger.error(f"FRED API Error for {series_id}: {str(e)}")
            return None

    def get_economic_indicators(self, asset_category: str = None, asset_symbol: str = None) -> Dict:
        """Get all available economic indicators"""
        try:
            # Combine all indicators from all categories
            all_indicators = {}
            for category_indicators in self.asset_indicators.values():
                all_indicators.update(category_indicators)
            
            # Process indicators
            indicators_data = []
            for name, info in all_indicators.items():
                try:
                    data = self._fetch_fred_data(info['series_id'])
                    if not data or 'observations' not in data or not data['observations']:
                        continue

                    obs = data['observations']
                    current_value = float(obs[0]['value'])
                    current_date = obs[0]['date']
                    previous_values = [(float(o['value']), o['date']) for o in obs[1:4]]

                    # Format values
                    value_format = lambda x: f"{x:.2f}%" if info['series_id'] in ['FEDFUNDS', 'DGS10', 'ECBDFR'] else f"{x:.1f}%"

                    indicator_data = {
                        'name': name,
                        'current': {
                            'value': value_format(current_value),
                            'date': current_date
                        },
                        'previous': [{'value': value_format(v), 'date': d} for v, d in previous_values],
                        'trend': self._calculate_trend([current_value] + [v for v, _ in previous_values]),
                        'correlation': info['correlation'],
                        'importance': info['importance'],
                        'description': info['description']
                    }
                    indicators_data.append(indicator_data)

                except Exception as e:
                    logger.error(f"Error processing indicator {name}: {e}")
                    continue

            return {
                'economic_indicators': sorted(indicators_data, key=lambda x: x['importance'], reverse=True),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error in get_economic_indicators: {e}")
            return {
                'economic_indicators': [],
                'message': str(e)
            }

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

    def _calculate_correlation_and_importance(self, series_id: str, asset_type: str) -> tuple:
        """
        Calculate correlation with asset price and determine importance
        Returns (correlation, importance)
        """
        try:
            # Get 5 years of monthly data for the indicator
            end_date = datetime.now()
            start_date = end_date - timedelta(days=5*365)
            
            # Get indicator data
            indicator_data = self._fetch_fred_data(series_id)
            if not indicator_data or 'observations' not in indicator_data:
                return (0, 50)  # Default values if no data
            
            # Convert to pandas series
            indicator_series = pd.Series([
                float(obs['value']) for obs in indicator_data['observations']
                if not pd.isna(float(obs['value']))
            ])
            
            # Get corresponding asset price data (implementation depends on your data source)
            asset_prices = self._get_asset_historical_prices(asset_type, start_date, end_date)
            
            if asset_prices is None or len(asset_prices) < 10:
                return (0, 50)
            
            # Calculate correlation
            correlation = indicator_series.corr(asset_prices)
            
            # Calculate importance based on:
            # 1. Correlation strength (absolute value)
            # 2. Data reliability (number of observations)
            # 3. Leading indicator properties
            # 4. Market consensus (predefined weights)
            
            correlation_strength = abs(correlation) * 40  # Up to 40 points
            data_quality = min(len(indicator_series) / 60 * 20, 20)  # Up to 20 points
            
            # Predefined weights for different types of indicators
            indicator_weights = {
                'FEDFUNDS': 40,  # Interest rates
                'CPIAUCSL': 35,  # Inflation
                'GDP': 30,      # Growth
                'UNRATE': 25,   # Employment
                'INDPRO': 20,   # Production
                'UMCSENT': 15   # Sentiment
            }
            
            base_importance = indicator_weights.get(series_id, 15)
            
            # Combine all factors
            importance = correlation_strength + data_quality + base_importance
            importance = min(max(importance, 0), 100)  # Ensure between 0-100
            
            return (correlation, importance)
            
        except Exception as e:
            logger.error(f"Error calculating correlation for {series_id}: {e}")
            return (0, 50)  # Default values on error 

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

    async def _get_economic_indicators(self, asset: str) -> Dict:
        """Get economic indicators from FRED"""
        try:
            # Get relevant series based on asset
            indicators = {}
            
            # Get GDP growth
            gdp = self.fred.get_series('GDP')
            gdp_growth = ((gdp.iloc[-1] - gdp.iloc[-2]) / gdp.iloc[-2]) * 100
            
            # Get inflation
            cpi = self.fred.get_series('CPIAUCSL')
            inflation = ((cpi.iloc[-1] - cpi.iloc[-13]) / cpi.iloc[-13]) * 100  # YoY
            
            # Get interest rates
            interest_rate = self.fred.get_series('FEDFUNDS').iloc[-1]
            
            return {
                'gdp_growth': {
                    'value': f"{gdp_growth:.1f}%",
                    'trend': 'up' if gdp_growth > 0 else 'down',
                    'impact': 'HIGH'
                },
                'inflation': {
                    'value': f"{inflation:.1f}%",
                    'trend': 'up' if inflation > 2 else 'stable',
                    'impact': 'HIGH'
                },
                'interest_rate': {
                    'value': f"{interest_rate:.2f}%",
                    'trend': 'stable',
                    'impact': 'HIGH'
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting economic indicators: {e}", exc_info=True)
            return {}

    def _get_asset_historical_prices(self, asset_type: str, start_date: datetime, end_date: datetime) -> pd.Series:
        """Get historical prices for a specific asset"""
        # Implementation of _get_asset_historical_prices method
        pass

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