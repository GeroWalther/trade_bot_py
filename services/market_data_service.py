from typing import Dict, Optional
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import logging
import aiohttp
import asyncio
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class MarketDataService:
    def __init__(self, alpha_vantage_key: str):
        self.alpha_vantage_key = alpha_vantage_key
        self.ts = TimeSeries(key=alpha_vantage_key, output_format='pandas')
        self.cache = {}
        self.cache_duration = timedelta(minutes=15)  # Increased cache duration to 15 minutes

    async def get_latest_price(self, symbol: str) -> Dict:
        """Get latest price data from multiple sources with rate limiting"""
        try:
            # Check cache first
            cached_data = self._get_from_cache(symbol)
            if cached_data:
                logger.debug(f"Returning cached data for {symbol}")
                return cached_data

            # Add delay between requests to avoid rate limiting
            await asyncio.sleep(0.5)  # 500ms delay between requests

            # Try Alpha Vantage first
            try:
                data = await self._get_alpha_vantage_data(symbol)
                if data:
                    self._update_cache(symbol, data)
                    return data
            except Exception as e:
                logger.warning(f"Alpha Vantage data fetch failed: {e}")
                await asyncio.sleep(1)  # Additional delay before fallback

            # Fallback to Yahoo Finance
            data = await self._get_yahoo_finance_data(symbol)
            if data:
                self._update_cache(symbol, data)
                return data

            raise ValueError(f"Could not fetch data for symbol {symbol}")

        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return self._get_default_data(symbol)

    async def _get_alpha_vantage_data(self, symbol: str) -> Optional[Dict]:
        """Fetch data from Alpha Vantage"""
        try:
            # Format symbol for forex pairs
            formatted_symbol = symbol.replace('_', '')
            
            data, meta = await asyncio.to_thread(
                self.ts.get_quote_endpoint,
                formatted_symbol
            )

            # Properly handle pandas Series data
            return {
                'symbol': symbol,
                'price': float(data['05. price'].iloc[0]),  # Use iloc[0] to get the value
                'change': float(data['09. change'].iloc[0]),
                'change_percent': float(data['10. change percent'].iloc[0].strip('%')),
                'volume': int(data['06. volume'].iloc[0]),
                'timestamp': data['07. latest trading day'].iloc[0],
                'source': 'alpha_vantage'
            }
        except Exception as e:
            logger.error(f"Alpha Vantage error: {e}")
            return None

    async def _get_yahoo_finance_data(self, symbol: str) -> Optional[Dict]:
        """Fetch data from Yahoo Finance as backup"""
        try:
            # Convert forex pair format if needed
            yf_symbol = symbol.replace('_', '') + '=X' if '_' in symbol else symbol
            
            ticker = yf.Ticker(yf_symbol)
            data = await asyncio.to_thread(ticker.history, period='1d')
            
            if data.empty:
                return None
                
            latest = data.iloc[-1]
            return {
                'symbol': symbol,
                'price': float(latest['Close']),
                'change': float(latest['Close'] - latest['Open']),
                'change_percent': float((latest['Close'] - latest['Open']) / latest['Open'] * 100),
                'volume': int(latest['Volume']),
                'timestamp': data.index[-1].strftime('%Y-%m-%d'),
                'source': 'yahoo_finance'
            }
        except Exception as e:
            logger.error(f"Yahoo Finance error: {e}")
            return None

    def _get_from_cache(self, symbol: str) -> Optional[Dict]:
        """Get data from cache if valid"""
        if symbol in self.cache:
            data, timestamp = self.cache[symbol]
            if datetime.now() - timestamp < self.cache_duration:
                return data
        return None

    def _update_cache(self, symbol: str, data: Dict):
        """Update cache with new data"""
        self.cache[symbol] = (data, datetime.now())

    def _get_default_data(self, symbol: str) -> Dict:
        """Return default data structure when all sources fail"""
        return {
            'symbol': symbol,
            'price': 0.0,
            'change': 0.0,
            'change_percent': 0.0,
            'volume': 0,
            'timestamp': datetime.now().strftime('%Y-%m-%d'),
            'source': 'default',
            'error': 'Data unavailable'
        }

    async def get_historical_data(self, symbol: str, timeframe: str = 'DAILY') -> pd.DataFrame:
        """Get historical price data"""
        try:
            # Convert timeframe to Alpha Vantage interval
            interval_map = {
                'INTRADAY': '60min',
                'DAILY': 'daily',
                'WEEKLY': 'weekly',
                'MONTHLY': 'monthly'
            }
            interval = interval_map.get(timeframe.upper(), 'daily')

            if interval == '60min':
                data, _ = await asyncio.to_thread(
                    self.ts.get_intraday,
                    symbol.replace('_', ''),
                    interval='60min',
                    outputsize='full'
                )
            else:
                data, _ = await asyncio.to_thread(
                    self.ts.get_daily,
                    symbol.replace('_', ''),
                    outputsize='full'
                )

            return data

        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return pd.DataFrame()

    async def get_comprehensive_market_data(self, symbol: str) -> Dict:
        """Get market data from both Alpha Vantage and Yahoo Finance"""
        try:
            # Alpha Vantage Data
            av_data = await self._get_alpha_vantage_data(symbol)
            
            # Yahoo Finance Data
            yf_data = await self._get_yahoo_finance_data(symbol)
            
            # Combine and validate data
            combined_data = {
                'symbol': symbol,
                'alpha_vantage': av_data if av_data else None,
                'yahoo_finance': yf_data if yf_data else None,
                'timestamp': datetime.now().isoformat()
            }
            
            return combined_data
            
        except Exception as e:
            logger.error(f"Error getting comprehensive market data: {e}")
            return self._get_default_data(symbol)

    async def get_forecast_data(self, symbol: str) -> Dict:
        """Get data for different timeframe forecasts"""
        try:
            return {
                'intraday': {  # 1-8 hours
                    'data': await self._get_intraday_data(symbol),
                    'timeframe': '1-8h',
                    'indicators': await self._get_short_term_indicators(symbol)
                },
                'swing': {     # 2-5 days
                    'data': await self._get_daily_data(symbol, days=10),
                    'timeframe': '2-5d',
                    'indicators': await self._get_swing_indicators(symbol)
                },
                'short': {     # 1-4 weeks
                    'data': await self._get_daily_data(symbol, days=30),
                    'timeframe': '1-4w',
                    'indicators': await self._get_medium_term_indicators(symbol)
                },
                'mid': {       # 1-6 months
                    'data': await self._get_weekly_data(symbol, months=6),
                    'timeframe': '1-6m',
                    'indicators': await self._get_long_term_indicators(symbol)
                },
                'long': {      # 6+ months
                    'data': await self._get_monthly_data(symbol, years=2),
                    'timeframe': '6m+',
                    'indicators': await self._get_long_term_indicators(symbol)
                }
            }
        except Exception as e:
            logger.error(f"Error getting forecast data: {e}")
            return {}

    async def _get_intraday_data(self, symbol: str) -> pd.DataFrame:
        """Get intraday data"""
        try:
            data, _ = await asyncio.to_thread(
                self.ts.get_intraday,
                symbol.replace('_', ''),
                interval='60min',
                outputsize='full'
            )
            return data
        except Exception as e:
            logger.error(f"Error getting intraday data: {e}")
            return pd.DataFrame()

    async def _get_daily_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Get daily data"""
        try:
            data, _ = await asyncio.to_thread(
                self.ts.get_daily,
                symbol.replace('_', ''),
                outputsize='full'
            )
            return data.tail(days)
        except Exception as e:
            logger.error(f"Error getting daily data: {e}")
            return pd.DataFrame()

    async def _get_weekly_data(self, symbol: str, months: int = 6) -> pd.DataFrame:
        """Get weekly data"""
        try:
            data, _ = await asyncio.to_thread(
                self.ts.get_weekly,
                symbol.replace('_', '')
            )
            return data.tail(months * 4)  # Approximate weeks in months
        except Exception as e:
            logger.error(f"Error getting weekly data: {e}")
            return pd.DataFrame()

    async def _get_monthly_data(self, symbol: str, years: int = 2) -> pd.DataFrame:
        """Get monthly data"""
        try:
            data, _ = await asyncio.to_thread(
                self.ts.get_monthly,
                symbol.replace('_', '')
            )
            return data.tail(years * 12)
        except Exception as e:
            logger.error(f"Error getting monthly data: {e}")
            return pd.DataFrame()

    async def _get_short_term_indicators(self, symbol: str) -> Dict:
        """Get indicators for short-term analysis"""
        # Implement short-term technical indicators
        return {}

    async def _get_swing_indicators(self, symbol: str) -> Dict:
        """Get indicators for swing trading"""
        # Implement swing trading indicators
        return {}

    async def _get_medium_term_indicators(self, symbol: str) -> Dict:
        """Get indicators for medium-term analysis"""
        # Implement medium-term technical indicators
        return {}

    async def _get_long_term_indicators(self, symbol: str) -> Dict:
        """Get indicators for long-term analysis"""
        # Implement long-term technical indicators
        return {} 