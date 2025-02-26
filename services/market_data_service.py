from typing import Dict, Optional, List
import logging
import asyncio
from datetime import datetime
from services.cache_service import Cache
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles
import os

logger = logging.getLogger(__name__)

class MarketDataService:
    def __init__(self, alpha_vantage_key: str):
        self.cache = Cache(expiry_minutes=240)  # 4 hour cache
        
        # Initialize OANDA client with correct env variables
        self.oanda = API(
            access_token=os.getenv('OANDA_ACCESS_TOKEN'),  # Changed from OANDA_API_KEY
            environment=os.getenv('OANDA_ENVIRONMENT', 'practice')
        )
        self.account_id = os.getenv('OANDA_ACCOUNT_ID')
        
        logger.info(f"Initialized OANDA client with account {self.account_id}")
        
        # Only forex pairs that we know work with your account
        self.symbol_map = {
            'EUR_USD': 'EUR_USD',
            'GBP_USD': 'GBP_USD',
            'USD_JPY': 'USD_JPY',
            'USD_CHF': 'USD_CHF',
            'AUD_USD': 'AUD_USD',
            'USD_CAD': 'USD_CAD',
            'NZD_USD': 'NZD_USD',
            'EUR_GBP': 'EUR_GBP',
            'EUR_JPY': 'EUR_JPY',
            'GBP_JPY': 'GBP_JPY'
        }

    async def get_historical_prices(self, symbol: str) -> List[Dict]:
        """Get historical price data for a symbol"""
        try:
            # Check cache first
            cache_key = f"historical_prices_{symbol}"
            cached_data = self.cache.get(cache_key)
            if cached_data:
                logger.info(f"Using cached data for {symbol}")
                return cached_data

            # Get symbol mapping
            oanda_symbol = self.symbol_map.get(symbol)
            if not oanda_symbol:
                logger.error(f"No symbol mapping for {symbol}")
                return []

            try:
                params = {
                    "count": 24,      # Last 24 hours
                    "granularity": "H1",  # 1 hour candles
                    "price": "M"      # Midpoint prices
                }
                
                request = InstrumentsCandles(
                    instrument=oanda_symbol,
                    params=params
                )
                
                logger.info(f"Fetching OANDA data for {symbol}")
                response = await asyncio.to_thread(self.oanda.request, request)
                
                # Debug the raw response
                logger.info(f"Raw OANDA response for {symbol}: {response}")
                
                if 'candles' not in response:
                    logger.error(f"No candles in response for {symbol}: {response}")
                    return []

                prices = []
                for candle in response['candles']:
                    try:
                        if not candle.get('complete'):
                            continue
                            
                        if 'mid' not in candle or 'c' not in candle['mid']:
                            logger.error(f"Invalid candle data: {candle}")
                            continue
                            
                        price = float(candle['mid']['c'])
                        timestamp = candle['time'][:16].replace('T', ' ')
                        
                        price_data = {
                            'timestamp': timestamp,
                            'price': price
                        }
                        prices.append(price_data)
                        logger.info(f"Added price: {price_data}")
                    except Exception as e:
                        logger.error(f"Error processing candle: {e}, candle: {candle}")
                        continue

                logger.info(f"Processed {len(prices)} prices for {symbol}")
                if prices:
                    self.cache.set(cache_key, prices)
                    return prices
                else:
                    logger.error(f"No valid prices found for {symbol}")
                    return []

            except Exception as e:
                logger.error(f"OANDA error for {symbol}: {e}")
                return []

        except Exception as e:
            logger.error(f"Error in get_historical_prices: {e}")
            return [] 