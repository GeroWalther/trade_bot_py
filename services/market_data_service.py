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
        
        # Complete symbol mapping for all asset types
        self.symbol_map = {
            # Forex
            'EUR_USD': 'EUR_USD',
            'GBP_USD': 'GBP_USD',
            'USD_JPY': 'USD_JPY',
            'USD_CHF': 'USD_CHF',
            'AUD_USD': 'AUD_USD',
            'USD_CAD': 'USD_CAD',
            'NZD_USD': 'NZD_USD',
            'EUR_GBP': 'EUR_GBP',
            'EUR_JPY': 'EUR_JPY',
            'GBP_JPY': 'GBP_JPY',
            
            # Indices
            'SPX500_USD': 'SPX500_USD',  # S&P 500
            'NAS100_USD': 'NAS100_USD',  # NASDAQ
            'JP225_USD': 'JP225_USD',    # Nikkei
            'UK100_GBP': 'UK100_GBP',    # FTSE
            'DE30_EUR': 'DE30_EUR',      # DAX
            'EU50_EUR': 'EU50_EUR',      # Euro Stoxx 50
            
            # Commodities
            'XAU_USD': 'XAU_USD',        # Gold
            'XAG_USD': 'XAG_USD',        # Silver
            'BCO_USD': 'BCO_USD',        # Brent Crude Oil
            'WTICO_USD': 'WTICO_USD',    # WTI Crude Oil
            'NATGAS_USD': 'NATGAS_USD',  # Natural Gas
            
            # Crypto (if available on your OANDA account)
            'BTC_USD': 'BTC_USD',
            'ETH_USD': 'ETH_USD',
            'LTC_USD': 'LTC_USD'
        }

        # Log available instruments
        logger.info(f"Available trading instruments: {list(self.symbol_map.keys())}")

        # Update timeframe configurations
        self.timeframe_config = {
            'Intraday': {
                'count': 100,
                'granularity': 'M15'  # 15-minute candles
            },
            'Swing': {
                'count': 100,
                'granularity': 'H1'   # 1-hour candles
            },
            'Position': {
                'count': 100,
                'granularity': 'D'    # Daily candles
            }
        }

    async def get_historical_prices(self, symbol: str, timeframe: str = 'Intraday') -> List[Dict]:
        """Get historical price data for a symbol with specified timeframe"""
        try:
            # Include timeframe in cache key
            cache_key = f"historical_prices_{symbol}_{timeframe}"
            cached_data = self.cache.get(cache_key)
            if cached_data:
                logger.info(f"Using cached data for {symbol} ({timeframe})")
                return cached_data

            oanda_symbol = self.symbol_map.get(symbol)
            if not oanda_symbol:
                logger.error(f"Unsupported symbol: {symbol}")
                return []

            try:
                # Get timeframe configuration
                tf_config = self.timeframe_config.get(timeframe, self.timeframe_config['Intraday'])
                
                params = {
                    "count": tf_config['count'],
                    "granularity": tf_config['granularity'],
                    "price": "M"
                }
                
                logger.info(f"Fetching {timeframe} data for {symbol} with {tf_config['granularity']} candles")
                
                request = InstrumentsCandles(
                    instrument=oanda_symbol,
                    params=params
                )
                
                response = await asyncio.to_thread(self.oanda.request, request)
                
                if 'errorMessage' in response:
                    logger.error(f"OANDA error for {symbol}: {response['errorMessage']}")
                    if 'Invalid instrument' in response.get('errorMessage', ''):
                        # Remove invalid instrument from map
                        del self.symbol_map[symbol]
                        logger.warning(f"Removed invalid instrument {symbol} from symbol map")
                    return []

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
                logger.error(f"OANDA error for {symbol} ({oanda_symbol}): {e}")
                return []

        except Exception as e:
            logger.error(f"Error in get_historical_prices: {e}")
            return [] 