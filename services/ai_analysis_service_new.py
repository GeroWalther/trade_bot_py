import logging
import os
from typing import Dict
import pandas as pd
import numpy as np
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class AIAnalysisService:
    def __init__(self, alpha_vantage_key: str = None):
        logger.info("Initializing AI Analysis Service")
        self.oanda_access_token = os.getenv('OANDA_ACCESS_TOKEN')
        if not self.oanda_access_token:
            raise ValueError("OANDA_ACCESS_TOKEN is required")
            
        self.client = API(access_token=self.oanda_access_token)
        logger.info("AI Analysis Service initialized successfully")

    async def get_technical_analysis(self, symbol: str, timeframe: str = 'SWING') -> Dict:
        """Get technical indicators using OANDA data"""
        try:
            logger.info(f"Starting technical analysis for {symbol} with timeframe {timeframe}")
            
            # Format symbol for OANDA based on asset type
            if symbol.startswith('XAU'):  # Gold
                formatted_symbol = 'XAU_USD'
            elif symbol.startswith('XAG'):  # Silver
                formatted_symbol = 'XAG_USD'
            elif symbol.startswith('BCO'):  # Brent Crude Oil
                formatted_symbol = 'BCO_USD'
            elif symbol.startswith('DE30'):  # DAX
                formatted_symbol = 'DE30_EUR'
            elif symbol.startswith('SPX500'):  # S&P 500
                formatted_symbol = 'SPX500_USD'
            elif symbol.startswith('NAS100'):  # Nasdaq
                formatted_symbol = 'NAS100_USD'
            elif symbol.startswith('JP225'):  # Nikkei
                formatted_symbol = 'JP225_USD'
            elif '_' not in symbol and len(symbol) == 6:  # Forex pairs
                formatted_symbol = f"{symbol[:3]}_{symbol[3:]}"
            else:
                formatted_symbol = symbol

            # List of supported instruments
            SUPPORTED_INSTRUMENTS = {
                'EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CHF', 'AUD_USD', 
                'USD_CAD', 'NZD_USD', 'EUR_GBP', 'EUR_JPY', 'GBP_JPY',
                'XAU_USD', 'XAG_USD', 'BCO_USD', 'DE30_EUR',
                'SPX500_USD', 'NAS100_USD', 'JP225_USD'  # Added indices
            }

            if formatted_symbol not in SUPPORTED_INSTRUMENTS:
                raise ValueError(f"Unsupported trading instrument: {symbol}")
            
            # Map timeframe to appropriate granularity
            timeframe_config = {
                'INTRADAY': {
                    'granularity': 'M15',
                    'count': 100
                },
                'SWING': {
                    'granularity': 'H1',
                    'count': 100
                },
                'POSITION': {
                    'granularity': 'D',
                    'count': 100
                }
            }

            # Get configuration for requested timeframe
            config = timeframe_config.get(timeframe.upper(), timeframe_config['SWING'])
            logger.info(f"Timeframe '{timeframe}' mapped to config: {config}")
            
            # Get candles from OANDA with appropriate granularity
            params = {
                "count": config['count'],
                "granularity": config['granularity'],
                "price": "M"
            }
            logger.info(f"Making OANDA request with params: {params}")
            
            try:
                r = instruments.InstrumentsCandles(instrument=formatted_symbol, params=params)
                response = self.client.request(r)
                logger.info(f"OANDA response received with {len(response.get('candles', []))} candles")
            except Exception as e:
                logger.error(f"OANDA API error for {formatted_symbol}: {e}")
                raise ValueError(f"Invalid or unsupported symbol: {symbol}")
            
            # Convert to pandas DataFrame
            candles = pd.DataFrame([
                {
                    'time': c['time'],
                    'open': float(c['mid']['o']),
                    'high': float(c['mid']['h']),
                    'low': float(c['mid']['l']),
                    'close': float(c['mid']['c']),
                    'volume': int(c['volume'])
                } for c in response['candles']
            ])
            
            if len(candles) == 0:
                raise ValueError(f"No candle data returned for {formatted_symbol}")
            
            logger.info(f"Got {len(candles)} candles")
            
            # Calculate indicators
            closes = candles['close'].values
            current_close = closes[-1]  # Store current close for EMA comparison
            
            # RSI
            rsi = self._calculate_rsi(closes)
            if len(rsi) > 0:
                rsi_value = float(rsi[-1])
            else:
                rsi_value = None
            
            # MACD
            macd, signal, hist = self._calculate_macd(closes)
            
            # EMA-50
            ema = self._calculate_ema(closes)
            if len(ema) > 0:
                ema_value = float(ema[-1])
            else:
                ema_value = None
            
            # Calculate ATR first since other calculations need it
            atr, candles_with_tr = self._calculate_atr(candles)
            
            # Support/Resistance
            support, resistance = self._calculate_support_resistance(candles)
            
            # ADX
            adx = self._calculate_adx(candles)
            if len(adx) > 0:
                adx_value = float(adx[-1])
            else:
                adx_value = None
            
            # Calculate signals
            macd_signal = self._get_macd_signal(
                macd[-1] if len(macd) > 0 else 0,
                signal[-1] if len(signal) > 0 else 0
            )
            
            rsi_signal, rsi_strength = self._get_rsi_signal(rsi_value or 50)
            
            # Calculate average ATR for comparison using the TR column we just created
            avg_atr = candles_with_tr['TR'].rolling(14 * 2).mean().mean()
            atr_status = self._get_atr_status(atr, avg_atr)
            
            # Determine EMA trend
            ema_signal = "Support" if current_close > ema_value else "Resistance" if ema_value else "NEUTRAL"

            indicators = {
                'RSI': {
                    'value': rsi_value,
                    'signal': rsi_signal,
                    'strength': rsi_strength
                },
                'MACD': {
                    'value': float(macd[-1]) if len(macd) > 0 else None,
                    'signal': float(signal[-1]) if len(signal) > 0 else None,
                    'hist': float(hist[-1]) if len(hist) > 0 else None,
                    'trend': macd_signal
                },
                'EMA50': {
                    'value': ema_value,
                    'signal': ema_signal,
                    'current_price': float(current_close)
                },
                'ATR': {
                    'value': float(atr) if atr is not None else None,
                    'status': atr_status
                },
                'levels': {
                    'support': [float(s) for s in support[:2]] if len(support) > 0 else [],
                    'resistance': [float(r) for r in resistance[:2]] if len(resistance) > 0 else []
                }
            }
            
            logger.info(f"Successfully calculated indicators: {indicators}")
            return {
                'status': 'success',
                'data': indicators
            }

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}", exc_info=True)
            return {
                'status': 'error',
                'message': str(e)
            }

    def _calculate_rsi(self, closes: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI"""
        deltas = np.diff(closes)
        gain = np.where(deltas > 0, deltas, 0)
        loss = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.concatenate(([0], np.convolve(gain, np.ones(period)/period, mode='valid')))
        avg_loss = np.concatenate(([0], np.convolve(loss, np.ones(period)/period, mode='valid')))
        
        rs = avg_gain / np.where(avg_loss == 0, 1, avg_loss)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, closes: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """Calculate MACD"""
        exp1 = pd.Series(closes).ewm(span=fast, adjust=False).mean()
        exp2 = pd.Series(closes).ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd.values, signal_line.values, histogram.values

    def _calculate_ema(self, closes: np.ndarray, period: int = 50) -> np.ndarray:
        """Calculate EMA"""
        return pd.Series(closes).ewm(span=period, adjust=False).mean().values

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> tuple:
        """Calculate Average True Range"""
        df = df.copy()
        df['H-L'] = df['high'] - df['low']
        df['H-PC'] = abs(df['high'] - df['close'].shift(1))
        df['L-PC'] = abs(df['low'] - df['close'].shift(1))
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        atr = df['TR'].rolling(period).mean().iloc[-1]
        return atr, df  # Return both ATR value and DataFrame with TR column

    def _calculate_support_resistance(self, df: pd.DataFrame, window: int = 20) -> tuple:
        """Calculate Support and Resistance levels using pivot points"""
        df = df.copy()
        
        # Calculate pivot points
        df['PP'] = (df['high'] + df['low'] + df['close']) / 3
        df['R1'] = 2 * df['PP'] - df['low']
        df['S1'] = 2 * df['PP'] - df['high']
        df['R2'] = df['PP'] + (df['high'] - df['low'])
        df['S2'] = df['PP'] - (df['high'] - df['low'])
        
        # Get recent levels
        support = sorted([
            df['S1'].iloc[-1],
            df['S2'].iloc[-1],
            df['low'].rolling(window).min().iloc[-1]
        ])
        
        resistance = sorted([
            df['R1'].iloc[-1],
            df['R2'].iloc[-1],
            df['high'].rolling(window).max().iloc[-1]
        ], reverse=True)
        
        return support, resistance

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> np.ndarray:
        """Calculate ADX"""
        df = df.copy()
        df['TR'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['+DM'] = np.where(
            (df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
            np.maximum(df['high'] - df['high'].shift(1), 0),
            0
        )
        df['-DM'] = np.where(
            (df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
            np.maximum(df['low'].shift(1) - df['low'], 0),
            0
        )
        
        df['TR' + str(period)] = df['TR'].rolling(window=period).mean()
        df['+DM' + str(period)] = df['+DM'].rolling(window=period).mean()
        df['-DM' + str(period)] = df['-DM'].rolling(window=period).mean()
        
        df['+DI' + str(period)] = 100 * df['+DM' + str(period)] / df['TR' + str(period)]
        df['-DI' + str(period)] = 100 * df['-DM' + str(period)] / df['TR' + str(period)]
        df['DX'] = 100 * abs(df['+DI' + str(period)] - df['-DI' + str(period)]) / (df['+DI' + str(period)] + df['-DI' + str(period)])
        adx = df['DX'].rolling(window=period).mean()
        
        return adx.values 

    def _get_macd_signal(self, macd: float, signal: float) -> str:
        """Get MACD signal strength"""
        if macd > signal:
            return "BULLISH"
        elif macd < signal:
            return "BEARISH"
        return "NEUTRAL"

    def _get_rsi_signal(self, rsi: float) -> tuple:
        """Get RSI signal and strength"""
        if rsi > 70:
            return "OVERBOUGHT", 100
        elif rsi < 30:
            return "OVERSOLD", 0
        return "NEUTRAL", 50

    def _get_atr_status(self, atr: float, avg_atr: float) -> str:
        """Get ATR status"""
        if atr > avg_atr * 1.5:
            return "HIGH VOLATILITY"
        elif atr < avg_atr * 0.5:
            return "LOW VOLATILITY"
        return "NORMAL RANGE" 