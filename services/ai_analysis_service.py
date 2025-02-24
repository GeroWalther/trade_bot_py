from anthropic import Anthropic
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import re
from alpha_vantage.techindicators import TechIndicators
import os
import asyncio

logger = logging.getLogger(__name__)

class AIAnalysisService:
    def __init__(self, alpha_vantage_key: str = None):
        # Get Anthropic API key
        self.anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        if not self.anthropic_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")
            
        # Initialize Anthropic client
        self.client = Anthropic(api_key=self.anthropic_key)
        
        # Set Alpha Vantage key
        self.alpha_vantage_key = alpha_vantage_key or os.getenv('ALPHA_VANTAGE_API_KEY')
        if not self.alpha_vantage_key:
            raise ValueError("ALPHA_VANTAGE_API_KEY is required")
            
        # Initialize Alpha Vantage
        self.ti = TechIndicators(key=self.alpha_vantage_key, output_format='pandas')
        logger.info("AI Analysis Service initialized successfully")

    async def get_technical_analysis(self, symbol: str) -> Dict:
        """Get technical indicators from Alpha Vantage"""
        try:
            # Format symbol for Alpha Vantage (remove underscore)
            formatted_symbol = symbol.replace('_', '')
            if not formatted_symbol:
                raise ValueError("Invalid symbol")
            
            logger.info(f"Getting technical indicators for {formatted_symbol}")
            
            # Add delay to avoid rate limiting
            await asyncio.sleep(1)
            
            # Get RSI
            rsi_data = await asyncio.to_thread(
                self.ti.get_rsi,
                symbol=formatted_symbol,
                interval='60min',
                time_period=14,
                series_type='close'
            )
            
            if rsi_data is None or rsi_data.empty:
                raise ValueError(f"No RSI data returned for {formatted_symbol}")
            
            logger.info("RSI data fetched successfully")
            logger.info(f"RSI data: {rsi_data}")
            
            # Extract RSI value
            rsi_value = float(rsi_data['RSI'].iloc[-1])
            logger.info(f"RSI value: {rsi_value}")
            
            # Get MACD
            macd_data = await asyncio.to_thread(
                self.ti.get_macd,
                symbol=formatted_symbol,
                interval='60min',
                series_type='close'
            )
            logger.info("MACD data fetched successfully")
            
            # Get SMA
            sma_data = await asyncio.to_thread(
                self.ti.get_sma,
                symbol=formatted_symbol,
                interval='60min',
                time_period=20,
                series_type='close'
            )
            logger.info("SMA data fetched successfully")
            
            # Get ADX
            adx_data = await asyncio.to_thread(
                self.ti.get_adx,
                symbol=formatted_symbol,
                interval='60min',
                time_period=14
            )
            logger.info("ADX data fetched successfully")

            # Extract latest values
            indicators = {
                'RSI': rsi_value,
                'MACD': {
                    'macd': float(macd_data['MACD'].iloc[-1]),
                    'signal': float(macd_data['MACD_Signal'].iloc[-1]),
                    'hist': float(macd_data['MACD_Hist'].iloc[-1])
                },
                'SMA': float(sma_data['SMA'].iloc[-1]),
                'ADX': float(adx_data['ADX'].iloc[-1])
            }

            logger.info(f"Successfully processed indicators: {indicators}")
            return {
                'status': 'success',
                'data': indicators
            }

        except Exception as e:
            logger.error(f"Error getting technical indicators: {e}", exc_info=True)
            return {
                'status': 'error',
                'message': str(e)
            }