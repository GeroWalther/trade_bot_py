import logging
import asyncio
from typing import Dict, Optional
from datetime import datetime
import os
import json
import re
from openai import OpenAI
import random

logger = logging.getLogger(__name__)

class SimpleAIService:
    """
    Ultra-Simple AI Service - Let OpenAI do EVERYTHING
    
    This service:
    1. Takes an asset name
    2. Sends it to OpenAI with instructions to research and analyze
    3. Gets back a complete trading strategy
    4. That's it - no local calculations at all!
    """
    
    def __init__(self):
        # Initialize OpenAI client
        self.openai_key = os.getenv('OPENAI_API_KEY')
        self.demo_mode = not self.openai_key
        
        if self.demo_mode:
            logger.warning("⚠️ OPENAI_API_KEY not found - running in DEMO MODE")
            logger.warning("Set OPENAI_API_KEY environment variable for real AI analysis")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.openai_key)
            logger.info("✅ Simple AI Service initialized with OpenAI")

    async def get_comprehensive_analysis(self, asset: str, timeframe: str = 'Day trade') -> Dict:
        """
        Get comprehensive AI analysis for an asset
        AI does ALL the research and analysis itself
        """
        try:
            logger.info(f"🤖 Sending {asset} to AI for complete analysis...")
            
            # Let AI do everything - research, analyze, decide
            ai_analysis = await self._get_ai_trading_analysis(asset, timeframe)
            
            logger.info(f"✅ AI analysis complete for {asset}")
            return ai_analysis
            
        except Exception as e:
            logger.error(f"Error in AI analysis for {asset}: {e}", exc_info=True)
            return self._get_fallback_analysis(asset)

    async def _get_ai_trading_analysis(self, asset: str, timeframe: str) -> Dict:
        """Let OpenAI do all the heavy lifting - research and analysis"""
        
        # Force real AI analysis - no demo mode
        if self.demo_mode:
            logger.error(f"❌ OPENAI_API_KEY missing - cannot analyze {asset}")
            return self._get_fallback_analysis(asset)
        
        try:
            # Randomize market bias to prevent consistent direction bias
            market_scenarios = [
                "Markets are showing mixed signals - analyze both bullish and bearish possibilities",
                "Consider potential market reversals and counter-trend opportunities", 
                "Look for oversold/overbought conditions that might favor contrarian positions",
                "Analyze if recent momentum is sustainable or due for reversal",
                "Consider both breakout and reversal scenarios equally"
            ]
            random_scenario = random.choice(market_scenarios)
            
            # Advanced AI prompt with trading term awareness
            prompt = f"""
You are an expert institutional trader analyzing {asset} for INTRADAY opportunities. Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

TRADING TERM: Intraday (positions close same day)
MARKET CONTEXT: Research current {asset} price action, news, economic factors, technical levels

{random_scenario}

CRITICAL: Respond with ONLY valid JSON. No explanations outside JSON.

ANALYSIS REQUIREMENTS:
1. Current market sentiment and momentum for {asset}
2. Key support/resistance levels based on recent price action  
3. Economic/news factors affecting {asset} today
4. Technical indicators and patterns
5. Risk/reward analysis with realistic targets
6. EQUAL consideration of LONG, SHORT, and NEUTRAL scenarios

STRATEGY REQUIREMENTS:
- For INTRADAY: Tight stops, quick profits, momentum-based entries
- Focus on current session volatility and key levels
- Consider market hours and liquidity
- Set realistic profit targets based on average daily range
- Use trailing stops for profitable positions
- Be willing to recommend SHORT positions when analysis suggests bearish conditions
- Consider NEUTRAL when market conditions are unclear or choppy

JSON FORMAT EXAMPLE (analyze REAL market conditions):

{{
    "trading_strategy": {{
        "direction": "[ANALYZE_MARKET_TO_DETERMINE]",
        "confidence": [YOUR_ANALYSIS_CONFIDENCE],
        "entry": {{"price": [REAL_MARKET_PRICE], "rationale": "[WHY_THIS_ENTRY]"}},
        "take_profit_1": {{"price": [REALISTIC_TARGET], "rationale": "[WHY_THIS_TARGET]"}},
        "stop_loss": {{"price": [RISK_MANAGEMENT_LEVEL], "rationale": "[WHY_THIS_STOP]"}},
        "rationale": "[YOUR_COMPLETE_MARKET_ANALYSIS_AND_REASONING]"
    }},
    "market_summary": "[CURRENT_MARKET_CONDITIONS_FOR_{asset}]",
    "key_drivers": ["[REAL_FACTOR_1]", "[REAL_FACTOR_2]", "[REAL_FACTOR_3]"],
    "risk_assessment": "[YOUR_RISK_ANALYSIS]",
    "probability_up": [YOUR_PROBABILITY_ASSESSMENT],
    "confidence_level": [YOUR_CONFIDENCE_0_TO_100]
}}

CRITICAL REQUIREMENTS:
- direction: MUST be "LONG", "SHORT", or "NEUTRAL" based on YOUR REAL ANALYSIS
- DO NOT copy the example - analyze the actual {asset} market conditions
- Consider bearish scenarios - not everything is bullish!
- Look for both uptrends AND downtrends in {asset}
- Consider market cycles, resistance levels, and potential reversals
- Give equal consideration to SHORT opportunities
- Base decisions on REAL technical and fundamental analysis

RESPOND WITH ONLY THE JSON."""

            # Send to OpenAI with better parameters
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Using mini for faster/cheaper responses
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                max_tokens=800,
                temperature=0.1,  # Lower temperature for more consistent format
                response_format={"type": "json_object"}  # Force JSON response
            )
            
            # Parse the response
            response_text = response.choices[0].message.content
            logger.info(f"🔍 Raw AI response for {asset}: {response_text}")
            
            # Try to parse as JSON directly
            try:
                analysis = json.loads(response_text)
                logger.info(f"📋 Parsed JSON for {asset}: {analysis}")
                
                # Validate the structure
                if self._validate_analysis_structure(analysis):
                    direction = analysis['trading_strategy'].get('direction', '').upper().strip()
                    logger.info(f"✅ Real AI analysis successful for {asset} - Direction: {direction}")
                    return analysis
                else:
                    logger.error(f"❌ AI response structure invalid for {asset}: {analysis}")
                    return self._get_fallback_analysis(asset)
                    
            except json.JSONDecodeError as e:
                logger.error(f"❌ JSON decode error for {asset}: {e}")
                logger.error(f"Raw response: {response_text}")
                return self._get_fallback_analysis(asset)
                
        except Exception as e:
            logger.error(f"❌ Error getting OpenAI analysis for {asset}: {e}", exc_info=True)
            return self._get_fallback_analysis(asset)

    # Demo analysis method removed - forcing real AI analysis only

    def _validate_analysis_structure(self, analysis: Dict) -> bool:
        """Validate that AI analysis has required structure"""
        try:
            required_fields = [
                'trading_strategy',
                'market_summary',
                'key_drivers',
                'risk_assessment'
            ]
            
            if not all(field in analysis for field in required_fields):
                logger.warning(f"Missing top-level fields: {[f for f in required_fields if f not in analysis]}")
                return False
            
            # Check trading strategy structure
            strategy = analysis['trading_strategy']
            strategy_fields = ['direction', 'entry', 'take_profit_1', 'stop_loss', 'rationale']
            
            if not all(field in strategy for field in strategy_fields):
                logger.warning(f"Missing strategy fields: {[f for f in strategy_fields if f not in strategy]}")
                return False
            
            # Check that prices are numbers
            try:
                entry_price = strategy['entry'].get('price', 0)
                tp_price = strategy['take_profit_1'].get('price', 0)
                sl_price = strategy['stop_loss'].get('price', 0)
                
                if not all(isinstance(price, (int, float)) for price in [entry_price, tp_price, sl_price]):
                    logger.warning("Prices are not numbers")
                    return False
            except Exception as e:
                logger.warning(f"Error validating prices: {e}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating analysis structure: {e}")
            return False

    def _get_fallback_analysis(self, asset: str) -> Dict:
        """Fallback analysis when AI fails - returns NEUTRAL with 0 score"""
        logger.error(f"❌ FALLBACK: AI analysis failed for {asset}")
        return {
            'trading_strategy': {
                'direction': 'NEUTRAL',
                'confidence': 0,
                'entry': {'price': 0, 'rationale': 'AI analysis failed'},
                'take_profit_1': {'price': 0, 'rationale': 'No target set'},
                'stop_loss': {'price': 0, 'rationale': 'No stop loss set'},
                'rationale': 'AI analysis failed - manual intervention required'
            },
            'market_summary': f'{asset} AI analysis failed',
            'key_drivers': ['AI service failure'],
            'risk_assessment': 'Cannot assess - AI analysis unavailable',
            'probability_up': 50,
            'confidence_level': 0
        } 