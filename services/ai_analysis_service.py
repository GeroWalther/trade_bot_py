from anthropic import Anthropic
from typing import Dict, List
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import re
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.timeseries import TimeSeries
import os

logger = logging.getLogger(__name__)

class AIAnalysisService:
    def __init__(self, api_key: str):
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        logger.info(f"Initializing AI Analysis Service. Anthropic API key present: {bool(anthropic_key)}")
        # Mask the key for security but show length
        if anthropic_key:
            logger.debug(f"API key length: {len(anthropic_key)}")
        
        self.anthropic = Anthropic(api_key=anthropic_key)
        self.alpha_vantage_key = api_key
        self.ti = TechIndicators(key=self.alpha_vantage_key)
        self.ts = TimeSeries(key=self.alpha_vantage_key)
        
    def _format_news_sentiment(self, news_data: List[Dict]) -> str:
        """Format news sentiment for AI analysis"""
        if not news_data:
            return "No recent news available"
            
        sentiments = []
        for news in news_data[:5]:  # Top 5 most relevant news
            sentiment = "positive" if news['sentiment'] > 0.1 else "negative" if news['sentiment'] < -0.1 else "neutral"
            sentiments.append(f"- {news['title']} (Sentiment: {sentiment})")
        
        return "\n".join(sentiments)

    async def generate_market_analysis(self, 
                                     asset: str,
                                     market_data: Dict,
                                     economic_data: Dict,
                                     news_data: List[Dict],
                                     sentiment_score: float) -> Dict:
        """Generate AI-powered market analysis"""
        try:
            logger.info(f"Starting AI analysis for {asset}")
            
            anthropic_key = os.getenv('ANTHROPIC_API_KEY')
            if not anthropic_key:
                logger.error("Anthropic API key not found")
                raise ValueError("ANTHROPIC_API_KEY is required")
            
            # Verify key format
            if not anthropic_key.startswith('sk-ant-'):
                logger.error("Invalid Anthropic API key format")
                raise ValueError("Invalid Anthropic API key format. Should start with 'sk-ant-'")
            
            # Prepare context for AI
            context = self._prepare_analysis_context(
                asset, market_data, economic_data, news_data, sentiment_score
            )
            
            try:
                # Get AI analysis from Claude
                logger.info("Requesting analysis from Claude...")
                
                # Create message synchronously - Anthropic's Python client doesn't support async
                response = self.anthropic.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=4000,
                    temperature=0.1,
                    system="You are an expert forex and financial markets trader. Analyze the provided market data and generate detailed trading recommendations.",
                    messages=[{
                        "role": "user",
                        "content": context
                    }]
                )
                
                logger.info("Received response from Claude")
                
                # Extract just the text content from the response
                analysis_text = str(response.content)  # Convert TextBlock to string
                logger.debug(f"Claude response text: {analysis_text}")
                
                # Return structured analysis
                result = {
                    'summary': analysis_text,  # Use the converted string
                    'key_factors': [
                        "Technical Indicators Analysis",
                        "Economic Data Impact",
                        "Market Sentiment Analysis",
                        "Risk Assessment"
                    ],
                    'trading_strategy': {
                        'direction': 'BULLISH' if market_data['signals']['trend']['primary'] == 'BULLISH' else 'BEARISH',
                        'entry': {
                            'price': market_data['current_price'],
                            'rationale': 'Based on technical analysis and market sentiment'
                        },
                        'stopLoss': {
                            'price': market_data['current_price'] * 0.99 if market_data['signals']['trend']['primary'] == 'BULLISH' else market_data['current_price'] * 1.01,
                            'rationale': 'Based on volatility and risk management'
                        }
                    }
                }
                logger.info("Analysis completed successfully")
                return result
                
            except Exception as e:
                logger.error(f"Error calling Claude API: {str(e)}")
                if "401" in str(e):
                    raise ValueError("Invalid Anthropic API key or authentication failed")
                elif "429" in str(e):
                    raise ValueError("Rate limit exceeded")
                raise
                
        except Exception as e:
            logger.error(f"Error in AI analysis: {e}", exc_info=True)
            return {
                'summary': f'Analysis not available: {str(e)}',
                'key_factors': [],
                'trading_strategy': {
                    'direction': 'NEUTRAL',
                    'entry': {'price': 0, 'rationale': ''},
                    'stopLoss': {'price': 0, 'rationale': ''}
                }
            }

    def _prepare_analysis_context(self, asset: str, market_data: Dict, 
                                economic_data: Dict, news_data: List[Dict], 
                                sentiment_score: float) -> str:
        """Prepare context for AI analysis"""
        return f"""
        Please analyze {asset} and provide a detailed trading strategy.

        Technical Data:
        - Current Price: {market_data['current_price']}
        - 24h Change: {market_data['change_percent']:.2f}%
        - RSI: {market_data['indicators'].get('rsi', 'N/A')}
        - SMA20: {market_data['indicators'].get('sma_20', 'N/A')}
        - SMA50: {market_data['indicators'].get('sma_50', 'N/A')}
        - Current Trend: {market_data['signals']['trend']['primary']}
        - Trend Strength: {market_data['signals']['trend']['strength']}

        Economic Indicators:
        {self._format_economic_data(economic_data)}

        Market Sentiment:
        - Overall Score: {sentiment_score:.2f}
        - Recent News Sentiment: {self._format_news_sentiment(news_data)}

        Please provide:
        1. Market Summary: Current situation and key drivers
        2. Pattern Recognition: Identify any chart patterns
        3. Risk Analysis: Key risks and volatility assessment
        4. Trading Strategy:
           - Direction (Long/Short)
           - Entry points with rationale
           - Stop loss levels
           - Take profit targets
           - Position sizing recommendation
        5. Probability Analysis:
           - Success probability
           - Risk/reward ratio
           - Confidence level
        """

    async def _get_technical_indicators(self, symbol: str) -> Dict:
        """Get technical indicators from Alpha Vantage"""
        try:
            # Get RSI
            rsi_data, _ = self.ti.get_rsi(symbol=symbol, interval='60min')
            rsi = float(list(rsi_data.values())[0]['RSI'])
            
            # Get MACD
            macd_data, _ = self.ti.get_macd(symbol=symbol, interval='60min')
            latest_macd = list(macd_data.values())[0]
            
            # Get Bollinger Bands
            bbands_data, _ = self.ti.get_bbands(symbol=symbol, interval='60min')
            latest_bbands = list(bbands_data.values())[0]
            
            # Get SMA
            sma20_data, _ = self.ti.get_sma(symbol=symbol, interval='60min', time_period=20)
            sma50_data, _ = self.ti.get_sma(symbol=symbol, interval='60min', time_period=50)
            
            return {
                'rsi': rsi,
                'macd': {
                    'macd': float(latest_macd['MACD']),
                    'signal': float(latest_macd['MACD_Signal']),
                    'hist': float(latest_macd['MACD_Hist'])
                },
                'bbands': {
                    'upper': float(latest_bbands['Real Upper Band']),
                    'middle': float(latest_bbands['Real Middle Band']),
                    'lower': float(latest_bbands['Real Lower Band'])
                },
                'sma': {
                    'sma20': float(list(sma20_data.values())[0]['SMA']),
                    'sma50': float(list(sma50_data.values())[0]['SMA'])
                }
            }
        except Exception as e:
            logger.error(f"Error getting technical indicators: {e}")
            return {}

    def _identify_patterns(self, market_data: Dict) -> Dict:
        """Identify technical patterns using Alpha Vantage data"""
        try:
            price = market_data['current_price']
            bbands = market_data.get('bbands', {})
            macd = market_data.get('macd', {})
            rsi = market_data.get('rsi', 50)
            
            patterns = {
                'type': [],
                'confidence': 0,
                'implications': ''
            }
            
            # Check for oversold/overbought
            if rsi > 70:
                patterns['type'].append('Overbought')
                patterns['implications'] = 'Potential reversal to downside'
            elif rsi < 30:
                patterns['type'].append('Oversold')
                patterns['implications'] = 'Potential reversal to upside'
                
            # Check for Bollinger Band squeeze
            if bbands:
                band_width = (bbands['upper'] - bbands['lower']) / bbands['middle']
                if band_width < 0.1:  # Tight bands
                    patterns['type'].append('Bollinger Squeeze')
                    patterns['implications'] = 'Potential breakout incoming'
                    
            # Check MACD crossover
            if macd:
                if macd['macd'] > macd['signal'] and macd['hist'] > 0:
                    patterns['type'].append('MACD Bullish Crossover')
                    patterns['implications'] = 'Bullish momentum building'
                elif macd['macd'] < macd['signal'] and macd['hist'] < 0:
                    patterns['type'].append('MACD Bearish Crossover')
                    patterns['implications'] = 'Bearish momentum building'
            
            patterns['confidence'] = len(patterns['type']) * 20  # 20% per pattern
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error in pattern recognition: {e}")
            return {'type': [], 'confidence': 0, 'implications': ''}

    def _analyze_risk(self, market_data: Dict, patterns: Dict) -> Dict:
        """Analyze trading risks"""
        return {
            'volatility_level': 'HIGH' if market_data['change_percent'] > 1 else 'MEDIUM',
            'key_risks': [
                'High volatility environment',
                'Major economic releases pending',
                'Pattern failure probability'
            ],
            'risk_reward_ratio': 2.5,  # Example
            'position_size_recommendation': 'Reduce size due to high volatility'
        }

    def _get_default_analysis(self) -> Dict:
        """Return default analysis structure"""
        return {
            'summary': 'Analysis not available',
            'patterns': [],
            'risk_analysis': {
                'volatility_level': 'UNKNOWN',
                'key_risks': [],
                'risk_reward_ratio': 0,
                'position_size_recommendation': 'No recommendation'
            },
            'trading_strategy': {
                'direction': 'NEUTRAL',
                'entry': {'price': 0, 'rationale': ''},
                'stopLoss': {'price': 0, 'rationale': ''},
                'targets': []
            }
        }

    def _parse_ai_response(self, response: str) -> Dict:
        """Parse and structure AI response"""
        try:
            # Here you would implement logic to parse Claude's response
            # For now, using a simple structure
            sections = self._extract_sections(response)
            
            return {
                'summary': sections.get('Market Summary', ''),
                'short_term_forecast': sections.get('Trading Strategy', {}),
                'mid_term_forecast': sections.get('Technical Analysis', {}),
                'long_term_forecast': sections.get('Economic Impact Analysis', {}),
                'probabilities': {
                    'bullish': float(sections.get('probability_bullish', 50)),
                    'bearish': float(sections.get('probability_bearish', 50)),
                    'confidence': float(sections.get('confidence', 70))
                },
                'risk_assessment': sections.get('Risk Assessment', {}),
                'trading_strategy': sections.get('Trading Strategy', {}),
                'key_factors': sections.get('Key Factors to Monitor', [])
            }
        except Exception as e:
            logger.error(f"Error parsing AI response: {e}")
            return self._get_default_analysis()

    def _format_economic_data(self, economic_data: Dict) -> str:
        """Format economic data for AI analysis"""
        formatted = []
        for indicator, data in economic_data.items():
            formatted.append(f"- {indicator}: {data['value']} ({data['trend']})")
        return "\n".join(formatted)

    def _format_news_data(self, news_data: List[Dict]) -> str:
        """Format news data for AI analysis"""
        formatted = []
        for news in news_data[:5]:  # Top 5 most relevant news
            formatted.append(f"- {news['title']} (Sentiment: {news['sentiment']:.2f})")
        return "\n".join(formatted)

    def _extract_sections(self, response: str) -> Dict:
        """Extract and structure sections from Claude's response"""
        sections = {}
        current_section = None
        current_content = []
        
        for line in response.split('\n'):
            # Check for main section headers
            if line.strip().startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.')):
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = line.split('.', 1)[1].split(':', 1)[0].strip()
                current_content = []
            else:
                current_content.append(line)
        
        # Add the last section
        if current_section:
            sections[current_section] = '\n'.join(current_content).strip()
        
        # Extract probability numbers
        if 'Probability Analysis' in sections:
            prob_text = sections['Probability Analysis']
            sections['probability_bullish'] = self._extract_probability(prob_text, 'bullish')
            sections['probability_bearish'] = self._extract_probability(prob_text, 'bearish')
            sections['confidence'] = self._extract_confidence(prob_text)
        
        return sections

    def _extract_probability(self, text: str, direction: str) -> float:
        """Extract probability numbers from text"""
        try:
            # Use regex to find percentage numbers
            matches = re.findall(rf"{direction}.*?(\d+)%", text.lower())
            return float(matches[0]) if matches else 50.0
        except Exception:
            return 50.0

    def _extract_confidence(self, text: str) -> float:
        """Extract confidence level from text"""
        try:
            matches = re.findall(r"confidence.*?(\d+)%", text.lower())
            return float(matches[0]) if matches else 70.0
        except Exception:
            return 70.0 