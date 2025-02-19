import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import talib
import joblib
import os

@dataclass
class MarketSignal:
    symbol: str
    timestamp: str
    probability_up: float
    confidence: float
    indicators: Dict
    signals: Dict
    recommendation: str
    supporting_factors: List[str]

class TechnicalAnalyzer:
    """Technical Analysis Component"""
    
    def __init__(self):
        self.required_periods = 200  # For long-term indicators

    def analyze(self, df: pd.DataFrame) -> Dict:
        """Calculate all technical indicators"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        return {
            # Trend Indicators
            'sma_20': talib.SMA(close, timeperiod=20)[-1],
            'sma_50': talib.SMA(close, timeperiod=50)[-1],
            'sma_200': talib.SMA(close, timeperiod=200)[-1],
            'ema_20': talib.EMA(close, timeperiod=20)[-1],
            'ema_50': talib.EMA(close, timeperiod=50)[-1],
            
            # Momentum Indicators
            'rsi': talib.RSI(close)[-1],
            'macd': talib.MACD(close)[0][-1],
            'macd_signal': talib.MACD(close)[1][-1],
            'macd_hist': talib.MACD(close)[2][-1],
            'stoch_k': talib.STOCH(high, low, close)[0][-1],
            'stoch_d': talib.STOCH(high, low, close)[1][-1],
            'cci': talib.CCI(high, low, close)[-1],
            'adx': talib.ADX(high, low, close)[-1],
            
            # Volatility Indicators
            'atr': talib.ATR(high, low, close)[-1],
            'bbands_upper': talib.BBANDS(close)[0][-1],
            'bbands_middle': talib.BBANDS(close)[1][-1],
            'bbands_lower': talib.BBANDS(close)[2][-1],
            
            # Volume Indicators
            'obv': talib.OBV(close, volume)[-1],
            'mfi': talib.MFI(high, low, close, volume)[-1],
            
            # Additional Indicators
            'ichimoku': self._calculate_ichimoku(df),
            'support_resistance': self._find_support_resistance(df),
            'pivot_points': self._calculate_pivot_points(df)
        }

    def _calculate_ichimoku(self, df: pd.DataFrame) -> Dict:
        """Calculate Ichimoku Cloud indicators"""
        high = df['high']
        low = df['low']
        
        # Tenkan-sen (Conversion Line)
        tenkan = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2
        
        # Kijun-sen (Base Line)
        kijun = (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan + kijun) / 2).shift(26)
        
        # Senkou Span B (Leading Span B)
        senkou_span_b = ((high.rolling(window=52).max() + low.rolling(window=52).min()) / 2).shift(26)
        
        return {
            'tenkan': tenkan.iloc[-1],
            'kijun': kijun.iloc[-1],
            'senkou_a': senkou_span_a.iloc[-1],
            'senkou_b': senkou_span_b.iloc[-1]
        }

class MarketPredictor:
    """ML-based Market Prediction Component"""
    
    def __init__(self, model_path: str = 'models'):
        self.model_path = model_path
        self.models = {
            'rf': RandomForestClassifier(n_estimators=200, max_depth=10),
            'gb': GradientBoostingClassifier(n_estimators=200)
        }
        self.scaler = StandardScaler()
        self._load_or_train_models()

    def _load_or_train_models(self):
        """Load existing models or train new ones"""
        try:
            for name in self.models.keys():
                model_file = os.path.join(self.model_path, f'{name}_model.joblib')
                if os.path.exists(model_file):
                    self.models[name] = joblib.load(model_file)
        except Exception as e:
            print(f"Error loading models: {e}. Will train new ones.")

    def prepare_features(self, technical_data: Dict, market_data: Dict) -> np.ndarray:
        """Prepare feature vector for prediction"""
        features = []
        
        # Technical indicators
        for indicator in ['rsi', 'macd', 'adx', 'mfi']:
            features.append(technical_data.get(indicator, 0))
        
        # Market conditions
        features.extend([
            market_data.get('volatility', 0),
            market_data.get('trend_strength', 0),
            market_data.get('volume_trend', 0)
        ])
        
        return self.scaler.transform(np.array(features).reshape(1, -1))

    def predict(self, features: np.ndarray) -> Tuple[float, float]:
        """Get ensemble prediction and confidence"""
        predictions = []
        for model in self.models.values():
            prob = model.predict_proba(features)[0][1]
            predictions.append(prob)
        
        avg_prob = np.mean(predictions)
        confidence = 1 - np.std(predictions)  # Higher agreement = higher confidence
        
        return avg_prob, confidence

class MarketAnalyzer:
    """Main Market Analysis Component"""
    
    def __init__(self):
        self.technical = TechnicalAnalyzer()
        self.predictor = MarketPredictor()
        
    async def analyze_symbol(self, df: pd.DataFrame, symbol: str, market_data: Dict) -> MarketSignal:
        """Perform comprehensive market analysis"""
        # Get technical indicators
        indicators = self.technical.analyze(df)
        
        # Prepare features for ML
        features = self.predictor.prepare_features(indicators, market_data)
        
        # Get prediction and confidence
        probability, confidence = self.predictor.predict(features)
        
        # Generate trading signals
        signals = self._generate_signals(indicators)
        
        # Generate recommendation
        recommendation, factors = self._generate_recommendation(
            indicators, signals, probability, confidence
        )
        
        return MarketSignal(
            symbol=symbol,
            timestamp=str(pd.Timestamp.now()),
            probability_up=probability * 100,
            confidence=confidence * 100,
            indicators=indicators,
            signals=signals,
            recommendation=recommendation,
            supporting_factors=factors
        )
    
    def _generate_signals(self, indicators: Dict) -> Dict:
        """Generate trading signals from indicators"""
        return {
            'trend': self._analyze_trend(indicators),
            'momentum': self._analyze_momentum(indicators),
            'volatility': self._analyze_volatility(indicators),
            'volume': self._analyze_volume(indicators)
        }
    
    def _analyze_trend(self, indicators: Dict) -> Dict:
        """Analyze price trend"""
        return {
            'primary': 'BULLISH' if indicators['sma_20'] > indicators['sma_50'] else 'BEARISH',
            'secondary': 'BULLISH' if indicators['ema_20'] > indicators['ema_50'] else 'BEARISH',
            'strength': 'STRONG' if abs(indicators['adx']) > 25 else 'MODERATE',
            'support': indicators['support_resistance']['support'],
            'resistance': indicators['support_resistance']['resistance']
        }

    def _analyze_momentum(self, indicators: Dict) -> Dict:
        """Analyze price momentum"""
        return {
            'rsi': {
                'value': indicators['rsi'],
                'signal': 'OVERSOLD' if indicators['rsi'] < 30 else 'OVERBOUGHT' if indicators['rsi'] > 70 else 'NEUTRAL'
            },
            'macd': {
                'value': indicators['macd'],
                'signal': 'BULLISH' if indicators['macd'] > indicators['macd_signal'] else 'BEARISH',
                'strength': abs(indicators['macd_hist'])
            },
            'stochastic': {
                'k': indicators['stoch_k'],
                'd': indicators['stoch_d'],
                'signal': 'BULLISH' if indicators['stoch_k'] > indicators['stoch_d'] else 'BEARISH'
            }
        }

    def _analyze_volatility(self, indicators: Dict) -> Dict:
        """Analyze market volatility"""
        bb_width = (indicators['bbands_upper'] - indicators['bbands_lower']) / indicators['bbands_middle']
        return {
            'bollinger': {
                'width': bb_width,
                'position': self._get_bb_position(indicators),
                'squeeze': bb_width < 0.1
            },
            'atr': {
                'value': indicators['atr'],
                'signal': 'HIGH' if indicators['atr'] > indicators['atr'] * 1.5 else 'LOW'
            }
        }

    def _analyze_volume(self, indicators: Dict) -> Dict:
        """Analyze volume trends"""
        return {
            'obv': {
                'trend': 'UP' if indicators['obv'] > 0 else 'DOWN',
                'strength': abs(indicators['obv'])
            },
            'mfi': {
                'value': indicators['mfi'],
                'signal': 'OVERSOLD' if indicators['mfi'] < 20 else 'OVERBOUGHT' if indicators['mfi'] > 80 else 'NEUTRAL'
            }
        }

    def _get_bb_position(self, indicators: Dict) -> str:
        """Get price position relative to Bollinger Bands"""
        current_price = indicators['bbands_middle']
        if current_price > indicators['bbands_upper']:
            return 'ABOVE'
        elif current_price < indicators['bbands_lower']:
            return 'BELOW'
        return 'INSIDE'

    def _generate_recommendation(self, indicators: Dict, signals: Dict, 
                               probability: float, confidence: float) -> Tuple[str, List[str]]:
        """Generate trading recommendation with supporting factors"""
        factors = []
        
        # Analyze trend signals
        if signals['trend']['primary'] == signals['trend']['secondary']:
            trend_strength = 'strong' if signals['trend']['strength'] == 'STRONG' else 'moderate'
            factors.append(f"{trend_strength} {signals['trend']['primary'].lower()} trend")

        # Check momentum
        if signals['momentum']['rsi']['signal'] != 'NEUTRAL':
            factors.append(f"RSI indicates {signals['momentum']['rsi']['signal'].lower()}")
        
        if signals['momentum']['macd']['signal'] == 'BULLISH':
            factors.append("MACD shows bullish momentum")
        elif signals['momentum']['macd']['signal'] == 'BEARISH':
            factors.append("MACD shows bearish momentum")

        # Check volatility
        if signals['volatility']['bollinger']['squeeze']:
            factors.append("Bollinger Band squeeze indicates potential breakout")
        
        # Make recommendation
        if probability > 70 and confidence > 0.7:
            recommendation = 'STRONG_BUY'
            factors.append(f"High probability ({probability:.1f}%) with strong confidence")
        elif probability > 60:
            recommendation = 'BUY'
            factors.append(f"Good probability ({probability:.1f}%) for upward movement")
        elif probability < 30 and confidence > 0.7:
            recommendation = 'STRONG_SELL'
            factors.append(f"Low probability ({probability:.1f}%) with strong confidence")
        elif probability < 40:
            recommendation = 'SELL'
            factors.append(f"Low probability ({probability:.1f}%) for upward movement")
        else:
            recommendation = 'HOLD'
            factors.append(f"Mixed signals with {probability:.1f}% probability")

        return recommendation, factors 