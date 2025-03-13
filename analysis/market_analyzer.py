import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# import talib  # Commented out ta-lib import
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
            'sma_20': self._sma(close, 20)[-1],
            'sma_50': self._sma(close, 50)[-1],
            'sma_200': self._sma(close, 200)[-1],
            'ema_20': self._ema(close, 20)[-1],
            'ema_50': self._ema(close, 50)[-1],
            
            # Momentum Indicators
            'rsi': self._rsi(close)[-1],
            'macd': self._macd(close)[0][-1],
            'macd_signal': self._macd(close)[1][-1],
            'macd_hist': self._macd(close)[2][-1],
            'stoch_k': self._stochastic(high, low, close)[0][-1],
            'stoch_d': self._stochastic(high, low, close)[1][-1],
            'cci': self._cci(high, low, close)[-1],
            'adx': self._adx(high, low, close)[-1],
            
            # Volatility Indicators
            'atr': self._atr(high, low, close)[-1],
            'bbands_upper': self._bbands(close)[0][-1],
            'bbands_middle': self._bbands(close)[1][-1],
            'bbands_lower': self._bbands(close)[2][-1],
            
            # Volume Indicators
            'obv': self._obv(close, volume)[-1],
            'mfi': self._mfi(high, low, close, volume)[-1],
            
            # Additional Indicators
            'ichimoku': self._calculate_ichimoku(df),
            'support_resistance': self._find_support_resistance(df),
            'pivot_points': self._calculate_pivot_points(df)
        }

    # Alternative implementations for technical indicators
    def _sma(self, data, period):
        """Simple Moving Average"""
        return pd.Series(data).rolling(window=period).mean().values
        
    def _ema(self, data, period):
        """Exponential Moving Average"""
        return pd.Series(data).ewm(span=period, adjust=False).mean().values
        
    def _rsi(self, data, period=14):
        """Relative Strength Index"""
        delta = pd.Series(data).diff().dropna()
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        roll_up = up.rolling(period).mean()
        roll_down = down.abs().rolling(period).mean()
        rs = roll_up / roll_down
        rsi = 100.0 - (100.0 / (1.0 + rs))
        # Pad the beginning with NaN values to match the original length
        padding = np.array([np.nan] * (len(data) - len(rsi)))
        return np.concatenate([padding, rsi.values])
        
    def _macd(self, data, fast_period=12, slow_period=26, signal_period=9):
        """Moving Average Convergence Divergence"""
        ema_fast = pd.Series(data).ewm(span=fast_period, adjust=False).mean()
        ema_slow = pd.Series(data).ewm(span=slow_period, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line.values, signal_line.values, histogram.values
        
    def _stochastic(self, high, low, close, k_period=14, d_period=3):
        """Stochastic Oscillator"""
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        close_series = pd.Series(close)
        
        # Calculate %K
        lowest_low = low_series.rolling(window=k_period).min()
        highest_high = high_series.rolling(window=k_period).max()
        k = 100 * ((close_series - lowest_low) / (highest_high - lowest_low))
        
        # Calculate %D
        d = k.rolling(window=d_period).mean()
        
        return k.values, d.values
        
    def _cci(self, high, low, close, period=14):
        """Commodity Channel Index"""
        tp = (high + low + close) / 3
        tp_series = pd.Series(tp)
        sma_tp = tp_series.rolling(window=period).mean()
        mad = tp_series.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        cci = (tp_series - sma_tp) / (0.015 * mad)
        return cci.values
        
    def _adx(self, high, low, close, period=14):
        """Average Directional Index"""
        # This is a simplified implementation
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        close_series = pd.Series(close)
        
        # Calculate +DI and -DI
        plus_dm = high_series.diff()
        minus_dm = low_series.diff(-1).abs()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = pd.DataFrame({
            'hl': high_series - low_series,
            'hc': (high_series - close_series.shift()).abs(),
            'lc': (low_series - close_series.shift()).abs()
        }).max(axis=1)
        
        atr = tr.rolling(window=period).mean()
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
        adx = dx.rolling(window=period).mean()
        
        return adx.values
        
    def _atr(self, high, low, close, period=14):
        """Average True Range"""
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        close_series = pd.Series(close)
        
        tr1 = high_series - low_series
        tr2 = (high_series - close_series.shift()).abs()
        tr3 = (low_series - close_series.shift()).abs()
        
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr.values
        
    def _bbands(self, data, period=20, std_dev=2):
        """Bollinger Bands"""
        series = pd.Series(data)
        sma = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band.values, sma.values, lower_band.values
        
    def _obv(self, close, volume):
        """On-Balance Volume"""
        close_series = pd.Series(close)
        volume_series = pd.Series(volume)
        
        obv = np.zeros_like(close)
        obv[0] = volume[0]
        
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
                
        return obv
        
    def _mfi(self, high, low, close, volume, period=14):
        """Money Flow Index"""
        typical_price = (high + low + close) / 3
        tp_series = pd.Series(typical_price)
        vol_series = pd.Series(volume)
        
        money_flow = tp_series * vol_series
        
        # Get positive and negative money flow
        diff = tp_series.diff()
        positive_flow = pd.Series(np.where(diff > 0, money_flow, 0))
        negative_flow = pd.Series(np.where(diff < 0, money_flow, 0))
        
        # Calculate money flow ratio
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        # Calculate MFI
        money_ratio = positive_mf / negative_mf
        mfi = 100 - (100 / (1 + money_ratio))
        
        return mfi.values

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
        
    def _find_support_resistance(self, df: pd.DataFrame) -> Dict:
        """Find support and resistance levels"""
        # Simple implementation - using recent highs and lows
        close = df['close'].iloc[-1]
        high = df['high']
        low = df['low']
        
        # Find recent support (lowest low in last 20 periods)
        support = low.rolling(window=20).min().iloc[-1]
        
        # Find recent resistance (highest high in last 20 periods)
        resistance = high.rolling(window=20).max().iloc[-1]
        
        return {
            'support': support,
            'resistance': resistance
        }
        
    def _calculate_pivot_points(self, df: pd.DataFrame) -> Dict:
        """Calculate pivot points"""
        # Use the most recent complete day
        if len(df) < 1:
            return {'pivot': 0, 'r1': 0, 's1': 0, 'r2': 0, 's2': 0}
            
        high = df['high'].iloc[-1]
        low = df['low'].iloc[-1]
        close = df['close'].iloc[-1]
        
        pivot = (high + low + close) / 3
        r1 = (2 * pivot) - low
        s1 = (2 * pivot) - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        
        return {
            'pivot': pivot,
            'r1': r1,
            's1': s1,
            'r2': r2,
            's2': s2
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