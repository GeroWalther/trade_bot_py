-- Supabase Schema for Multi-User Trading Bot Platform
-- Run this in your Supabase SQL Editor

-- Enable Row Level Security
-- This ensures users can only access their own data

-- Users table (extends Supabase auth.users)
CREATE TABLE public.user_profiles (
    id UUID REFERENCES auth.users(id) PRIMARY KEY,
    email TEXT,
    full_name TEXT,
    avatar_url TEXT,
    subscription_tier TEXT DEFAULT 'free', -- free, pro, enterprise
    max_bots INTEGER DEFAULT 3, -- Free tier: 3 bots, Pro: unlimited
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Custom Trading Bots
CREATE TABLE public.custom_bots (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    code TEXT NOT NULL,
    instruments TEXT[], -- Array of trading instruments
    risk_level TEXT NOT NULL CHECK (risk_level IN ('low', 'medium', 'high')),
    execution_interval INTEGER NOT NULL, -- Minutes between executions
    trailing_stop_type TEXT DEFAULT 'none' CHECK (trailing_stop_type IN ('none', 'fixed_pips')),
    trailing_stop_pips INTEGER DEFAULT 20,
    status TEXT DEFAULT 'stopped' CHECK (status IN ('stopped', 'running', 'error', 'paused')),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_run TIMESTAMP WITH TIME ZONE,
    error_count INTEGER DEFAULT 0,
    last_error TEXT
);

-- Bot Performance Metrics
CREATE TABLE public.bot_performance (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    bot_id UUID REFERENCES public.custom_bots(id) ON DELETE CASCADE NOT NULL,
    user_id UUID REFERENCES auth.users(id) NOT NULL,
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    total_pnl DECIMAL(15,2) DEFAULT 0,
    largest_win DECIMAL(15,2) DEFAULT 0,
    largest_loss DECIMAL(15,2) DEFAULT 0,
    max_drawdown DECIMAL(15,2) DEFAULT 0,
    current_drawdown DECIMAL(15,2) DEFAULT 0,
    sharpe_ratio DECIMAL(8,4) DEFAULT 0,
    win_rate DECIMAL(5,2) DEFAULT 0,
    avg_trade_duration_hours DECIMAL(8,2) DEFAULT 0,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(bot_id)
);

-- Individual Trades History
CREATE TABLE public.bot_trades (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    bot_id UUID REFERENCES public.custom_bots(id) ON DELETE CASCADE NOT NULL,
    user_id UUID REFERENCES auth.users(id) NOT NULL,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('BUY', 'SELL')),
    entry_price DECIMAL(12,6) NOT NULL,
    exit_price DECIMAL(12,6),
    position_size DECIMAL(15,2) NOT NULL,
    entry_time TIMESTAMP WITH TIME ZONE NOT NULL,
    exit_time TIMESTAMP WITH TIME ZONE,
    pnl DECIMAL(15,2),
    reason TEXT, -- Entry/exit reason
    trade_status TEXT DEFAULT 'open' CHECK (trade_status IN ('open', 'closed', 'cancelled')),
    position_id TEXT, -- OANDA position ID
    fees DECIMAL(10,2) DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Active Positions (for tracking open positions)
CREATE TABLE public.bot_positions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    bot_id UUID REFERENCES public.custom_bots(id) ON DELETE CASCADE NOT NULL,
    user_id UUID REFERENCES auth.users(id) NOT NULL,
    symbol TEXT NOT NULL,
    position_id TEXT NOT NULL, -- OANDA position ID
    direction TEXT NOT NULL CHECK (direction IN ('BUY', 'SELL')),
    entry_price DECIMAL(12,6) NOT NULL,
    current_price DECIMAL(12,6),
    position_size DECIMAL(15,2) NOT NULL,
    unrealized_pnl DECIMAL(15,2),
    entry_time TIMESTAMP WITH TIME ZONE NOT NULL,
    stop_loss DECIMAL(12,6),
    take_profit DECIMAL(12,6),
    trailing_stop_distance DECIMAL(8,4),
    is_active BOOLEAN DEFAULT true,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(bot_id, symbol) -- One position per symbol per bot
);

-- Bot Execution Logs
CREATE TABLE public.bot_logs (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    bot_id UUID REFERENCES public.custom_bots(id) ON DELETE CASCADE NOT NULL,
    user_id UUID REFERENCES auth.users(id) NOT NULL,
    log_level TEXT NOT NULL CHECK (log_level IN ('INFO', 'WARNING', 'ERROR', 'DEBUG')),
    message TEXT NOT NULL,
    details JSONB, -- Additional structured data
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- User API Keys (encrypted)
CREATE TABLE public.user_api_keys (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) NOT NULL,
    broker_name TEXT NOT NULL, -- 'oanda', 'mt4', etc.
    api_key_encrypted TEXT, -- Encrypted API key
    account_id TEXT,
    environment TEXT DEFAULT 'practice' CHECK (environment IN ('practice', 'live')),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, broker_name)
);

-- Subscription Management
CREATE TABLE public.user_subscriptions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) NOT NULL,
    subscription_id TEXT, -- Stripe subscription ID
    tier TEXT NOT NULL CHECK (tier IN ('free', 'pro', 'enterprise')),
    status TEXT NOT NULL CHECK (status IN ('active', 'cancelled', 'past_due', 'incomplete')),
    current_period_start TIMESTAMP WITH TIME ZONE,
    current_period_end TIMESTAMP WITH TIME ZONE,
    cancel_at_period_end BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ========================================
-- ROW LEVEL SECURITY POLICIES
-- ========================================

-- Enable RLS on all tables
ALTER TABLE public.user_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.custom_bots ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.bot_performance ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.bot_trades ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.bot_positions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.bot_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.user_api_keys ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.user_subscriptions ENABLE ROW LEVEL SECURITY;

-- User Profiles: Users can only see/edit their own profile
CREATE POLICY "Users can view own profile" ON public.user_profiles
    FOR SELECT USING (auth.uid() = id);
    
CREATE POLICY "Users can update own profile" ON public.user_profiles
    FOR UPDATE USING (auth.uid() = id);

-- Custom Bots: Users can only access their own bots
CREATE POLICY "Users can view own bots" ON public.custom_bots
    FOR SELECT USING (auth.uid() = user_id);
    
CREATE POLICY "Users can insert own bots" ON public.custom_bots
    FOR INSERT WITH CHECK (auth.uid() = user_id);
    
CREATE POLICY "Users can update own bots" ON public.custom_bots
    FOR UPDATE USING (auth.uid() = user_id);
    
CREATE POLICY "Users can delete own bots" ON public.custom_bots
    FOR DELETE USING (auth.uid() = user_id);

-- Bot Performance: Users can only see their bots' performance
CREATE POLICY "Users can view own bot performance" ON public.bot_performance
    FOR ALL USING (auth.uid() = user_id);

-- Bot Trades: Users can only see their bots' trades
CREATE POLICY "Users can view own bot trades" ON public.bot_trades
    FOR ALL USING (auth.uid() = user_id);

-- Bot Positions: Users can only see their bots' positions
CREATE POLICY "Users can view own bot positions" ON public.bot_positions
    FOR ALL USING (auth.uid() = user_id);

-- Bot Logs: Users can only see their bots' logs
CREATE POLICY "Users can view own bot logs" ON public.bot_logs
    FOR ALL USING (auth.uid() = user_id);

-- API Keys: Users can only see their own API keys
CREATE POLICY "Users can manage own API keys" ON public.user_api_keys
    FOR ALL USING (auth.uid() = user_id);

-- Subscriptions: Users can only see their own subscription
CREATE POLICY "Users can view own subscription" ON public.user_subscriptions
    FOR ALL USING (auth.uid() = user_id);

-- ========================================
-- INDEXES FOR PERFORMANCE
-- ========================================

-- Custom Bots
CREATE INDEX idx_custom_bots_user_id ON public.custom_bots(user_id);
CREATE INDEX idx_custom_bots_status ON public.custom_bots(status);
CREATE INDEX idx_custom_bots_user_status ON public.custom_bots(user_id, status);

-- Bot Trades
CREATE INDEX idx_bot_trades_bot_id ON public.bot_trades(bot_id);
CREATE INDEX idx_bot_trades_user_id ON public.bot_trades(user_id);
CREATE INDEX idx_bot_trades_symbol ON public.bot_trades(symbol);
CREATE INDEX idx_bot_trades_entry_time ON public.bot_trades(entry_time);

-- Bot Positions
CREATE INDEX idx_bot_positions_bot_id ON public.bot_positions(bot_id);
CREATE INDEX idx_bot_positions_user_id ON public.bot_positions(user_id);
CREATE INDEX idx_bot_positions_active ON public.bot_positions(is_active);

-- Bot Logs
CREATE INDEX idx_bot_logs_bot_id ON public.bot_logs(bot_id);
CREATE INDEX idx_bot_logs_timestamp ON public.bot_logs(timestamp);
CREATE INDEX idx_bot_logs_user_id ON public.bot_logs(user_id);

-- ========================================
-- FUNCTIONS AND TRIGGERS
-- ========================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers to auto-update updated_at
CREATE TRIGGER update_user_profiles_updated_at 
    BEFORE UPDATE ON public.user_profiles 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_custom_bots_updated_at 
    BEFORE UPDATE ON public.custom_bots 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to automatically create user profile after signup
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.user_profiles (id, email, full_name)
    VALUES (
        NEW.id, 
        NEW.email, 
        COALESCE(NEW.raw_user_meta_data->>'full_name', '')
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Trigger to create profile for new users
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- Function to update bot performance when trades are added
CREATE OR REPLACE FUNCTION update_bot_performance()
RETURNS TRIGGER AS $$
BEGIN
    -- Update performance metrics when a trade is closed
    IF TG_OP = 'UPDATE' AND OLD.trade_status = 'open' AND NEW.trade_status = 'closed' THEN
        INSERT INTO public.bot_performance (bot_id, user_id)
        VALUES (NEW.bot_id, NEW.user_id)
        ON CONFLICT (bot_id) DO NOTHING;
        
        -- Update statistics
        WITH trade_stats AS (
            SELECT 
                COUNT(*) as total_trades,
                COUNT(CASE WHEN pnl > 0 THEN 1 END) as winning_trades,
                COUNT(CASE WHEN pnl < 0 THEN 1 END) as losing_trades,
                COALESCE(SUM(pnl), 0) as total_pnl,
                COALESCE(MAX(pnl), 0) as largest_win,
                COALESCE(MIN(pnl), 0) as largest_loss
            FROM public.bot_trades 
            WHERE bot_id = NEW.bot_id AND trade_status = 'closed'
        )
        UPDATE public.bot_performance bp
        SET 
            total_trades = ts.total_trades,
            winning_trades = ts.winning_trades,
            losing_trades = ts.losing_trades,
            total_pnl = ts.total_pnl,
            largest_win = ts.largest_win,
            largest_loss = ts.largest_loss,
            win_rate = CASE WHEN ts.total_trades > 0 
                           THEN (ts.winning_trades::DECIMAL / ts.total_trades) * 100 
                           ELSE 0 END,
            last_updated = NOW()
        FROM trade_stats ts
        WHERE bp.bot_id = NEW.bot_id;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Trigger to update performance on trade completion
CREATE TRIGGER trigger_update_bot_performance
    AFTER UPDATE ON public.bot_trades
    FOR EACH ROW EXECUTE FUNCTION update_bot_performance();

-- Bot Templates
CREATE TABLE public.bot_templates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL,
    description TEXT,
    language VARCHAR(20) NOT NULL CHECK (language IN ('javascript', 'python')),
    category VARCHAR(50) DEFAULT 'beginner' CHECK (category IN ('beginner', 'intermediate', 'advanced')),
    strategy_type VARCHAR(50) DEFAULT 'trend_following' CHECK (strategy_type IN ('trend_following', 'mean_reversion', 'breakout', 'scalping', 'arbitrage')),
    code TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE,
    usage_count INTEGER DEFAULT 0,
    author VARCHAR(100) DEFAULT 'System',
    tags TEXT[] DEFAULT ARRAY[]::TEXT[]
);

-- Enable RLS on bot templates
ALTER TABLE public.bot_templates ENABLE ROW LEVEL SECURITY;

-- Bot templates are readable by all authenticated users
CREATE POLICY "Templates are readable by authenticated users" ON public.bot_templates
    FOR SELECT USING (is_active = true);

-- Add index for template queries
CREATE INDEX idx_bot_templates_language ON public.bot_templates(language);
CREATE INDEX idx_bot_templates_category ON public.bot_templates(category);
CREATE INDEX idx_bot_templates_strategy ON public.bot_templates(strategy_type);
CREATE INDEX idx_bot_templates_active ON public.bot_templates(is_active);

-- Add trigger for updated_at
CREATE TRIGGER update_bot_templates_updated_at 
    BEFORE UPDATE ON public.bot_templates 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ========================================
-- INITIAL DATA
-- ========================================

-- Default subscription tiers
INSERT INTO public.user_subscriptions (id, user_id, tier, status) 
SELECT gen_random_uuid(), id, 'free', 'active' 
FROM auth.users 
ON CONFLICT DO NOTHING;

-- Seed with current templates
INSERT INTO bot_templates (name, description, language, category, strategy_type, code) VALUES
('JavaScript Tutorial Bot', '🎓 Complete Tutorial Bot - Learn to build any trading strategy! Includes 50MA example plus RSI, MACD, and Bollinger Bands examples. Step-by-step commented code teaches bot structure, signal generation, risk management, and advanced features.', 'javascript', 'beginner', 'trend_following', $$/**
 * 🎓 TRADING BOT DEVELOPMENT TUTORIAL (JavaScript)
 * 
 * 📖 LEARN BY EXAMPLE:
 * This template teaches you the fundamental building blocks of trading bots.
 * Follow the comments to understand each component and build your own strategies.
 * 
 * 🎯 WHAT YOU'LL LEARN:
 * ✅ How to structure a trading bot class
 * ✅ How to analyze market data and indicators  
 * ✅ How to generate buy/sell signals
 * ✅ How to manage risk and position sizing
 * ✅ How to implement entry/exit logic
 * ✅ How to add custom parameters and settings
 * 
 * 💡 EXAMPLE STRATEGY: 50MA Trend Following
 * - This is just ONE example - you can implement ANY strategy!
 * - RSI strategies, Bollinger Bands, MACD crossovers, etc.
 * - The structure remains the same, just change the logic
 */

class TradingBotTutorial {
  /**
   * 🏗️ STEP 1: CONSTRUCTOR - Initialize Your Bot
   * This runs once when your bot is created
   */
  constructor(config) {
    // 📝 Basic bot information
    this.name = config.name || "Learning Bot";
    this.instruments = config.instruments || ['EUR_USD'];
    this.riskLevel = config.riskLevel || 'medium';
    
    // 🔧 Strategy Parameters (CUSTOMIZE THESE!)
    // Change these values to create different strategies:
    this.maLength = 50;           // Try 20, 100, 200 for different timeframes
    this.profitTarget = 0.015;    // 1.5% profit target (try 0.01 for 1%, 0.02 for 2%)
    this.stopLoss = 0.0075;       // 0.75% stop loss (try 0.005 or 0.01)
    this.confidence = 0.8;        // 80% confidence (try 0.6-0.9)
    
    // 📊 You can add MORE parameters for different strategies:
    // this.rsiPeriod = 14;       // For RSI strategies
    // this.bbPeriod = 20;        // For Bollinger Bands
    // this.macdFast = 12;        // For MACD strategies
    
    console.log(`🎓 ${this.name} tutorial bot initialized!`);
    console.log(`📈 Strategy: ${this.maLength}-period Moving Average`);
  }

  /**
   * 🎯 STEP 2: ANALYZE FUNCTION - Your Trading Logic
   * This is called every execution interval to check for trading opportunities
   * 
   * CUSTOMIZE THIS SECTION TO CREATE YOUR OWN STRATEGIES!
   */
  async analyze(marketData) {
    const { symbol, price, indicators } = marketData;
    
    // 📊 STEP 2A: Get Your Indicators
    // The 'indicators' object contains pre-calculated technical indicators
    // Available indicators: sma_20, sma_50, sma_200, rsi_14, macd, etc.
    
    const ma50 = indicators.sma_50 || indicators.ma_50;
    
    if (!ma50) {
    return {
      action: 'HOLD',
        confidence: 0,
        reason: `❌ Missing 50MA data - need at least ${this.maLength} price points`
      };
    }
    
    // 🔍 STEP 2B: Implement Your Strategy Logic
    // This example uses 50MA, but you can implement ANY strategy here:
    
    // EXAMPLE 1: Moving Average Strategy (current example)
    if (price > ma50) {
      return this.generateBuySignal(price, ma50);
    } else if (price < ma50) {
      return this.generateSellSignal(price, ma50);
    }
    
    // EXAMPLE 2: RSI Strategy (uncomment to use instead)
    /*
    const rsi = indicators.rsi_14;
    if (rsi < 30) {
      return this.generateBuySignal(price, rsi, 'RSI Oversold');
    } else if (rsi > 70) {
      return this.generateSellSignal(price, rsi, 'RSI Overbought');
    }
    */
    
    // EXAMPLE 3: MACD Strategy (uncomment to use instead)
    /*
    const macd = indicators.macd;
    const macdSignal = indicators.macd_signal;
    if (macd > macdSignal) {
      return this.generateBuySignal(price, macd, 'MACD Bullish Crossover');
    } else if (macd < macdSignal) {
      return this.generateSellSignal(price, macd, 'MACD Bearish Crossover');
    }
    */
    
    return {
      action: 'HOLD',
      confidence: 0.3,
      reason: `⚖️ No clear signal - price near 50MA (${ma50.toFixed(5)})`
    };
  }

  /**
   * 🟢 STEP 3A: BUY Signal Generator
   * Customize this to change your buy conditions
   */
  generateBuySignal(price, indicator, customReason = null) {
    const percentAbove = ((price - indicator) / indicator) * 100;
    
    return {
      action: 'BUY',
      confidence: this.confidence,
      reason: customReason || `💹 BULLISH: Price ${price.toFixed(5)} is ${percentAbove.toFixed(2)}% above 50MA (${indicator.toFixed(5)})`,
      entryPrice: price,
      stopLoss: price * (1 - this.stopLoss),
      takeProfit: price * (1 + this.profitTarget),
      
      // 📝 Add custom data for your strategy
      strategyData: {
        indicator: indicator,
        percentAbove: percentAbove,
        timeframe: '50MA'
      }
    };
  }

  /**
   * 🔴 STEP 3B: SELL Signal Generator  
   * Customize this to change your sell conditions
   */
  generateSellSignal(price, indicator, customReason = null) {
    const percentBelow = ((indicator - price) / indicator) * 100;
    
    return {
      action: 'SELL',
      confidence: this.confidence,
      reason: customReason || `📉 BEARISH: Price ${price.toFixed(5)} is ${percentBelow.toFixed(2)}% below 50MA (${indicator.toFixed(5)})`,
      entryPrice: price,
      stopLoss: price * (1 + this.stopLoss),
      takeProfit: price * (1 - this.profitTarget),
      
      // 📝 Add custom data for your strategy
      strategyData: {
        indicator: indicator,
        percentBelow: percentBelow,
        timeframe: '50MA'
      }
    };
  }

  /**
   * 💰 STEP 4: POSITION SIZING (Optional - Advanced)
   * Calculate how much to trade based on risk
   */
  calculatePositionSize(accountBalance, riskPerTrade = 1) {
    const riskAmount = accountBalance * (riskPerTrade / 100);
    const stopDistance = this.stopLoss;
    return riskAmount / stopDistance;
  }

  /**
   * 🎯 STEP 5: ENTRY FILTER (Optional - Advanced)
   * Add additional filters before entering trades
   */
  shouldEnter(signal) {
    // 📊 Basic filter: Only enter with high confidence
    if (signal.confidence < 0.7) return false;
    
    // 🕐 Time filter example (uncomment to use)
    /*
    const hour = new Date().getHours();
    if (hour < 8 || hour > 22) return false; // Only trade 8 AM - 10 PM
    */
    
    // 📈 Trend filter example (uncomment to use)
    /*
    if (signal.strategyData && signal.strategyData.percentAbove < 0.5) {
      return false; // Only enter if price is at least 0.5% above/below MA
    }
    */
    
    return true;
  }

  /**
   * 🚪 STEP 6: EXIT LOGIC (Optional - Advanced)
   * Custom exit conditions beyond stop/target
   */
  shouldExit(position, currentPrice) {
    // 📊 This is called every interval while you have a position
    // Add custom exit logic here:
    
    // Example: Time-based exit
    const positionAge = Date.now() - new Date(position.entryTime).getTime();
    const maxHoldTime = 24 * 60 * 60 * 1000; // 24 hours
    
    if (positionAge > maxHoldTime) {
      return true; // Exit after 24 hours regardless of P&L
    }
    
    // Example: Trailing stop logic
    if (position.side === 'BUY') {
      const profit = (currentPrice - position.entryPrice) / position.entryPrice;
      if (profit > 0.01) { // If 1% profit, trail stop to breakeven
        return currentPrice <= position.entryPrice;
      }
    }
    
    return false; // Keep position open
  }

  /**
   * 📊 STEP 7: STATUS AND INFO
   * Display bot information and current settings
   */
  getStatus() {
        return {
      name: this.name,
      strategy: "Educational Tutorial - 50MA Trend Following",
      instruments: this.instruments,
      settings: {
        maLength: this.maLength,
        profitTarget: `${(this.profitTarget * 100).toFixed(1)}%`,
        stopLoss: `${(this.stopLoss * 100).toFixed(1)}%`,
        confidence: `${(this.confidence * 100).toFixed(0)}%`
      },
      tutorial: {
        level: "Beginner",
        concepts: ["Moving Averages", "Trend Following", "Risk Management"],
        nextSteps: ["Try RSI strategy", "Add time filters", "Implement trailing stops"]
      }
    };
  }
}

// 🚀 EXPORT THE BOT CLASS
module.exports = TradingBotTutorial;

/**
 * 🎓 LEARNING EXERCISES - TRY THESE NEXT:
 * 
 * 📚 BEGINNER EXERCISES:
 * 1. Change MA period from 50 to 20 or 200 
 * 2. Adjust profit target and stop loss percentages
 * 3. Modify confidence threshold
 * 
 * 📈 INTERMEDIATE EXERCISES:
 * 1. Implement RSI strategy (uncomment the RSI example)
 * 2. Add time-based trading (only trade certain hours)
 * 3. Combine multiple indicators (MA + RSI)
 * 
 * 🚀 ADVANCED EXERCISES:
 * 1. Create multi-timeframe analysis
 * 2. Add correlation filters between instruments
 * 3. Implement dynamic position sizing
 * 4. Build mean-reversion strategies
 * 
 * 💡 STRATEGY IDEAS TO IMPLEMENT:
 * - Bollinger Bands mean reversion
 * - MACD crossover systems  
 * - Support/resistance breakouts
 * - News-based trading
 * - Volatility breakout strategies
 * - Grid trading systems
 * 
 * 🔧 HOW TO CUSTOMIZE:
 * 1. Keep the same class structure
 * 2. Modify the analyze() function with your logic
 * 3. Change the parameters in constructor
 * 4. Add new helper functions as needed
 * 5. Test with paper trading first!
 * 
 * 📖 REMEMBER:
 * - This is a LEARNING template - the structure is more important than the strategy
 * - Start simple, then add complexity gradually
 * - Always test your strategies before using real money
 * - The bot framework handles execution - you focus on the trading logic
 */$$),

('Python Tutorial Bot', '🎓 Complete Tutorial Bot - Advanced Python version with type hints, logging, and comprehensive examples. Learn professional bot development with multiple strategy examples (MA, RSI, MACD, BB), position sizing, filters, and event handlers.', 'python', 'beginner', 'trend_following', $$"""
🎓 TRADING BOT DEVELOPMENT TUTORIAL (Python)

📖 LEARN BY EXAMPLE:
This template teaches you the fundamental building blocks of trading bots.
Follow the comments to understand each component and build your own strategies.

🎯 WHAT YOU'LL LEARN:
✅ How to structure a trading bot class
✅ How to analyze market data and indicators  
✅ How to generate buy/sell signals
✅ How to manage risk and position sizing
✅ How to implement entry/exit logic
✅ How to add custom parameters and settings

💡 EXAMPLE STRATEGY: 50MA Trend Following
- This is just ONE example - you can implement ANY strategy!
- RSI strategies, Bollinger Bands, MACD crossovers, etc.
- The structure remains the same, just change the logic
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional

class TradingBotTutorial:
    """
    🏗️ STEP 1: CLASS INITIALIZATION - Set Up Your Bot
    This runs once when your bot is created
    """
    
    def __init__(self, config: Dict[str, Any]):
        # 📝 Basic bot information
        self.name = config.get('name', 'Learning Bot')
        self.instruments = config.get('instruments', ['EUR_USD'])
        self.risk_level = config.get('riskLevel', 'medium')
        
        # 🔧 Strategy Parameters (CUSTOMIZE THESE!)
        # Change these values to create different strategies:
        self.ma_length = 50              # Try 20, 100, 200 for different timeframes
        self.profit_target = 0.02        # 2% profit target (try 0.01 for 1%, 0.03 for 3%)
        self.stop_loss = 0.01            # 1% stop loss (try 0.005 or 0.015)
        self.confidence = 0.85           # 85% confidence (try 0.6-0.9)
        
        # 📊 You can add MORE parameters for different strategies:
        # self.rsi_period = 14           # For RSI strategies
        # self.bb_period = 20            # For Bollinger Bands
        # self.macd_fast = 12            # For MACD strategies
        # self.atr_period = 14           # For volatility-based strategies
        
        # 📊 Position and risk management
        self.position_size_pct = 1.0     # 1% risk per trade
        self.max_positions = 3           # Maximum concurrent positions
        
        # 📝 Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(f"TutorialBot_{self.name}")
        
        self.logger.info(f"🎓 {self.name} tutorial bot initialized!")
        self.logger.info(f"📈 Strategy: {self.ma_length}-period Moving Average")
        self.logger.info(f"🎯 Target: {self.profit_target*100}%, Stop: {self.stop_loss*100}%")

    def analyze(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        🎯 STEP 2: ANALYZE FUNCTION - Your Trading Logic
        This is called every execution interval to check for trading opportunities
        
        CUSTOMIZE THIS SECTION TO CREATE YOUR OWN STRATEGIES!
        """
        symbol = market_data.get('symbol')
        price = market_data.get('price')
        indicators = market_data.get('indicators', {})
        
        # 📊 STEP 2A: Get Your Indicators
        # The 'indicators' dict contains pre-calculated technical indicators
        # Available indicators: sma_20, sma_50, sma_200, rsi_14, macd, etc.
        
        ma_50 = indicators.get('sma_50') or indicators.get('ma_50')
        
        if not ma_50:
            return {
                'action': 'HOLD',
                'confidence': 0,
                'reason': f"❌ Missing 50MA data - need at least {self.ma_length} price points"
            }
        
        # 🔍 STEP 2B: Implement Your Strategy Logic
        # This example uses 50MA, but you can implement ANY strategy here:
        
        # EXAMPLE 1: Moving Average Strategy (current example)
        if price > ma_50:
            return self._generate_buy_signal(price, ma_50)
        elif price < ma_50:
            return self._generate_sell_signal(price, ma_50)
        
        # EXAMPLE 2: RSI Strategy (uncomment to use instead)
        """
        rsi = indicators.get('rsi_14')
        if rsi and rsi < 30:
            return self._generate_buy_signal(price, rsi, 'RSI Oversold')
        elif rsi and rsi > 70:
            return self._generate_sell_signal(price, rsi, 'RSI Overbought')
        """
        
        # EXAMPLE 3: MACD Strategy (uncomment to use instead)
        """
        macd = indicators.get('macd')
        macd_signal = indicators.get('macd_signal')
        if macd and macd_signal:
            if macd > macd_signal:
                return self._generate_buy_signal(price, macd, 'MACD Bullish Crossover')
            elif macd < macd_signal:
                return self._generate_sell_signal(price, macd, 'MACD Bearish Crossover')
        """
        
        # EXAMPLE 4: Bollinger Bands Strategy (uncomment to use instead)
        """
        bb_upper = indicators.get('bb_upper')
        bb_lower = indicators.get('bb_lower')
        if bb_lower and price < bb_lower:
            return self._generate_buy_signal(price, bb_lower, 'Bollinger Band Oversold')
        elif bb_upper and price > bb_upper:
            return self._generate_sell_signal(price, bb_upper, 'Bollinger Band Overbought')
        """
        
        return {
            'action': 'HOLD',
            'confidence': 0.3,
            'reason': f"⚖️ No clear signal - price near 50MA ({ma_50:.5f})"
        }

    def _generate_buy_signal(self, price: float, indicator: float, custom_reason: str = None) -> Dict[str, Any]:
        """
        🟢 STEP 3A: BUY Signal Generator
        Customize this to change your buy conditions
        """
        percent_above = ((price - indicator) / indicator) * 100
        
        # 🎯 Calculate entry levels
        entry_price = price
        stop_loss_price = price * (1 - self.stop_loss)
        take_profit_price = price * (1 + self.profit_target)
        
        return {
            'action': 'BUY',
            'confidence': self.confidence,
            'reason': custom_reason or f"💹 BULLISH: Price {price:.5f} is {percent_above:.2f}% above 50MA ({indicator:.5f})",
            'entry_price': entry_price,
            'stop_loss': stop_loss_price,
            'take_profit': take_profit_price,
            'position_size': self._calculate_position_size(price),
            
            # 📝 Add custom data for your strategy
            'strategy_data': {
                'indicator': indicator,
                'percent_above': percent_above,
                'timeframe': '50MA',
                'signal_strength': 'STRONG' if percent_above > 1.0 else 'MODERATE'
            }
        }

    def _generate_sell_signal(self, price: float, indicator: float, custom_reason: str = None) -> Dict[str, Any]:
        """
        🔴 STEP 3B: SELL Signal Generator  
        Customize this to change your sell conditions
        """
        percent_below = ((indicator - price) / indicator) * 100
        
        # 🎯 Calculate entry levels
        entry_price = price
        stop_loss_price = price * (1 + self.stop_loss)
        take_profit_price = price * (1 - self.profit_target)
        
        return {
            'action': 'SELL',
            'confidence': self.confidence,
            'reason': custom_reason or f"📉 BEARISH: Price {price:.5f} is {percent_below:.2f}% below 50MA ({indicator:.5f})",
            'entry_price': entry_price,
            'stop_loss': stop_loss_price,
            'take_profit': take_profit_price,
            'position_size': self._calculate_position_size(price),
            
            # 📝 Add custom data for your strategy
            'strategy_data': {
                'indicator': indicator,
                'percent_below': percent_below,
                'timeframe': '50MA',
                'signal_strength': 'STRONG' if percent_below > 1.0 else 'MODERATE'
            }
        }

    def _calculate_position_size(self, entry_price: float, account_balance: float = 10000) -> float:
        """
        💰 STEP 4: POSITION SIZING (Advanced)
        Calculate how much to trade based on risk management
        """
        # 📊 Risk-based position sizing
        risk_amount = account_balance * (self.position_size_pct / 100)
        stop_distance = entry_price * self.stop_loss
        
        # Calculate units to trade
        position_size = risk_amount / stop_distance
        
        # 📝 Log position sizing decision
        self.logger.info(f"💰 Position size: {position_size:.2f} units (risking ${risk_amount:.2f})")
        
        return position_size

    def should_enter(self, signal: Dict[str, Any], current_positions: int = 0) -> bool:
        """
        🎯 STEP 5: ENTRY FILTER (Advanced)
        Add additional filters before entering trades
        """
        # 📊 Basic filters
        if signal['confidence'] < 0.7:
            self.logger.info("❌ Signal confidence too low")
            return False
        
        if current_positions >= self.max_positions:
            self.logger.info("❌ Maximum positions reached")
            return False
        
        # 🕐 Time filter example (uncomment to use)
        """
        current_hour = datetime.now().hour
        if current_hour < 8 or current_hour > 22:
            self.logger.info("❌ Outside trading hours (8 AM - 10 PM)")
            return False
        """
        
        # 📈 Trend strength filter (uncomment to use)
        """
        strategy_data = signal.get('strategy_data', {})
        if strategy_data.get('signal_strength') != 'STRONG':
            self.logger.info("❌ Signal not strong enough")
            return False
        """
        
        # 📊 Volatility filter example (uncomment to use)
        """
        # Only trade if market is not too volatile
        atr = signal.get('indicators', {}).get('atr_14')
        if atr and atr > entry_price * 0.02:  # More than 2% ATR
            self.logger.info("❌ Market too volatile")
            return False
        """
        
        self.logger.info("✅ All entry filters passed")
        return True

    def should_exit(self, position: Dict[str, Any], current_price: float, market_data: Dict[str, Any]) -> bool:
        """
        🚪 STEP 6: EXIT LOGIC (Advanced)
        Custom exit conditions beyond stop/target
        """
        # 📊 This is called every interval while you have a position
        # Add custom exit logic here:
        
        # Example 1: Time-based exit
        entry_time = datetime.fromisoformat(position.get('entry_time', datetime.now().isoformat()))
        position_age = datetime.now() - entry_time
        max_hold_hours = 24
        
        if position_age.total_seconds() > max_hold_hours * 3600:
            self.logger.info(f"⏰ Exiting position after {max_hold_hours} hours")
            return True
        
        # Example 2: Trailing stop logic
        if position['side'] == 'BUY':
            profit_pct = (current_price - position['entry_price']) / position['entry_price']
            if profit_pct > 0.01:  # If 1% profit, trail stop to breakeven
                breakeven_price = position['entry_price']
                if current_price <= breakeven_price:
                    self.logger.info("📈 Trailing stop triggered at breakeven")
                    return True
        
        # Example 3: Indicator-based exit
        indicators = market_data.get('indicators', {})
        ma_50 = indicators.get('sma_50')
        
        if ma_50:
            # Exit long position if price crosses back below MA
            if position['side'] == 'BUY' and current_price < ma_50:
                self.logger.info("📉 Exiting long position - price below 50MA")
                return True
            # Exit short position if price crosses back above MA  
            elif position['side'] == 'SELL' and current_price > ma_50:
                self.logger.info("📈 Exiting short position - price above 50MA")
                return True
        
        return False  # Keep position open

    def get_status(self) -> Dict[str, Any]:
        """
        📊 STEP 7: STATUS AND INFO
        Display bot information and current settings
        """
        return {
            'name': self.name,
            'strategy': "Educational Tutorial - 50MA Trend Following",
            'instruments': self.instruments,
            'settings': {
                'ma_length': self.ma_length,
                'profit_target': f"{self.profit_target * 100:.1f}%",
                'stop_loss': f"{self.stop_loss * 100:.1f}%", 
                'confidence': f"{self.confidence * 100:.0f}%",
                'position_size': f"{self.position_size_pct}%"
            },
            'risk_management': {
                'max_positions': self.max_positions,
                'risk_per_trade': f"{self.position_size_pct}%"
            },
            'tutorial': {
                'level': "Beginner to Advanced",
                'concepts': ["Moving Averages", "Trend Following", "Risk Management", "Position Sizing"],
                'next_steps': ["Try RSI strategy", "Add time filters", "Implement trailing stops", "Multi-timeframe analysis"]
            }
        }

    def on_trade_opened(self, trade_data: Dict[str, Any]) -> None:
        """
        📈 OPTIONAL: Trade Event Handler
        Called when a trade is opened
        """
        self.logger.info(f"🎯 Trade opened: {trade_data['side']} {trade_data['units']} {trade_data['instrument']}")
        self.logger.info(f"💰 Entry: {trade_data['price']}, Target: {trade_data.get('take_profit')}, Stop: {trade_data.get('stop_loss')}")

    def on_trade_closed(self, trade_data: Dict[str, Any]) -> None:
        """
        📊 OPTIONAL: Trade Event Handler
        Called when a trade is closed
        """
        pnl = trade_data.get('pnl', 0)
        pnl_emoji = "💚" if pnl > 0 else "❌" if pnl < 0 else "⚪"
        
        self.logger.info(f"{pnl_emoji} Trade closed: P&L = {pnl:.2f}")
        self.logger.info(f"📈 Performance update available")

# 🚀 EXPORT THE BOT CLASS
# (In Python, this is handled by the import system)

"""
🎓 LEARNING EXERCISES - TRY THESE NEXT:

📚 BEGINNER EXERCISES:
1. Change MA period from 50 to 20 or 200 
2. Adjust profit target and stop loss percentages
3. Modify confidence threshold
4. Change position sizing percentage

📈 INTERMEDIATE EXERCISES:
1. Implement RSI strategy (uncomment the RSI example)
2. Add time-based trading (only trade certain hours)
3. Combine multiple indicators (MA + RSI)
4. Add volatility filters using ATR

🚀 ADVANCED EXERCISES:
1. Create multi-timeframe analysis
2. Add correlation filters between instruments
3. Implement dynamic position sizing based on volatility
4. Build mean-reversion strategies
5. Add machine learning predictions

💡 STRATEGY IDEAS TO IMPLEMENT:
- Bollinger Bands mean reversion
- MACD crossover systems  
- Support/resistance breakouts
- News-based trading (sentiment analysis)
- Volatility breakout strategies
- Grid trading systems
- Pairs trading (correlation-based)
- Momentum strategies

🔧 HOW TO CUSTOMIZE:
1. Keep the same class structure and method names
2. Modify the analyze() method with your trading logic
3. Change the parameters in __init__()
4. Add new helper methods as needed
5. Use the optional event handlers for trade management
6. Test with paper trading first!

📖 PYTHON SPECIFIC TIPS:
- Use type hints for better code clarity
- Leverage pandas for data analysis if needed
- Use logging instead of print() for better debugging
- Consider using dataclasses for complex data structures
- Add unit tests for your strategy logic

🔍 AVAILABLE INDICATORS (in market_data['indicators']):
- sma_20, sma_50, sma_200 (Simple Moving Averages)
- ema_12, ema_26 (Exponential Moving Averages)  
- rsi_14 (Relative Strength Index)
- macd, macd_signal, macd_histogram (MACD)
- bb_upper, bb_middle, bb_lower (Bollinger Bands)
- atr_14 (Average True Range)
- stoch_k, stoch_d (Stochastic)
- adx (Average Directional Index)

📖 REMEMBER:
- This is a LEARNING template - the structure is more important than the strategy
- Start simple, then add complexity gradually  
- Always test your strategies before using real money
- The bot framework handles execution - you focus on the trading logic
- Use logging to understand what your bot is doing
- Risk management is more important than signal generation
"""$$); 