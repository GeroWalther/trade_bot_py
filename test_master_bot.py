#!/usr/bin/env python3
"""
Test script for the Master Trading Bot
Tests basic functionality without executing real trades
"""

import asyncio
import logging
import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from master_trading_bot import MasterTradingBot
from config import OANDA_CREDS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def test_master_bot():
    """Test the Master Trading Bot functionality"""
    try:
        print("🧪 Testing Master Trading Bot...")
        print("=" * 50)
        
        # Initialize the bot
        print("1. Initializing Master Trading Bot...")
        bot = MasterTradingBot()
        print("   ✅ Bot initialized successfully")
        
        # Test broker connection
        print("\n2. Testing broker connection...")
        try:
            cash, _, total_value = bot.broker._get_balances_at_broker()
            print(f"   ✅ Connected to OANDA - Account Balance: ${total_value:.2f}")
        except Exception as e:
            print(f"   ❌ Broker connection failed: {e}")
            return False
        
        # Test market data
        print("\n3. Testing market data...")
        test_symbol = 'EUR_USD'
        price = bot.broker.get_last_price(test_symbol)
        if price:
            print(f"   ✅ Market data working - {test_symbol}: {price}")
        else:
            print(f"   ❌ No market data for {test_symbol}")
            return False
        
        # Test AI analysis for one asset
        print("\n4. Testing AI analysis...")
        try:
            analysis = await bot.get_ai_analysis_for_asset('XAU_USD')
            if analysis and 'trading_strategy' in analysis:
                direction = analysis['trading_strategy'].get('direction', 'UNKNOWN')
                print(f"   ✅ AI analysis working - Gold recommendation: {direction}")
            else:
                print("   ❌ AI analysis failed or incomplete")
                return False
        except Exception as e:
            print(f"   ❌ AI analysis error: {e}")
            return False
        
        # Test profit scoring
        print("\n5. Testing profit scoring...")
        current_price = bot.broker.get_last_price('XAU_USD')
        if current_price and analysis:
            score = bot.calculate_profit_score('XAU_USD', analysis, current_price)
            print(f"   ✅ Profit scoring working - Score: {score:.2f}/100")
        else:
            print("   ❌ Profit scoring failed")
            return False
        
        # Test position sizing
        print("\n6. Testing position sizing...")
        try:
            trading_strategy = analysis['trading_strategy']
            entry_price = bot.parse_price(trading_strategy.get('entry', {}).get('price', current_price))
            stop_loss = bot.parse_price(trading_strategy.get('stop_loss', {}).get('price', 0))
            
            if entry_price and stop_loss:
                position_size = bot.calculate_position_size('XAU_USD', entry_price, stop_loss)
                print(f"   ✅ Position sizing working - Size: {position_size}")
            else:
                print("   ❌ Position sizing failed - invalid prices")
                return False
        except Exception as e:
            print(f"   ❌ Position sizing error: {e}")
            return False
        
        # Test status reporting
        print("\n7. Testing status reporting...")
        status = bot.get_status()
        if status and 'is_running' in status:
            print("   ✅ Status reporting working")
        else:
            print("   ❌ Status reporting failed")
            return False
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed! Master Trading Bot is ready.")
        print("\n📋 Next Steps:")
        print("1. Start the master server: python master_server.py")
        print("2. Start the Electron client: cd ../electron_client && npm start")
        print("3. Navigate to the Master Bot interface")
        print("4. Use 'Analyze Markets' to test market analysis")
        print("5. Start the bot for automated trading")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        logger.error(f"Test error: {e}", exc_info=True)
        return False

async def quick_analysis_test():
    """Run a quick analysis test on all assets"""
    try:
        print("\n🔍 Running quick analysis test...")
        bot = MasterTradingBot()
        
        # Test a few assets quickly
        test_assets = ['XAU_USD', 'EUR_USD', 'BTC_USD']
        results = {}
        
        for symbol in test_assets:
            try:
                print(f"   Analyzing {symbol}...")
                price = bot.broker.get_last_price(symbol)
                if price:
                    # Create a minimal analysis for testing
                    test_analysis = {
                        'trading_strategy': {
                            'direction': 'LONG',
                            'entry': {'price': price},
                            'take_profit_1': {'price': price * 1.02},
                            'stop_loss': {'price': price * 0.98},
                            'rationale': 'Test analysis'
                        }
                    }
                    score = bot.calculate_profit_score(symbol, test_analysis, price)
                    results[symbol] = score
                    print(f"     ✅ {symbol}: Score {score:.1f}")
            except Exception as e:
                print(f"     ❌ {symbol}: Error - {e}")
        
        if results:
            best_asset = max(results, key=results.get)
            print(f"\n🏆 Best opportunity: {best_asset} (Score: {results[best_asset]:.1f})")
            return True
        else:
            print("   ❌ No analysis results")
            return False
            
    except Exception as e:
        print(f"   ❌ Analysis test failed: {e}")
        return False

if __name__ == '__main__':
    print("🚀 Master Trading Bot Test Suite")
    print("=" * 50)
    
    # Check if we're in practice mode
    if OANDA_CREDS.get('ENVIRONMENT') == 'practice':
        print("✅ Running in PRACTICE mode - Safe for testing")
    else:
        print("⚠️  WARNING: Not in practice mode!")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Test aborted.")
            sys.exit(1)
    
    try:
        # Run main test
        success = asyncio.run(test_master_bot())
        
        if success:
            # Run quick analysis test
            print("\n" + "=" * 50)
            analysis_success = asyncio.run(quick_analysis_test())
            
            if analysis_success:
                print("\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
                print("The Master Trading Bot is ready for use.")
            else:
                print("\n⚠️  Main tests passed but analysis test had issues.")
        else:
            print("\n❌ Tests failed. Please check the configuration.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user.")
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        sys.exit(1) 