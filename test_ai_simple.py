#!/usr/bin/env python3

import asyncio
import os
from services.simple_ai_service import SimpleAIService

async def test_ai_analysis():
    """Test the AI analysis service"""
    
    print("🧪 Testing AI Analysis Service...")
    
    # Check if OpenAI API key is available
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ No OPENAI_API_KEY found in environment")
        return
    
    try:
        # Initialize AI service
        ai_service = SimpleAIService()
        print("✅ AI Service initialized")
        
        # Test with Gold (should be easy to analyze)
        print("\n🔍 Testing with Gold...")
        result = await ai_service.get_comprehensive_analysis('Gold', 'Day trade')
        
        print("\n📊 AI Analysis Result:")
        print(f"Direction: {result.get('trading_strategy', {}).get('direction', 'N/A')}")
        print(f"Entry Price: {result.get('trading_strategy', {}).get('entry', {}).get('price', 'N/A')}")
        print(f"Take Profit: {result.get('trading_strategy', {}).get('take_profit_1', {}).get('price', 'N/A')}")
        print(f"Stop Loss: {result.get('trading_strategy', {}).get('stop_loss', {}).get('price', 'N/A')}")
        print(f"Confidence: {result.get('confidence_level', 'N/A')}")
        print(f"Rationale: {result.get('trading_strategy', {}).get('rationale', 'N/A')[:200]}...")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    asyncio.run(test_ai_analysis()) 