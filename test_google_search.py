#!/usr/bin/env python3
"""
Test script for Google Custom Search Engine integration
"""

import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env file
load_dotenv()

from services.google_search_service import GoogleSearchService

def test_google_search():
    """Test the Google CSE integration"""
    
    print("🔍 Testing Google Custom Search Engine Integration")
    print("=" * 50)
    
    # Initialize search service
    search_service = GoogleSearchService()
    
    # Check if service is enabled
    if not search_service.is_enabled():
        print("❌ Google CSE is not configured properly!")
        print("\nPlease check:")
        print("1. GOOGLE_CSE_API_KEY environment variable is set")
        print("2. GOOGLE_CSE_ID environment variable is set")
        print("\nSee GOOGLE_CSE_SETUP.md for setup instructions.")
        return False
    
    print("✅ Google CSE service is configured and enabled")
    
    # Test queries
    test_queries = [
        "What's happening in the markets today?",
        "EUR/USD latest news",
        "Fed interest rate policy",
        "Bitcoin price analysis",
        "This should not trigger search"  # Should not trigger search
    ]
    
    print(f"\n🧪 Testing query detection and search functionality")
    print("-" * 50)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Testing query: '{query}'")
        
        # Test if query should trigger search
        should_search = search_service.should_search_web(query)
        print(f"   Should search: {'✅ Yes' if should_search else '❌ No'}")
        
        if should_search:
            print("   🔍 Performing search...")
            try:
                results = search_service.search_financial_news(query)
                
                if results:
                    print(f"   ✅ Found {len(results)} results")
                    for j, result in enumerate(results[:2], 1):  # Show first 2 results
                        print(f"     {j}. {result['title'][:60]}...")
                        print(f"        Source: {result['displayLink']}")
                else:
                    print("   ⚠️  No results found")
                    
            except Exception as e:
                print(f"   ❌ Search failed: {e}")
                return False
        
        print()
    
    # Test context formatting
    print("📝 Testing context formatting...")
    sample_results = [
        {
            'title': 'EUR/USD Falls on ECB Policy Uncertainty',
            'link': 'https://example.com/news1',
            'snippet': 'The EUR/USD pair declined following ECB policy statements...',
            'displayLink': 'reuters.com'
        },
        {
            'title': 'Federal Reserve Holds Rates Steady',
            'link': 'https://example.com/news2', 
            'snippet': 'The Federal Reserve maintained interest rates at current levels...',
            'displayLink': 'bloomberg.com'
        }
    ]
    
    context = search_service.format_search_context(sample_results)
    frontend_sources = search_service.extract_sources_for_frontend(sample_results)
    
    print("   ✅ Context formatting successful")
    print(f"   ✅ Frontend sources formatted: {len(frontend_sources)} sources")
    
    print(f"\n🎉 Google CSE integration test completed successfully!")
    print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return True

def test_environment_setup():
    """Test if environment variables are properly set"""
    
    print("\n🔧 Checking environment setup...")
    print("-" * 30)
    
    api_key = os.getenv('GOOGLE_CSE_API_KEY')
    cse_id = os.getenv('GOOGLE_CSE_ID')
    
    if api_key:
        print(f"✅ GOOGLE_CSE_API_KEY: {api_key[:20]}...")
    else:
        print("❌ GOOGLE_CSE_API_KEY: Not set")
    
    if cse_id:
        print(f"✅ GOOGLE_CSE_ID: {cse_id}")
    else:
        print("❌ GOOGLE_CSE_ID: Not set")
    
    return bool(api_key and cse_id)

if __name__ == "__main__":
    print("🚀 Google CSE Integration Test")
    print("=" * 50)
    
    # Test environment setup first
    env_ok = test_environment_setup()
    
    if not env_ok:
        print("\n❌ Environment setup incomplete!")
        print("Please set up your Google CSE credentials.")
        print("See GOOGLE_CSE_SETUP.md for instructions.")
        sys.exit(1)
    
    # Run main test
    success = test_google_search()
    
    if success:
        print("\n🎉 All tests passed! Your Google CSE integration is ready.")
        print("\nNext steps:")
        print("1. Start your trading bot server: python master_server.py")
        print("2. Open the Electron client")
        print("3. Try asking questions like 'What's the latest EUR/USD news?'")
    else:
        print("\n❌ Some tests failed. Please check your configuration.")
        sys.exit(1) 