import requests
import os
import logging
from typing import List, Dict, Optional
from datetime import datetime
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class GoogleSearchService:
    def __init__(self):
        self.api_key = os.getenv('GOOGLE_CSE_API_KEY')
        self.cse_id = os.getenv('GOOGLE_CSE_ID')
        
        if not self.api_key or not self.cse_id:
            logger.warning("Google CSE API key or CSE ID not configured. Web search will be disabled.")
            self.enabled = False
        else:
            self.enabled = True
            logger.info("Google CSE service initialized successfully")
    
    def is_enabled(self) -> bool:
        """Check if Google CSE is properly configured"""
        return self.enabled
    
    def should_search_web(self, query: str) -> bool:
        """
        Determine if a query requires web search based on keywords and patterns
        """
        if not self.enabled:
            return False
        
        # Keywords that indicate need for real-time information
        realtime_keywords = [
            'latest', 'current', 'today', 'now', 'recent', 'breaking',
            'this week', 'this month', 'live', 'real-time', 'update',
            'news', 'announcement', 'report', 'earnings', 'fed', 'ecb',
            'inflation', 'gdp', 'unemployment', 'interest rate', 'policy',
            'price', 'market', 'trading', 'stock', 'forex', 'crypto',
            'bitcoin', 'ethereum', 'oil', 'gold', 'dollar', 'euro'
        ]
        
        # Convert to lowercase for matching
        query_lower = query.lower()
        
        # Check for realtime keywords
        has_realtime_keywords = any(keyword in query_lower for keyword in realtime_keywords)
        
        # Check for question patterns that likely need current info
        question_patterns = [
            r'what.*happening',
            r'what.*latest',
            r'what.*current',
            r'how.*today',
            r'why.*moving',
            r'when.*next',
            r'.*price.*now',
            r'.*market.*today'
        ]
        
        has_question_pattern = any(re.search(pattern, query_lower) for pattern in question_patterns)
        
        # Check for specific financial instrument mentions
        instruments = ['eur/usd', 'gbp/usd', 'usd/jpy', 'btc', 'eth', 'xau', 'oil', 'sp500', 'nasdaq']
        mentions_instruments = any(instrument in query_lower for instrument in instruments)
        
        return has_realtime_keywords or has_question_pattern or mentions_instruments
    
    def search(self, query: str, num_results: int = 5) -> List[Dict]:
        """
        Perform Google Custom Search and return formatted results
        """
        if not self.enabled:
            logger.warning("Google CSE not enabled, skipping search")
            return []
        
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": self.api_key,
                "cx": self.cse_id,
                "q": query,
                "num": min(num_results, 10),  # Google CSE max is 10
                "dateRestrict": "m1",  # Prefer results from last month
                "sort": "date"  # Sort by date to get latest results
            }
            
            logger.info(f"Performing Google search for: {query}")
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get("items", []):
                # Clean up the snippet
                snippet = item.get("snippet", "").replace("\n", " ").strip()
                if len(snippet) > 200:
                    snippet = snippet[:200] + "..."
                
                results.append({
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": snippet,
                    "displayLink": item.get("displayLink", ""),
                    "formattedUrl": item.get("formattedUrl", ""),
                    "timestamp": datetime.now().isoformat()
                })
            
            logger.info(f"Found {len(results)} search results")
            return results
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Google CSE request error: {e}")
            return []
        except Exception as e:
            logger.error(f"Google CSE error: {e}")
            return []
    
    def search_financial_news(self, query: str) -> List[Dict]:
        """
        Search specifically for financial news with targeted sources
        """
        if not self.enabled:
            return []
        
        # Enhance query with financial terms and target financial websites
        enhanced_query = f"{query} site:reuters.com OR site:bloomberg.com OR site:marketwatch.com OR site:cnbc.com OR site:yahoo.com/finance OR site:investing.com OR site:forexfactory.com"
        
        return self.search(enhanced_query, num_results=5)
    
    def search_specific_instrument(self, instrument: str, query_type: str = "news") -> List[Dict]:
        """
        Search for specific financial instrument information
        """
        if not self.enabled:
            return []
        
        # Create targeted search query
        if query_type == "news":
            search_query = f"{instrument} news analysis forecast"
        elif query_type == "analysis":
            search_query = f"{instrument} technical analysis outlook"
        elif query_type == "price":
            search_query = f"{instrument} price movement today"
        else:
            search_query = f"{instrument} {query_type}"
        
        return self.search_financial_news(search_query)
    
    def format_search_context(self, results: List[Dict]) -> str:
        """
        Format search results into a context string for AI
        """
        if not results:
            return ""
        
        context_parts = []
        context_parts.append("CURRENT WEB SEARCH RESULTS:")
        context_parts.append("=" * 40)
        
        for i, result in enumerate(results[:5], 1):
            context_parts.append(f"\n{i}. {result['title']}")
            context_parts.append(f"   Source: {result['displayLink']}")
            context_parts.append(f"   {result['snippet']}")
            context_parts.append(f"   URL: {result['link']}")
        
        context_parts.append("\n" + "=" * 40)
        context_parts.append("Use this information to provide current, accurate analysis.\n")
        
        return "\n".join(context_parts)
    
    def extract_sources_for_frontend(self, results: List[Dict]) -> List[Dict]:
        """
        Format search results for the frontend sources display
        """
        if not results:
            return []
        
        sources = []
        for result in results[:5]:  # Limit to top 5 sources
            sources.append({
                'title': result['title'],
                'snippet': result['snippet'],
                'url': result['link'],
                'timestamp': datetime.now().isoformat()
            })
        
        return sources 