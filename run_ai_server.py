#!/usr/bin/env python3
"""
Simple script to run the AI analysis server with minimal output.
"""
import os
import sys
import logging

# Configure logging to suppress most logs
logging.basicConfig(level=logging.ERROR)

# Silence common noisy loggers
for logger_name in [
    'quart', 'asyncio', 'hypercorn', 'urllib3', 
    'anthropic', 'openai', 'werkzeug', 'httpx'
]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  STARTING AI ANALYSIS SERVER ON PORT 5005")
    print("  Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    # Import and run the AI analysis server
    from ai_trade_analysis import app
    
    # Run the app with minimal output
    app.run(host='0.0.0.0', port=5005, debug=False) 