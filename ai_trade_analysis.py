# This should be your AI analysis server
from quart import Quart, jsonify, request
from routes.analysis_routes import analysis_bp
from routes.market_intelligence import market_bp
from routes.ai_analysis_routes import ai_analysis_bp
from services.ai_analysis_service_new import AIAnalysisService
from services.market_data_service import MarketDataService
from services.market_analyzer import MarketAnalyzer
import logging
import os
from datetime import datetime
from fastapi import HTTPException
from quart_cors import cors

# Configure logging - Set higher log levels to reduce output
logging.basicConfig(
    level=logging.WARNING,  # Set to WARNING to reduce info logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# Only show INFO logs for our application
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Silence other loggers
logging.getLogger('quart').setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.WARNING)
logging.getLogger('hypercorn').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('anthropic').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)

def create_app():
    app = Quart(__name__)
    
    # Initialize services
    try:
        # Get API keys from environment
        alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if not alpha_vantage_key:
            raise ValueError("ALPHA_VANTAGE_API_KEY is required")

        # Initialize services as app extensions
        app.market_data = MarketDataService(alpha_vantage_key=alpha_vantage_key)
        app.ai_analysis = AIAnalysisService(alpha_vantage_key=alpha_vantage_key)
        
        logger.info("Services initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

    # Register blueprints
    app.register_blueprint(analysis_bp)
    app.register_blueprint(market_bp)
    app.register_blueprint(ai_analysis_bp)
    
    app = cors(app, allow_origin="*")  # Enable CORS
    
    return app

app = create_app()

@app.route('/api/news/<symbol>')
async def get_news(symbol):
    try:
        analyzer = MarketAnalyzer(
            alpha_vantage_key=os.getenv('ALPHA_VANTAGE_API_KEY')
        )
        news_data = await analyzer._fetch_news_data(symbol)
        
        if not news_data:
            return jsonify({
                'status': 'error',
                'message': 'No news data found'
            })
            
        return jsonify({
            'status': 'success',
            'data': news_data
        })
    except Exception as e:
        logger.error(f"Error fetching news: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    print("\n" + "="*50)
    print("Starting AI Analysis server on port 5005...")
    print("="*50 + "\n")
    logger.info("AI Analysis server is running")
    # Set debug to False to reduce logs
    app.run(host='0.0.0.0', port=5005, debug=False) 