# This should be your AI analysis server
from quart import Quart, jsonify
from quart_cors import cors
from routes.analysis_routes import analysis_bp
from services.ai_analysis_service_new import AIAnalysisService
from services.market_data_service import MarketDataService
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    app = Quart(__name__)
    app = cors(app)
    
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

    # Register blueprint with services
    app.register_blueprint(analysis_bp)
    
    return app

app = create_app()

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5003))
    debug = os.getenv('FLASK_ENV') == 'development'
    
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.info("Running in development mode")
    
    app.run(
        host='0.0.0.0', 
        port=port,
        debug=debug
    ) 