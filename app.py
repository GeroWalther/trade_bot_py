from quart import Quart
from quart_cors import cors
from routes.trading_routes import trading_bp
from config import validate_api_keys
import logging
from routes.analysis_routes import analysis_bp
from routes.ai_analysis_routes import ai_analysis_bp

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def create_app():
    app = Quart(__name__)
    app = cors(app)
    
    # Validate API keys on startup
    validate_api_keys()
    
    # Register trading routes
    app.register_blueprint(trading_bp)
    app.register_blueprint(analysis_bp)
    app.register_blueprint(ai_analysis_bp)
    
    @app.before_serving
    async def startup():
        logger.info("Starting up the application...")
        # Test OANDA connection
        from oandapyV20.endpoints.accounts import AccountSummary
        from config import OANDA_CREDS
        try:
            from services.market_analyzer import MarketAnalyzer
            analyzer = MarketAnalyzer()
            r = AccountSummary(OANDA_CREDS['ACCOUNT_ID'])
            analyzer.client.request(r)
            logger.info("OANDA connection test successful")
        except Exception as e:
            logger.error(f"OANDA connection test failed: {e}")
            raise
    
    return app

app = create_app()

# This is your main trading server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)  # Trading server on 5002 