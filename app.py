from quart import Quart
from quart_cors import cors
from routes.trading_routes import trading_bp
from config import validate_api_keys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    app = Quart(__name__)
    app = cors(app)
    
    # Validate API keys on startup
    validate_api_keys()
    
    # Register trading routes
    app.register_blueprint(trading_bp)
    
    return app

app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)  # Trading server on 5002 