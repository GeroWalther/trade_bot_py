from quart import Quart
from quart_cors import cors
from routes.analysis_routes import analysis_bp
from config import validate_api_keys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    app = Quart(__name__)
    app = cors(app)
    
    # Validate API keys on startup
    validate_api_keys()
    
    # Register blueprint without prefix since this is a dedicated analysis server
    app.register_blueprint(analysis_bp)
    
    return app

app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=True)  # Note port 5003 