from quart import Quart
from quart_cors import cors
from routes.ai_analysis_routes import ai_analysis_bp
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def create_app():
    app = Quart(__name__)
    app = cors(app, allow_origin="*")
    
    # Register AI analysis routes
    logger.info("Registering AI analysis blueprint...")
    app.register_blueprint(ai_analysis_bp)
    
    return app

app = create_app()

if __name__ == '__main__':
    logger.info("Starting AI Analysis server on port 5005...")
    app.run(host='0.0.0.0', port=5005, debug=True) 