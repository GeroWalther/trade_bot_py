from quart import Quart
from routes.market_intelligence import market_bp

app = Quart('ai_trade_analysis')
app.register_blueprint(market_bp) 