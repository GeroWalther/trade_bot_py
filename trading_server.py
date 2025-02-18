from flask import Flask, jsonify
from oanda_trader import OandaTrader
from config import OANDA_CREDS
from flask_cors import CORS
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

try:
    broker = OandaTrader(OANDA_CREDS)
    logger.info("Successfully initialized OANDA broker")
except Exception as e:
    logger.error(f"Failed to initialize OANDA broker: {e}", exc_info=True)
    raise

@app.route('/execute-trade', methods=['POST'])
def execute_trade():
    try:
        order = {
            'strategy': None,
            'symbol': 'EUR_USD',
            'quantity': 1000,
            'side': 'buy'
        }
        
        order_id = broker.submit_order(order)
        
        if not order_id:
            return jsonify({
                'status': 'error',
                'message': 'Order submission failed'
            }), 400
            
        return jsonify({
            'status': 'success',
            'order_id': order_id
        })
        
    except Exception as e:
        logger.error(f"Trade execution error: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/close-position/<symbol>', methods=['POST'])
def close_position(symbol):
    try:
        positions = broker.get_tracked_positions()
        if symbol not in positions:
            return jsonify({
                'status': 'error',
                'message': 'No position found'
            }), 404
            
        position = positions[symbol]
        order = {
            'strategy': None,
            'symbol': symbol,
            'quantity': abs(position['quantity']),
            'side': 'sell' if position['quantity'] > 0 else 'buy'
        }
        
        order_id = broker.submit_order(order)
        return jsonify({'status': 'success', 'order_id': order_id})
    except Exception as e:
        logger.error(f"Position closing error: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5002, debug=True) 