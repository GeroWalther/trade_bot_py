from flask import Flask, jsonify, request
from oanda_trader import OandaTrader
from config import OANDA_CREDS
from flask_cors import CORS
import logging
from datetime import datetime
import pytz

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
        # Log the incoming request
        data = request.get_json()
        logger.info(f"Received trade request: {data}")
        
        symbol = data.get('symbol', 'EUR_USD')
        side = data.get('side', 'buy')
        quantity = data.get('quantity', 1000)
        
        logger.info(f"Processed request parameters: symbol={symbol}, side={side}, quantity={quantity}")
        
        order = {
            'strategy': None,
            'symbol': symbol,
            'quantity': quantity,
            'side': side
        }
        
        logger.info(f"Submitting order to broker: {order}")
        order_id = broker.submit_order(order)
        logger.info(f"Broker response - order_id: {order_id}")
        
        # Check if we have a new position or updated position
        positions = broker.get_tracked_positions()
        if symbol in positions:
            response = {
                'status': 'success',
                'order_id': order_id or 'unknown',  # Use order_id if available
                'message': 'Order executed successfully',
                'position': positions[symbol]  # Include the updated position
            }
            logger.info(f"Sending success response: {response}")
            return jsonify(response), 200
            
        response = {
            'status': 'error',
            'message': 'Order submission failed'
        }
        logger.info(f"Sending error response: {response}")
        return jsonify(response), 400
        
    except Exception as e:
        error_response = {
            'status': 'error',
            'message': str(e)
        }
        logger.error(f"Exception in execute_trade: {e}", exc_info=True)
        logger.error(f"Sending error response: {error_response}")
        return jsonify(error_response), 500

@app.route('/close-position/<symbol>', methods=['POST'])
def close_position(symbol):
    try:
        # Get current positions before closing
        initial_positions = broker.get_tracked_positions()
        if symbol not in initial_positions:
            return jsonify({
                'status': 'error',
                'message': 'No position found'
            }), 404
            
        position = initial_positions[symbol]
        order = {
            'strategy': None,
            'symbol': symbol,
            'quantity': abs(position['quantity']),
            'side': 'sell' if position['quantity'] > 0 else 'buy'
        }
        
        logger.info(f"Closing position for {symbol}: {order}")
        order_id = broker.submit_order(order)
        
        # Verify position was actually closed by checking updated positions
        updated_positions = broker.get_tracked_positions()
        if symbol not in updated_positions:
            # Position was successfully closed
            return jsonify({
                'status': 'success',
                'order_id': order_id,
                'message': 'Position closed successfully'
            }), 200
            
        # Position still exists - closing failed
        return jsonify({
            'status': 'error',
            'message': 'Position closing failed - position still exists'
        }), 400
        
    except Exception as e:
        logger.error(f"Error closing position: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/trading-status', methods=['GET'])
def get_trading_status():
    try:
        # Get account info
        cash, positions_value, total_value = broker._get_balances_at_broker()
        
        # Get positions
        positions = broker.get_tracked_positions()
        
        # Get market prices
        market_prices = {
            'EUR_USD': {
                'price': broker.get_last_price('EUR_USD'),
                'action': broker.get_price_action('EUR_USD')
            },
            'BTC_USD': {
                'price': broker.get_last_price('BTC_USD'),
                'action': broker.get_price_action('BTC_USD')
            },
            'last_update': datetime.now(pytz.UTC).isoformat()
        }
        
        # Get market status
        market_status = {
            'is_market_open': broker.is_market_open(),
            'active_symbols': ['EUR_USD', 'BTC_USD']
        }
        
        return jsonify({
            'account': {
                'balance': cash,
                'unrealized_pl': positions_value,
                'total_value': total_value
            },
            'positions': positions,
            'market_prices': market_prices,
            'market_status': market_status,
            'trading_stats': {
                'win_rate': 0,
                'winning_trades': 0,
                'losing_trades': 0
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting trading status: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5002, debug=True) 