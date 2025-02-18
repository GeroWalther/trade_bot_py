from flask import Flask, jsonify
from oanda_trader import OandaTrader
from config import OANDA_CREDS
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

broker = OandaTrader(OANDA_CREDS)

@app.route('/trading-status')
def get_trading_status():
    # Get account info
    cash, positions_value, total_value = broker._get_balances_at_broker()
    
    # Get current market prices
    eur_usd_price = broker.get_last_price("EUR_USD", quote="USD")
    btc_usd_price = broker.get_last_price("BTC_USD", quote="USD")
    
    # Get price action for each symbol
    eur_usd_action = broker.get_price_action("EUR_USD")
    btc_usd_action = broker.get_price_action("BTC_USD")
    
    data = {
        'account': {
            'balance': cash,
            'unrealized_pl': positions_value,
            'total_value': total_value,
        },
        'market_prices': {
            'EUR_USD': {
                'price': float(eur_usd_price) if eur_usd_price is not None else None,
                'action': eur_usd_action
            },
            'BTC_USD': {
                'price': float(btc_usd_price) if btc_usd_price is not None else None,
                'action': btc_usd_action
            },
            'last_update': broker.get_time().isoformat()
        },
        'positions': broker.get_tracked_positions(),
        'market_status': {
            'is_market_open': broker.is_market_open(),
            'active_symbols': ["EUR_USD", "BTC_USD"]
        }
    }
    return jsonify(data)

@app.route('/test-trade')
def make_test_trade():
    try:
        # Example: Buy 1000 units of EUR/USD
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
                'message': 'Order submission failed. Check server logs for details.'
            }), 400
            
        return jsonify({
            'status': 'success',
            'order_id': order_id,
            'message': 'Order submitted successfully'
        })
        
    except Exception as e:
        logging.error(f"Error in test trade endpoint: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/close-position/<symbol>')
def close_position(symbol):
    try:
        # Get current position
        positions = broker.get_tracked_positions()
        if symbol not in positions:
            return jsonify({'status': 'error', 'message': 'No position found'}), 404
            
        position = positions[symbol]
        
        # Create opposite order to close
        order = {
            'strategy': None,
            'symbol': symbol,
            'quantity': abs(position['quantity']),  # Use absolute value
            'side': 'sell' if position['quantity'] > 0 else 'buy'  # Opposite of current position
        }
        
        order_id = broker.submit_order(order)
        return jsonify({'status': 'success', 'order_id': order_id})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True) 