from flask import Flask, jsonify
from oanda_trader import OandaTrader
from config import OANDA_CREDS
from flask_cors import CORS
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.pricing as pricing
from oandapyV20 import API
import logging

app = Flask(__name__)
CORS(app)

# Initialize OANDA API client
api = API(access_token=OANDA_CREDS["ACCESS_TOKEN"])
broker = OandaTrader(OANDA_CREDS)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

@app.route('/close-position/<symbol>', methods=['POST'])
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

@app.route('/api/positions', methods=['GET'])
def get_positions():
    try:
        # Get positions from OANDA
        r = positions.OpenPositions(accountID=OANDA_CREDS["ACCOUNT_ID"])
        response = api.request(r)
        
        # Format positions for display
        formatted_positions = {}
        for position in response['positions']:
            symbol = position['instrument']
            
            # Get current price
            params = {"instruments": symbol}
            price_r = pricing.PricingInfo(accountID=OANDA_CREDS["ACCOUNT_ID"], params=params)
            price_response = api.request(price_r)
            current_price = float(price_response['prices'][0]['closeoutAsk']) if price_response['prices'] else None
            
            # Determine if long or short position
            is_long = 'long' in position and float(position['long']['units']) > 0
            quantity = float(position['long']['units']) if is_long else float(position['short']['units'])
            entry_price = float(position['long']['averagePrice']) if is_long else float(position['short']['averagePrice'])
            
            # Calculate P/L
            pl_euro = float(position['unrealizedPL'])
            pl_pct = (current_price - entry_price) / entry_price * 100
            if not is_long:  # Invert for short positions
                pl_pct = -pl_pct
            
            formatted_positions[symbol] = {
                'quantity': quantity,
                'entry_price': entry_price,
                'current_price': current_price,
                'pl_euro': pl_euro,
                'profit_pct': pl_pct,
                'side': 'LONG' if is_long else 'SHORT',
                'unrealized_pl': float(position['unrealizedPL'])
            }
            
        logger.info(f"Current open positions: {formatted_positions}")
        return jsonify({
            'status': 'success',
            'positions': formatted_positions,
            'count': len(formatted_positions)
        })
        
    except Exception as e:
        logger.error(f"Error getting positions: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True) 