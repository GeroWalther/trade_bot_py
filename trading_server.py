from flask import Flask, jsonify, request
from oanda_trader import OandaTrader
from config import OANDA_CREDS
from flask_cors import CORS
import oandapyV20.endpoints.positions as positions
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
        data = request.get_json()
        logger.info(f"Received trade request: {data}")
        
        symbol = data.get('symbol', 'EUR_USD')
        side = data.get('side', 'buy')
        quantity = data.get('quantity', 1000)
        
        logger.info(f"Processed request parameters: symbol={symbol}, side={side}, quantity={quantity}")
        
        # Get current positions to check if we already have one
        current_positions = broker.get_tracked_positions()
        has_position = symbol in current_positions
        logger.info(f"Current positions check - Has position for {symbol}: {has_position}")
        if has_position:
            logger.info(f"Existing position details: {current_positions[symbol]}")
        
        # Check if market is open for this symbol
        if not broker.is_market_open(symbol):
            return jsonify({
                'status': 'error',
                'message': f'Market is closed for {symbol}'
            }), 400
        
        order = {
            'strategy': None,
            'symbol': symbol,
            'quantity': quantity,
            'side': side
        }
        
        logger.info(f"Submitting order to broker: {order}")
        try:
            order_id = broker.submit_order(order)
            logger.info(f"Broker response - order_id: {order_id}")
        except Exception as e:
            logger.error(f"Error submitting order: {e}", exc_info=True)
            return jsonify({
                'status': 'error',
                'message': f'Order submission error: {str(e)}'
            }), 400
        
        if order_id:
            # Get updated position info
            positions = broker.get_tracked_positions()
            position_info = positions.get(symbol, {})
            logger.info(f"Updated position info: {position_info}")
            
            response = {
                'status': 'success',
                'order_id': order_id,
                'message': 'Order executed successfully',
                'position': position_info
            }
            logger.info(f"Sending success response: {response}")
            return jsonify(response), 200
            
        response = {
            'status': 'error',
            'message': 'Order submission failed'
        }
        logger.error(f"Order submission failed - no order_id returned")
        return jsonify(response), 400
        
    except Exception as e:
        error_response = {
            'status': 'error',
            'message': str(e)
        }
        logger.error(f"Exception in execute_trade: {e}", exc_info=True)
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
        
        # Define all available instruments
        instruments = [
            'EUR_USD', 'GBP_USD', 'USD_JPY', 'AUD_USD', 'USD_CAD', 'BTC_USD',
            'SPX500_USD', 'NAS100_USD', 'XAU_USD', 'BCO_USD'
        ]
        
        # Get market prices for all instruments
        market_prices = {}
        for instrument in instruments:
            market_prices[instrument] = {
                'price': broker.get_last_price(instrument),
                'action': broker.get_price_action(instrument)
            }
        market_prices['last_update'] = datetime.now(pytz.UTC).isoformat()
        
        # Get market status
        market_status = {
            'is_market_open': broker.is_market_open(),
            'active_symbols': instruments
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

@app.route('/api/positions', methods=['GET'])
def get_positions():
    try:
        # Get positions directly from OANDA
        r = positions.OpenPositions(accountID=OANDA_CREDS["ACCOUNT_ID"])
        response = broker.api.request(r)
        logger.info(f"OANDA positions response: {response}")
        
        # Format positions for display
        formatted_positions = {}
        for position in response.get('positions', []):
            symbol = position['instrument']
            
            # Get long/short position details
            long_units = float(position.get('long', {}).get('units', 0))
            short_units = float(position.get('short', {}).get('units', 0))
            
            # Determine if long or short position
            quantity = long_units if long_units != 0 else short_units
            entry_price = float(position.get('long' if long_units != 0 else 'short', {}).get('averagePrice', 0))
            
            # Get current price
            current_price = broker.get_last_price(symbol)
            
            # Calculate P/L
            pl_euro = float(position.get('unrealizedPL', 0))
            pl_pct = ((current_price - entry_price) / entry_price * 100) if current_price else 0
            if quantity < 0:  # Invert percentage for short positions
                pl_pct = -pl_pct
            
            formatted_positions[symbol] = {
                'quantity': quantity,
                'entry_price': entry_price,
                'current_price': current_price,
                'pl_euro': pl_euro,
                'profit_pct': pl_pct,
                'side': 'LONG' if quantity > 0 else 'SHORT',
                'unrealized_pl': pl_euro
            }
            
        logger.info(f"Formatted positions: {formatted_positions}")
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
    app.run(host='0.0.0.0', port=5002, debug=True) 