from flask import Flask, jsonify, request
from oanda_trader import OandaTrader
from config import OANDA_CREDS
from flask_cors import CORS
import oandapyV20.endpoints.positions as positions
import logging
from datetime import datetime
import pytz
from trading_state import strategy, initialize_strategy
from strategies.ema_trend import EMATrendStrategy
from strategies.bb_strategy import BBStrategy
import threading
import time
from services.market_intelligence_service import MarketIntelligenceService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

try:
    broker = OandaTrader(OANDA_CREDS)
    logger.info("Successfully initialized OANDA broker")
    
    # Initialize both strategies
    bb_strategy = BBStrategy(broker)
    ema_strategy = EMATrendStrategy(broker=broker)
    
    logger.info("Successfully initialized strategies")
except Exception as e:
    logger.error(f"Failed to initialize broker or strategies: {e}", exc_info=True)
    raise

# Initialize the service
market_intelligence = MarketIntelligenceService()

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

@app.route('/api/bots', methods=['GET'])
def get_bots():
    """Get status of all bots"""
    try:
        logger.info("Getting bots status...")
        logger.info(f"EMA Strategy exists: {ema_strategy is not None}")
        logger.info(f"BB Strategy exists: {bb_strategy is not None}")
        
        response = {
            'status': 'success',
            'bots': {
                'ema_strategy': {
                    'name': 'EMA Trend Strategy',
                    'running': ema_strategy.should_continue(),
                    'parameters': ema_strategy.parameters,
                    'status': ema_strategy.get_status()
                },
                'bb_strategy': {
                    'name': 'Bollinger Bands Strategy',
                    'running': bb_strategy.should_continue(),
                    'parameters': bb_strategy.parameters,
                    'status': bb_strategy.get_status()
                }
            }
        }
        logger.info(f"Returning bots: {response}")
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error getting bots: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

def start_strategy(strategy):
    """Start strategy in a background thread"""
    def run_strategy():
        try:
            strategy.start()
        except Exception as e:
            logger.error(f"Strategy thread error: {str(e)}")
            strategy._continue = False
    
    thread = threading.Thread(target=run_strategy, daemon=True)
    thread.start()
    return thread

@app.route('/api/bots/<bot_id>/toggle', methods=['POST'])
def toggle_bot(bot_id):
    try:
        if bot_id == 'bb_strategy':
            if not bb_strategy:
                return jsonify({
                    'status': 'error',
                    'message': 'Strategy not initialized'
                }), 400
                
            was_running = bb_strategy.should_continue()
            if was_running:
                bb_strategy.stop()
                new_status = 'stopped'
            else:
                start_strategy(bb_strategy)
                new_status = 'running'
                
        elif bot_id == 'ema_strategy':
            if not ema_strategy:
                return jsonify({
                    'status': 'error',
                    'message': 'Strategy not initialized'
                }), 400
                
            was_running = ema_strategy.should_continue()
            if was_running:
                ema_strategy.stop()
                new_status = 'stopped'
            else:
                start_strategy(ema_strategy)
                new_status = 'running'
            
        return jsonify({
            'status': 'success',
            'bot_id': bot_id,
            'bot_status': new_status
        })
        
    except Exception as e:
        logger.error(f"Error toggling bot {bot_id}: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/bots/<bot_id>/parameters', methods=['PUT'])
def update_bot_parameters(bot_id):
    try:
        settings = request.get_json()
        logger.info(f"Received settings update for {bot_id}: {settings}")
        
        if bot_id == 'ema_strategy':
            # Log current state
            logger.info(f"Current parameters: {ema_strategy.parameters}")
            
            # Create a copy of current parameters
            new_params = ema_strategy.parameters.copy()
            
            # Validate and update settings
            if 'check_interval' in settings:
                new_params['check_interval'] = int(settings['check_interval'])
            if 'continue_after_trade' in settings:
                new_params['continue_after_trade'] = bool(settings['continue_after_trade'])
            if 'max_concurrent_trades' in settings:
                new_params['max_concurrent_trades'] = max(1, min(5, int(settings['max_concurrent_trades'])))
            
            # Stop strategy if running
            was_running = ema_strategy._continue
            if was_running:
                ema_strategy.stop()
                time.sleep(1)  # Give it time to stop
            
            # Update parameters
            ema_strategy.parameters = new_params
            
            # Log updated state
            logger.info(f"Updated parameters: {ema_strategy.parameters}")
            
            # Restart if it was running
            if was_running:
                start_strategy(ema_strategy)
            
            return jsonify({
                'status': 'success',
                'message': 'Bot parameters updated successfully',
                'parameters': ema_strategy.parameters
            })
            
        elif bot_id == 'bb_strategy':
            # Validate settings
            if 'check_interval' in settings:
                settings['check_interval'] = int(settings['check_interval'])
            if 'continue_after_trade' in settings:
                settings['continue_after_trade'] = bool(settings['continue_after_trade'])
            if 'max_concurrent_trades' in settings:
                settings['max_concurrent_trades'] = max(1, min(5, int(settings['max_concurrent_trades'])))
            
            # Stop the strategy temporarily
            was_running = bb_strategy._continue
            if was_running:
                bb_strategy.stop()
            
            # Update settings
            bb_strategy.parameters.update(settings)
            
            # Restart if it was running
            if was_running:
                bb_strategy.start()
            
            return jsonify({
                'status': 'success',
                'message': 'Bot parameters updated successfully',
                'parameters': bb_strategy.parameters
            })
        
        return jsonify({
            'status': 'error',
            'message': f'Bot {bot_id} not found'
        }), 404
        
    except Exception as e:
        logger.error(f"Error updating bot parameters: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/api/bots/<bot_id>/status', methods=['GET'])
def get_bot_status(bot_id):
    try:
        if bot_id == 'bb_strategy' and bb_strategy:
            return jsonify({
                'status': 'success',
                'data': bb_strategy.get_status()
            })
        elif bot_id == 'ema_strategy' and ema_strategy:
            return jsonify({
                'status': 'success',
                'data': ema_strategy.get_status()
            })
            
        logger.error(f"Bot {bot_id} not found or not initialized")
        return jsonify({
            'status': 'error',
            'message': f'Bot {bot_id} not found or not initialized'
        }), 404
    except Exception as e:
        logger.error(f"Error getting bot status: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/debug/strategies')
def debug_strategies():
    return jsonify({
        'bb_strategy': {
            'initialized': bb_strategy is not None,
            'running': bb_strategy.should_continue() if bb_strategy else False
        },
        'ema_strategy': {
            'initialized': ema_strategy is not None,
            'running': ema_strategy.should_continue() if ema_strategy else False,
            'available_instruments': ema_strategy.available_instruments if ema_strategy else None
        }
    })

@app.route('/')
def index():
    return jsonify({
        'status': 'success',
        'message': 'Trading server is running'
    })

@app.route('/api/bots/<bot_id>/settings', methods=['POST'])
def update_bot_settings(bot_id):
    try:
        settings = request.get_json()
        
        if bot_id == 'ema_strategy':
            # Stop the strategy temporarily
            was_running = ema_strategy._continue
            if was_running:
                ema_strategy.stop()
            
            # Update settings
            ema_strategy.parameters.update({
                'check_interval': settings.get('check_interval', 240),
                'allow_multiple_trades': settings.get('allow_multiple_trades', False),
                'continue_after_trade': settings.get('continue_after_trade', True),
                'max_concurrent_trades': settings.get('max_concurrent_trades', 1)
            })
            
            # Restart if it was running
            if was_running:
                ema_strategy.start()
            
            return jsonify({
                'status': 'success',
                'message': 'Bot settings updated successfully'
            })
            
        return jsonify({
            'status': 'error',
            'message': f'Bot {bot_id} not found'
        }), 404
        
    except Exception as e:
        logger.error(f"Error updating bot settings: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

# Add CORS headers to all responses
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response

@app.route('/api/market-intelligence')
async def get_market_intelligence():
    try:
        analysis = await market_intelligence.get_market_analysis()
        return jsonify(analysis)
    except Exception as e:
        return jsonify({
            'error': True,
            'message': str(e)
        }), 500

@app.route('/api/economic-indicators')
def get_economic_indicators():
    try:
        indicators = market_intelligence.get_economic_indicators()
        return jsonify(indicators)
    except Exception as e:
        logger.error(f"Error getting economic indicators: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5002, debug=True) 