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
from strategies.ai_strategy import AIStrategy
import threading
import time
from services.market_intelligence_service import MarketIntelligenceService
import oandapyV20.endpoints.trades as trades

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

try:
    broker = OandaTrader(OANDA_CREDS)
    logger.info("Successfully initialized OANDA broker")
    
    # Initialize strategies
    bb_strategy = BBStrategy(broker, parameters={
        'symbol': 'BTC_USD',
        'check_interval': 240,
        'continue_after_trade': True,
        'max_concurrent_trades': 1,
        'risk_percent': 0.5  # Add risk_percent parameter (0.5%)
    })
    ema_strategy = EMATrendStrategy(broker=broker, parameters={
        'symbol': 'BTC_USD',
        'check_interval': 240,
        'continue_after_trade': True,
        'max_concurrent_trades': 1,
        'take_profit_level': 0.04,  # 4% take profit level
        'risk_percent': 0.5,  # 0.5% risk per trade for position sizing
        'trailing_stop_loss': False
    })
    
    # Initialize AI strategies for different symbols
    ai_gold_strategy = AIStrategy(broker=broker, parameters={
        'symbol': 'XAU_USD',  # Gold
        'quantity': 0.1,
        'check_interval': 1800,  # Check every 30 minutes
        'continue_after_trade': True,
        'max_concurrent_trades': 1,
        'trading_term': 'Day trade',
        'risk_level': 'conservative',
        'risk_percent': 0.5,  # Default to 0.5% risk per trade
        'trailing_stop_loss': False
    })
    
    ai_sp500_strategy = AIStrategy(broker=broker, parameters={
        'symbol': 'SPX500_USD',  # S&P 500
        'quantity': 0.1,
        'check_interval': 1800,  # Check every 30 minutes
        'continue_after_trade': True,
        'max_concurrent_trades': 1,
        'trading_term': 'Day trade',
        'risk_level': 'conservative',
        'risk_percent': 0.5,  # Default to 0.5% risk per trade
        'trailing_stop_loss': False
    })
    
    ai_btc_strategy = AIStrategy(broker=broker, parameters={
        'symbol': 'BTC_USD',  # Bitcoin
        'quantity': 0.1,
        'check_interval': 1800,  # Check every 30 minutes
        'continue_after_trade': True,
        'max_concurrent_trades': 1,
        'trading_term': 'Day trade',
        'risk_level': 'conservative',
        'risk_percent': 0.5,  # Default to 0.5% risk per trade
        'trailing_stop_loss': False
    })
    
    ai_eurusd_strategy = AIStrategy(broker=broker, parameters={
        'symbol': 'EUR_USD',  # EUR/USD
        'quantity': 0.1,
        'check_interval': 1800,  # Check every 30 minutes
        'continue_after_trade': True,
        'max_concurrent_trades': 1,
        'trading_term': 'Day trade',
        'risk_level': 'conservative',
        'risk_percent': 0.5,  # Default to 0.5% risk per trade
        'trailing_stop_loss': False
    })
    
    ai_nasdaq_strategy = AIStrategy(broker=broker, parameters={
        'symbol': 'NAS100_USD',  # Nasdaq
        'quantity': 0.1,
        'check_interval': 1800,  # Check every 30 minutes
        'continue_after_trade': True,
        'max_concurrent_trades': 1,
        'trading_term': 'Day trade',
        'risk_level': 'conservative',
        'risk_percent': 0.5,  # Default to 0.5% risk per trade
        'trailing_stop_loss': False
    })
    
    # Create a dictionary of all AI strategies for easier access
    ai_strategies = {
        'ai_gold_strategy': ai_gold_strategy,
        'ai_sp500_strategy': ai_sp500_strategy,
        'ai_btc_strategy': ai_btc_strategy,
        'ai_eurusd_strategy': ai_eurusd_strategy,
        'ai_nasdaq_strategy': ai_nasdaq_strategy
    }
    
    # Initialize market intelligence service
    market_intelligence = MarketIntelligenceService()
    
    logger.info("Successfully initialized strategies")
except Exception as e:
    logger.error(f"Error initializing broker or strategies: {e}", exc_info=True)
    broker = None
    bb_strategy = None
    ema_strategy = None
    ai_gold_strategy = None
    ai_sp500_strategy = None
    ai_btc_strategy = None
    ai_eurusd_strategy = None
    ai_nasdaq_strategy = None
    ai_strategies = {}

@app.route('/execute-trade', methods=['POST'])
def execute_trade():
    try:
        data = request.get_json()
        logger.info(f"Received trade request: {data}")
        
        symbol = data.get('symbol', 'EUR_USD')
        side = data.get('side', 'buy')
        quantity = data.get('quantity', 1000)
        order_type = data.get('order_type', 'market')
        price = data.get('price')  # Entry price for pending orders
        take_profit = data.get('take_profit')  # Take profit price
        stop_loss = data.get('stop_loss')  # Stop loss price
        
        logger.info(f"Processed request parameters: symbol={symbol}, side={side}, quantity={quantity}, order_type={order_type}, price={price}, take_profit={take_profit}, stop_loss={stop_loss}")
        
        # Check if market is open for this symbol
        if not broker.is_market_open(symbol):
            logger.error(f"Market is closed for {symbol}")
            return jsonify({
                'status': 'error',
                'message': f'Market is closed for {symbol}',
                'error_code': 'MARKET_HALTED'
            }), 400
        
        # For pending orders, we need a price
        if order_type == 'pending':
            if not price:
                logger.error("Price is required for pending orders")
                return jsonify({
                    'status': 'error',
                    'message': 'Price is required for pending orders'
                }), 400
                
            order = {
                'strategy': None,
                'symbol': symbol,
                'quantity': quantity,
                'side': side,
                'order_type': 'pending',
                'price': price,
                'take_profit': take_profit,
                'stop_loss': stop_loss,
                'position_fill': 'OPEN_ONLY'  # Prevent closing existing positions
            }
        else:
            # For market orders, just submit the order
            order = {
                'strategy': None,
                'symbol': symbol,
                'quantity': quantity,
                'side': side,
                'order_type': 'market',
                'take_profit': take_profit,
                'stop_loss': stop_loss,
                'position_fill': 'DEFAULT'  # Allow both opening and closing
            }
        
        logger.info(f"Submitting order to broker: {order}")
        try:
            order_id = broker.submit_order(order)
            logger.info(f"Broker response - order_id: {order_id}")
        except Exception as e:
            error_message = str(e)
            logger.error(f"Error submitting order: {error_message}", exc_info=True)
            
            # Check for specific error messages
            if "MARKET_HALTED" in error_message:
                return jsonify({
                    'status': 'error',
                    'message': f'Trading for {symbol} is currently halted. Please try again later or choose a different instrument.',
                    'error_code': 'MARKET_HALTED'
                }), 400
            
            return jsonify({
                'status': 'error',
                'message': f'Order submission error: {error_message}'
            }), 400
        
        if order_id:
            # For pending orders, don't expect an immediate position
            if order_type == 'pending':
                return jsonify({
                    'status': 'success',
                    'order_id': order_id,
                    'message': 'Pending order created successfully'
                }), 200
            
            # For market orders, check the position
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

@app.route('/close-position/<trade_id>', methods=['POST'])
def close_position(trade_id):
    try:
        # Get current positions before closing
        initial_positions = broker.get_tracked_positions()
        
        # Find the position with the given trade ID
        position = None
        for pos_key, pos_data in initial_positions.items():
            if pos_data['trade_id'] == trade_id:
                position = pos_data
                break
                
        if not position:
            return jsonify({
                'status': 'error',
                'message': 'No position found with the given trade ID'
            }), 404
            
        order = {
            'strategy': None,
            'symbol': position['symbol'],
            'quantity': abs(position['quantity']),
            'side': 'sell' if position['quantity'] > 0 else 'buy'
        }
        
        logger.info(f"Closing position for trade ID {trade_id}: {order}")
        order_id = broker.submit_order(order)
        
        # Verify position was actually closed by checking updated positions
        updated_positions = broker.get_tracked_positions()
        position_still_exists = False
        for pos_data in updated_positions.values():
            if pos_data['trade_id'] == trade_id:
                position_still_exists = True
                break
                
        if not position_still_exists:
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

@app.route('/trading-status')
def get_trading_status():
    try:
        # Get account info
        cash, positions_value, total_value = broker._get_balances_at_broker()
        
        # Get positions
        positions = broker.get_tracked_positions()
        
        # Get pending orders
        pending_orders = broker.get_pending_orders()
        
        # Get open trades to include take profit and stop loss information
        trades = broker.get_open_trades()
        
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
        
        return jsonify({
            'account': {
                'balance': cash,
                'unrealized_pl': positions_value,
                'total_value': total_value
            },
            'positions': positions,
            'pending_orders': pending_orders,
            'market_prices': market_prices,
            'market_status': {
                'is_market_open': broker.is_market_open(),
                'active_symbols': instruments
            },
            'trading_stats': {
                'win_rate': 0,  # Add real stats here
                'winning_trades': 0,
                'losing_trades': 0
            },
            'trades': trades  # Include trades in the response
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
    logger.info("Getting bots status...")
    
    # Check if strategies exist
    logger.info(f"EMA Strategy exists: {ema_strategy is not None}")
    logger.info(f"BB Strategy exists: {bb_strategy is not None}")
    for bot_id, strategy in ai_strategies.items():
        logger.info(f"{bot_id} Strategy exists: {strategy is not None}")
    
    response = {
        'status': 'success',
        'bots': {
            'ema_strategy': {
                'name': 'EMA Trend Strategy',
                'running': ema_strategy.should_continue() if ema_strategy else False,
                'parameters': ema_strategy.parameters if ema_strategy else {},
                'status': ema_strategy.get_status() if ema_strategy else {}
            },
            'bb_strategy': {
                'name': 'Bollinger Bands Strategy',
                'running': bb_strategy.should_continue() if bb_strategy else False,
                'parameters': bb_strategy.parameters if bb_strategy else {},
                'status': bb_strategy.get_status() if bb_strategy else {}
            },
            **{bot_id: {
                'name': f'AI Strategy for {strategy.parameters["symbol"]}',
                'running': strategy.should_continue() if strategy else False,
                'parameters': strategy.parameters if strategy else {},
                'status': strategy.get_status() if strategy else {}
            } for bot_id, strategy in ai_strategies.items()}
        }
    }
    
    logger.info(f"Returning bots: {response}")
    return jsonify(response)

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
                
        elif bot_id in ai_strategies:
            strategy_instance = ai_strategies[bot_id]
            if not strategy_instance:
                return jsonify({
                    'status': 'error',
                    'message': 'Strategy not initialized'
                }), 400
            
            was_running = strategy_instance.should_continue()
            if was_running:
                strategy_instance.stop()
                logger.info(f"{bot_id} strategy stopped")
                # Add a small delay to ensure the bot is fully stopped
                time.sleep(0.5)
                # Double-check that the bot is actually stopped
                if strategy_instance.should_continue():
                    logger.error(f"{bot_id} strategy failed to stop properly")
                    return jsonify({
                        'status': 'error',
                        'message': 'Failed to stop strategy'
                    }), 500
                new_status = 'stopped'
            else:
                # Always get a fresh analysis when starting
                strategy_instance.ai_analysis = None
                strategy_instance.last_analysis_time = None
                strategy_instance.current_symbol = strategy_instance.parameters['symbol']
                logger.info(f"Starting {bot_id} strategy with fresh analysis")
                start_strategy(strategy_instance)
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
        data = request.json
        logger.info(f"Updating parameters for {bot_id}: {data}")
        
        # Get the appropriate strategy based on bot_id
        if bot_id == 'bb_strategy':
            if not bb_strategy:
                return jsonify({'status': 'error', 'message': 'BB Strategy not initialized'}), 400
            strategy_instance = bb_strategy
        elif bot_id == 'ema_strategy':
            if not ema_strategy:
                return jsonify({'status': 'error', 'message': 'EMA Strategy not initialized'}), 400
            strategy_instance = ema_strategy
        elif bot_id in ai_strategies:
            strategy_instance = ai_strategies[bot_id]
            if not strategy_instance:
                return jsonify({'status': 'error', 'message': f'{bot_id} Strategy not initialized'}), 400
        else:
            return jsonify({'status': 'error', 'message': f'Bot {bot_id} not found'}), 404
        
        # Check if the bot is running - if so, prevent parameter changes
        if strategy_instance.should_continue():
            return jsonify({
                'status': 'error', 
                'message': 'Cannot change parameters while the bot is running. Please stop the bot first.'
            }), 400
        
        # Get the current parameters
        current_params = strategy_instance.parameters.copy()
        
        # Check if symbol is changing for AI strategy
        symbol_changed = False
        if bot_id in ai_strategies and 'symbol' in data and data['symbol'] != current_params.get('symbol'):
            symbol_changed = True
            logger.info(f"Symbol changing from {current_params.get('symbol')} to {data['symbol']}")
        
        # Update parameters
        for key, value in data.items():
            if key in current_params:
                # Handle numeric values
                if isinstance(current_params[key], (int, float)) and isinstance(value, (int, float)):
                    current_params[key] = value
                # Handle boolean values
                elif isinstance(current_params[key], bool) and isinstance(value, bool):
                    current_params[key] = value
                # Handle string values
                elif isinstance(current_params[key], str) and isinstance(value, str):
                    current_params[key] = value
                else:
                    # For any other type, just update it
                    current_params[key] = value
        
        # Update the strategy parameters
        strategy_instance.parameters = current_params
        
        # For AI strategy, if symbol changed, force a new analysis on next check
        if symbol_changed:
            logger.info(f"Symbol changed, forcing new analysis on next check")
            # Clear the current analysis data
            if hasattr(strategy_instance, 'ai_analysis'):
                strategy_instance.ai_analysis = None
                strategy_instance.last_analysis_time = None
                strategy_instance.current_symbol = current_params['symbol']
        
        logger.info(f"Updated parameters for {bot_id}: {current_params}")
        return jsonify({'status': 'success', 'message': 'Parameters updated', 'parameters': current_params})
    
    except Exception as e:
        logger.error(f"Error updating parameters for {bot_id}: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500

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
        elif bot_id in ai_strategies:
            strategy_instance = ai_strategies[bot_id]
            if strategy_instance:
                return jsonify({
                    'status': 'success',
                    'data': strategy_instance.get_status()
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
        },
        **{bot_id: {
            'initialized': strategy is not None,
            'running': strategy.should_continue() if strategy else False
        } for bot_id, strategy in ai_strategies.items()}
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
            # Check if the bot is running - if so, prevent settings changes
            if ema_strategy.should_continue():
                return jsonify({
                    'status': 'error', 
                    'message': 'Cannot change settings while the bot is running. Please stop the bot first.'
                }), 400
            
            # Update settings
            ema_strategy.parameters.update({
                'check_interval': settings.get('check_interval', 240),
                'allow_multiple_trades': settings.get('allow_multiple_trades', False),
                'continue_after_trade': settings.get('continue_after_trade', True),
                'max_concurrent_trades': settings.get('max_concurrent_trades', 1)
            })
            
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

@app.route('/cancel-order/<order_id>', methods=['POST'])
def cancel_order(order_id):
    try:
        logger.info(f"Canceling order: {order_id}")
        
        # Cancel the order using the broker
        broker.cancel_order(order_id)
        
        return jsonify({
            'status': 'success',
            'message': 'Order canceled successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error canceling order: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/modify-position', methods=['POST'])
def modify_position():
    try:
        data = request.get_json()
        logger.info(f"Received position modification request: {data}")
        
        trade_id = data.get('trade_id')
        take_profit = data.get('take_profit')
        stop_loss = data.get('stop_loss')
        
        if not trade_id:
            return jsonify({
                'status': 'error',
                'message': 'Trade ID is required'
            }), 400
            
        # Get trade details from OANDA
        r = trades.TradeDetails(accountID=broker.account_id, tradeID=trade_id)
        response = broker.api.request(r)
        
        if 'trade' not in response:
            return jsonify({
                'status': 'error',
                'message': 'Trade not found'
            }), 404
            
        trade = response['trade']
        
        # Prepare modification data
        data = {}
        if take_profit is not None:
            data["takeProfit"] = {
                "price": str(take_profit),
                "timeInForce": "GTC"
            }
        if stop_loss is not None:
            data["stopLoss"] = {
                "price": str(stop_loss),
                "timeInForce": "GTC"
            }
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No modifications specified'
            }), 400
            
        # Submit modification request
        r = trades.TradeCRCDO(accountID=broker.account_id, tradeID=trade_id, data=data)
        response = broker.api.request(r)
        
        if 'errorMessage' in response:
            return jsonify({
                'status': 'error',
                'message': response['errorMessage']
            }), 400
            
        return jsonify({
            'status': 'success',
            'message': 'Position modified successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error modifying position: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/test-trade', methods=['GET'])
def test_trade():
    """Test endpoint to execute a trade with specific stop loss and take profit values"""
    try:
        # Parameters for the Nasdaq SHORT trade with realistic price levels
        symbol = 'NAS100_USD'
        side = 'sell'
        quantity = 0.1
        take_profit = 20500.0  # Realistic take profit for Nasdaq
        stop_loss = 21100.0    # Realistic stop loss for Nasdaq
        
        logger.info(f"Executing test trade: {symbol} {side} {quantity} with TP={take_profit} and SL={stop_loss}")
        
        # Create order
        order = {
            'strategy': None,
            'symbol': symbol,
            'quantity': quantity,
            'side': side,
            'order_type': 'market',
            'take_profit': take_profit,
            'stop_loss': stop_loss,
            'position_fill': 'DEFAULT'
        }
        
        logger.info(f"Test order details: {order}")
        
        # Submit the order
        order_id = broker.submit_order(order)
        
        if order_id:
            return jsonify({
                'status': 'success',
                'message': 'Test trade executed successfully',
                'order_id': order_id
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Test trade failed'
            }), 400
            
    except Exception as e:
        logger.error(f"Error in test trade: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/reset-ai-strategy', methods=['GET'])
def reset_ai_strategy():
    """Reset the AI strategy by clearing its analysis data"""
    try:
        if not ai_gold_strategy:
            return jsonify({
                'status': 'error',
                'message': 'AI Strategy not initialized'
            }), 400
            
        # Stop the strategy if it's running
        was_running = ai_gold_strategy.should_continue()
        if was_running:
            ai_gold_strategy.stop()
            
        # Reset the analysis data
        ai_gold_strategy.ai_analysis = None
        ai_gold_strategy.last_analysis_time = None
        ai_gold_strategy.current_symbol = ai_gold_strategy.parameters['symbol']
        
        logger.info(f"Reset AI strategy analysis data for {ai_gold_strategy.parameters['symbol']}")
        
        # Restart if it was running
        if was_running:
            start_strategy(ai_gold_strategy)
            status = 'restarted'
        else:
            status = 'reset'
            
        return jsonify({
            'status': 'success',
            'message': f'AI Strategy {status} successfully',
            'symbol': ai_gold_strategy.parameters['symbol']
        })
            
    except Exception as e:
        logger.error(f"Error resetting AI strategy: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/set-manual-analysis', methods=['GET'])
def set_manual_analysis():
    """Set manual analysis data for the AI strategy"""
    try:
        if not ai_gold_strategy:
            return jsonify({
                'status': 'error',
                'message': 'AI Strategy not initialized'
            }), 400
            
        # Get parameters from query string with defaults for Nasdaq
        symbol = request.args.get('symbol', 'NAS100_USD')
        direction = request.args.get('direction', 'SHORT')
        entry_price = float(request.args.get('entry', '20888.3'))
        take_profit = float(request.args.get('tp', '20500.0'))
        stop_loss = float(request.args.get('sl', '21100.0'))
        
        # Update the strategy parameters if symbol is different
        if symbol != ai_gold_strategy.parameters['symbol']:
            ai_gold_strategy.parameters['symbol'] = symbol
            ai_gold_strategy.current_symbol = symbol
            logger.info(f"Updated AI strategy symbol to {symbol}")
        
        # Set the manual analysis
        success = ai_gold_strategy.set_manual_analysis(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            take_profit=take_profit,
            stop_loss=stop_loss
        )
        
        if success:
            # Make sure the strategy is running
            if not ai_gold_strategy.should_continue():
                start_strategy(ai_gold_strategy)
                logger.info("Started AI strategy")
            
            return jsonify({
                'status': 'success',
                'message': 'Manual analysis set successfully',
                'data': {
                    'symbol': symbol,
                    'direction': direction,
                    'entry_price': entry_price,
                    'take_profit': take_profit,
                    'stop_loss': stop_loss
                }
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to set manual analysis'
            }), 400
            
    except Exception as e:
        logger.error(f"Error setting manual analysis: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/bots/<bot_id>/clear-logs', methods=['POST'])
def clear_bot_logs(bot_id):
    """Clear logs for a specific bot - only when explicitly requested"""
    logger.info(f"Received request to clear logs for bot: {bot_id}")
    
    try:
        if bot_id == 'bollinger_bands':
            if not bb_strategy:
                logger.error("Bollinger Bands strategy not initialized")
                return jsonify({
                    'status': 'error',
                    'message': 'Strategy not initialized'
                }), 400
                
            result = bb_strategy.clear_logs()
            logger.info(f"Cleared logs for Bollinger Bands strategy, result: {result}")
            
        elif bot_id == 'ema_strategy':
            if not ema_strategy:
                logger.error("EMA strategy not initialized")
                return jsonify({
                    'status': 'error',
                    'message': 'Strategy not initialized'
                }), 400
                
            result = ema_strategy.clear_logs()
            logger.info(f"Cleared logs for EMA strategy, result: {result}")
                
        elif bot_id in ai_strategies:
            strategy_instance = ai_strategies[bot_id]
            if strategy_instance:
                result = strategy_instance.clear_logs()
                logger.info(f"Cleared logs for AI strategy {bot_id}, result: {result}")
            else:
                logger.error(f"AI strategy {bot_id} exists but instance is None")
                return jsonify({
                    'status': 'error',
                    'message': f'Strategy {bot_id} not properly initialized'
                }), 400
        else:
            logger.error(f"Unknown bot ID: {bot_id}")
            return jsonify({
                'status': 'error',
                'message': f'Unknown bot ID: {bot_id}'
            }), 404
            
        return jsonify({
            'status': 'success',
            'message': f'Logs cleared for {bot_id}'
        })
        
    except Exception as e:
        logger.error(f"Error clearing logs for {bot_id}: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Error clearing logs: {str(e)}'
        }), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5002, debug=True) 