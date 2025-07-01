from quart import Quart, request, jsonify
from quart_cors import cors
import asyncio
import logging
import sys
import os
import jwt
from datetime import datetime
from functools import wraps

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from master_trading_bot import MasterTradingBot
from routes.ai_analysis_routes import ai_analysis_bp
from services.supabase_service import SupabaseService
from config import validate_api_keys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Create Quart app
app = Quart(__name__)
app = cors(app)

# Global instances
master_bot = None
supabase_service = None
custom_bot_service = None

# Authentication decorator
def require_auth(f):
    """Require JWT authentication for route"""
    @wraps(f)
    async def decorated_function(*args, **kwargs):
        token = None
        
        # Check Authorization header
        auth_header = request.headers.get('Authorization')
        if auth_header:
            try:
                token = auth_header.split(' ')[1]  # Bearer <token>
            except IndexError:
                return jsonify({'status': 'error', 'message': 'Invalid token format'}), 401
        
        if not token:
            return jsonify({'status': 'error', 'message': 'Authentication token required'}), 401
        
        try:
            # For demo purposes, we'll use a simple JWT without verification
            # In production, you should verify the JWT signature with Supabase
            payload = jwt.decode(token, options={"verify_signature": False})
            request.user_id = payload.get('sub')
            
            if not request.user_id:
                return jsonify({'status': 'error', 'message': 'Invalid token'}), 401
                
        except jwt.DecodeError:
            return jsonify({'status': 'error', 'message': 'Invalid token'}), 401
        
        return await f(*args, **kwargs)
    
    return decorated_function

# Optional auth decorator (allows access without auth but sets user_id if present)
def optional_auth(f):
    """Optional JWT authentication for route"""
    @wraps(f)
    async def decorated_function(*args, **kwargs):
        request.user_id = None
        
        auth_header = request.headers.get('Authorization')
        if auth_header:
            try:
                token = auth_header.split(' ')[1]
                payload = jwt.decode(token, options={"verify_signature": False})
                request.user_id = payload.get('sub')
            except (IndexError, jwt.DecodeError):
                pass  # Continue without auth
        
        return await f(*args, **kwargs)
    
    return decorated_function

@app.before_serving
async def startup():
    """Initialize services on startup"""
    global supabase_service, custom_bot_service
    
    logger.info("🚀 Starting Master Trading Server...")
    
    try:
        # Validate API keys
        validate_api_keys()
        
        # Initialize Supabase service
        try:
            supabase_service = SupabaseService()
            logger.info("✅ Supabase service initialized")
        except Exception as e:
            logger.warning(f"⚠️ Supabase service not available: {e}")
            supabase_service = None
        
        # Custom Bot Service will be initialized on-demand
        custom_bot_service = None
        logger.info("ℹ️ Custom Bot Service will be initialized on-demand")
        
        logger.info("✅ Master Trading Server initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize server: {e}")
        raise

def get_or_create_custom_bot_service():
    """Get or create CustomBotService on-demand"""
    global custom_bot_service
    
    if custom_bot_service is None:
        try:
            from oanda_trader import OandaTrader
            from services.custom_bot_service import CustomBotService
            from config.oanda_config import OANDA_CREDS
            
            # Create broker instance for custom bots with proper credentials
            broker = OandaTrader(OANDA_CREDS)
            custom_bot_service = CustomBotService(broker, supabase_service)
            logger.info("✅ Custom Bot Service initialized on-demand")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Custom Bot Service: {e}")
            raise Exception(f"Failed to initialize execution service: {e}")
    
    return custom_bot_service

# Master Bot Routes
@app.route('/api/master-bot/start', methods=['POST'])
async def start_master_bot():
    """Start the Master Trading Bot (deprecated - use analyze-and-start)"""
    global master_bot
    
    try:
        if master_bot and master_bot.is_running:
            return jsonify({
                'status': 'error',
                'message': 'Master bot is already running'
            }), 400
        
        # Create new bot instance if needed
        if not master_bot:
            master_bot = MasterTradingBot()
        
        # Check if it's weekend
        if master_bot.is_weekend():
            return jsonify({
                'status': 'error',
                'message': '🚫 Weekend trading is disabled. Markets are closed on Saturday and Sunday.',
                'weekend_warning': True
            }), 400
        
        # Check if we have analysis
        if not master_bot.asset_rankings:
            return jsonify({
                'status': 'error',
                'message': 'No market analysis available. Please run analysis first.',
                'need_analysis': True
            }), 400
        
        # Start bot in background
        asyncio.create_task(master_bot.start())
        
        return jsonify({
            'status': 'success',
            'message': 'Master Trading Bot started successfully',
            'bot_config': master_bot.risk_config
        })
        
    except Exception as e:
        logger.error(f"Error starting master bot: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to start master bot: {str(e)}'
        }), 500

@app.route('/api/master-bot/analyze-and-start', methods=['POST'])
async def analyze_and_start_master_bot():
    """Analyze markets and start the Master Trading Bot"""
    global master_bot
    
    try:
        if master_bot and master_bot.is_running:
            return jsonify({
                'status': 'error',
                'message': 'Master bot is already running'
            }), 400
        
        # Create new bot instance if needed
        if not master_bot:
            master_bot = MasterTradingBot()
        
        # Check if it's weekend
        if master_bot.is_weekend():
            return jsonify({
                'status': 'error',
                'message': '🚫 Weekend trading is disabled. Markets are closed on Saturday and Sunday.',
                'weekend_warning': True
            }), 400
        
        # Run analysis first
        logger.info("🔍 Running market analysis before starting bot...")
        rankings = await master_bot.analyze_all_assets()
        
        # Check if we have good opportunities
        good_opportunities = sum(1 for v in rankings.values() if v['profit_score'] >= 50)
        if good_opportunities == 0:
            return jsonify({
                'status': 'error',
                'message': f'No profitable opportunities found. All {len(rankings)} assets have low profit scores.',
                'analysis_completed': True,
                'asset_count': len(rankings)
            }), 400
        
        # Start bot in background
        asyncio.create_task(master_bot.start())
        
        return jsonify({
            'status': 'success',
            'message': f'Analysis complete! Found {good_opportunities} good opportunities. Bot started successfully.',
            'data': {
                'bot_config': master_bot.risk_config,
                'assets_analyzed': len(rankings),
                'good_opportunities': good_opportunities,
                'asset_rankings': {k: v['profit_score'] for k, v in rankings.items()}
            }
        })
        
    except Exception as e:
        logger.error(f"Error analyzing and starting master bot: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to analyze and start bot: {str(e)}'
        }), 500

@app.route('/api/master-bot/stop', methods=['POST'])
async def stop_master_bot():
    """Stop the Master Trading Bot"""
    global master_bot
    
    try:
        if not master_bot:
            return jsonify({
                'status': 'error',
                'message': 'Master bot is not running'
            }), 400
        
        master_bot.stop()
        
        return jsonify({
            'status': 'success',
            'message': 'Master Trading Bot stopped successfully'
        })
        
    except Exception as e:
        logger.error(f"Error stopping master bot: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to stop master bot: {str(e)}'
        }), 500

@app.route('/api/master-bot/status', methods=['GET'])
async def get_master_bot_status():
    """Get current Master Trading Bot status"""
    global master_bot
    
    try:
        if not master_bot:
            return jsonify({
                'status': 'success',
                'data': {
                    'is_running': False,
                    'message': 'Master bot is not initialized'
                }
            })
        
        status = master_bot.get_status()
        
        return jsonify({
            'status': 'success',
            'data': status
        })
        
    except Exception as e:
        logger.error(f"Error getting master bot status: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to get status: {str(e)}'
        }), 500

@app.route('/api/master-bot/analyze', methods=['POST'])
async def analyze_markets():
    """Trigger market analysis manually"""
    global master_bot
    
    try:
        # Always ensure we have a master bot instance
        if not master_bot:
            logger.info("🔄 Creating Master Bot instance for analysis...")
            master_bot = MasterTradingBot()
        
        # Run analysis
        rankings = await master_bot.analyze_all_assets()
        
        # Create enhanced asset rankings with summaries
        enhanced_rankings = {}
        for k, v in rankings.items():
            ai_analysis = v.get('ai_analysis', {})
            enhanced_rankings[k] = {
                'score': v['profit_score'],
                'summary': ai_analysis.get('market_summary', f'{k} analysis'),
                'direction': ai_analysis.get('trading_strategy', {}).get('direction', 'NEUTRAL'),
                'key_drivers': ai_analysis.get('key_drivers', []),
                'risk_assessment': ai_analysis.get('risk_assessment', 'Standard risk')
            }
        
        return jsonify({
            'status': 'success',
            'message': f'Analyzed {len(rankings)} assets',
            'data': {
                'asset_rankings': {k: v['score'] for k, v in enhanced_rankings.items()},
                'asset_details': enhanced_rankings,
                'analysis_time': master_bot.last_analysis_time.isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Error analyzing markets: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to analyze markets: {str(e)}'
        }), 500

@app.route('/api/master-bot/config', methods=['POST'])
async def update_bot_config():
    """Update bot risk configuration"""
    global master_bot
    
    try:
        data = await request.get_json()
        
        if not master_bot:
            return jsonify({
                'status': 'error',
                'message': 'Master bot is not initialized'
            }), 400
        
        # Update risk config
        if 'risk_config' in data:
            master_bot.risk_config.update(data['risk_config'])
        
        return jsonify({
            'status': 'success',
            'message': 'Configuration updated successfully',
            'data': {
                'risk_config': master_bot.risk_config
            }
        })
        
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to update config: {str(e)}'
        }), 500

@app.route('/api/master-bot/force-trade', methods=['POST'])
async def force_trade():
    """Force execute the best available trade"""
    global master_bot
    
    try:
        if not master_bot:
            return jsonify({
                'status': 'error',
                'message': 'Master bot is not initialized'
            }), 400
        
        if master_bot.current_position:
            return jsonify({
                'status': 'error',
                'message': 'Bot already has an active position'
            }), 400
        
        # Force analyze and trade
        await master_bot.analyze_all_assets()
        best_trade = await master_bot.select_best_trade()
        
        if not best_trade:
            return jsonify({
                'status': 'error',
                'message': 'No suitable trading opportunities found'
            }), 400
        
        success = await master_bot.execute_trade(best_trade)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': 'Trade executed successfully',
                'data': {
                    'trade': best_trade,
                    'position': master_bot.current_position
                }
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to execute trade'
            }), 500
        
    except Exception as e:
        logger.error(f"Error forcing trade: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to force trade: {str(e)}'
        }), 500

@app.route('/api/master-bot/performance', methods=['GET'])
async def get_performance():
    """Get detailed performance metrics"""
    global master_bot
    
    try:
        if not master_bot:
            return jsonify({
                'status': 'error',
                'message': 'Master bot is not initialized'
            }), 400
        
        return jsonify({
            'status': 'success',
            'data': {
                'performance_metrics': master_bot.performance_metrics,
                'trade_history': master_bot.trade_history,
                'total_trades': len(master_bot.trade_history),
                'current_position': master_bot.current_position
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting performance: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to get performance: {str(e)}'
        }), 500

# Trading status endpoint (for backward compatibility)
@app.route('/trading-status', methods=['GET'])
async def get_trading_status():
    """Get trading status - compatibility endpoint for MarketOverview"""
    global master_bot
    
    try:
        # Always ensure we have a master bot instance for account data
        if not master_bot:
            master_bot = MasterTradingBot()
        
        # Get account balances from broker
        try:
            cash, unrealized_pl, total_value = master_bot.broker._get_balances_at_broker()
        except Exception:
            # Fallback values if broker is not available
            cash = 10000.0
            unrealized_pl = 0.0
            total_value = 10000.0
        
        # Get current market prices
        market_prices = {}
        for symbol in master_bot.instruments.keys():
            try:
                price = master_bot.broker.get_last_price(symbol)
                if price:
                    market_prices[symbol] = {'price': price}
            except Exception:
                # Skip symbols that can't get prices
                continue
        
        # Get actual positions from OANDA broker
        positions = {}
        try:
            # Get all open positions from broker
            oanda_positions = master_bot.broker.get_tracked_positions()
            
            for position_id, position_data in oanda_positions.items():
                # Use the correct field names from get_tracked_positions()
                symbol = position_data.get('symbol', '')
                quantity = position_data.get('quantity', 0)
                entry_price = position_data.get('entry_price', 0)
                current_price = position_data.get('current_price', entry_price)
                pl_euro = position_data.get('pl_euro', 0)
                profit_pct = position_data.get('profit_pct', 0)
                side = position_data.get('side', 'LONG')
                trade_id = position_data.get('trade_id', position_id)
                
                if abs(quantity) > 0:  # Only include positions with actual quantity
                    positions[position_id] = {
                        'symbol': symbol,
                        'trade_id': trade_id,
                        'side': side,
                        'quantity': abs(quantity),
                        'entry_price': entry_price,
                        'current_price': current_price,
                        'take_profit': None,  # OANDA doesn't provide TP in position data
                        'stop_loss': None,    # OANDA doesn't provide SL in position data
                        'pl_euro': pl_euro,
                        'profit_pct': profit_pct
                    }
        except Exception as e:
            logger.error(f"Error getting OANDA positions: {e}")
            # Fallback to master bot position if OANDA fails
            if master_bot.current_position:
                pos = master_bot.current_position
                position_key = f"{pos['symbol']}_{pos['entry_time'].strftime('%Y%m%d_%H%M%S')}"
                positions[position_key] = {
                    'symbol': pos['symbol'],
                    'trade_id': pos['order_id'],
                    'side': pos['direction'],
                    'quantity': pos['position_size'],
                    'entry_price': pos['entry_price'],
                    'current_price': market_prices.get(pos['symbol'], {}).get('price', pos['entry_price']),
                    'take_profit': pos.get('take_profit'),
                    'stop_loss': pos.get('stop_loss'),
                    'pl_euro': (market_prices.get(pos['symbol'], {}).get('price', pos['entry_price']) - pos['entry_price']) * pos['position_size'],
                    'profit_pct': ((market_prices.get(pos['symbol'], {}).get('price', pos['entry_price']) - pos['entry_price']) / pos['entry_price']) * 100 if pos['entry_price'] > 0 else 0
                }
        
        return jsonify({
            'account': {
                'balance': cash,
                'unrealized_pl': unrealized_pl,
                'total_value': total_value
            },
            'market_prices': market_prices,
            'positions': positions,
            'pending_orders': [],  # Master bot doesn't use pending orders currently
            'trades': [],  # Simplified for now
            'bot_status': 'running' if master_bot.is_running else 'stopped',
            'active_positions': list(positions.values()) if positions else [],
            'total_trades': master_bot.performance_metrics.get('total_trades', 0),
            'win_rate': master_bot.performance_metrics.get('win_rate', 0)
        })
        
    except Exception as e:
        logger.error(f"Error getting trading status: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to get trading status: {str(e)}'
        }), 500

# Health check
@app.route('/api/health', methods=['GET'])
async def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'success',
        'message': 'Master Trading Server is running',
        'bot_status': 'running' if (master_bot and master_bot.is_running) else 'stopped'
    })

# Trading endpoints for MarketOverview frontend compatibility
@app.route('/execute-trade', methods=['POST'])
async def execute_trade():
    """Execute a trade order"""
    global master_bot
    
    try:
        # Always ensure we have a master bot instance for broker access
        if not master_bot:
            master_bot = MasterTradingBot()
        
        data = await request.get_json()
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
        if not master_bot.broker.is_market_open(symbol):
            logger.error(f"Market is closed for {symbol}")
            return jsonify({
                'status': 'error',
                'message': f'Market is closed for {symbol}',
                'error_code': 'MARKET_HALTED'
            }), 400
        
        # Build order object
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
            # For market orders
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
            order_id = master_bot.broker.submit_order(order)
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
            positions = master_bot.broker.get_tracked_positions()
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
async def close_position(trade_id):
    """Close a specific position"""
    global master_bot
    
    try:
        # Always ensure we have a master bot instance for broker access
        if not master_bot:
            master_bot = MasterTradingBot()
        
        # Get current positions before closing
        initial_positions = master_bot.broker.get_tracked_positions()
        
        # Find the position with the given trade ID
        position = None
        for pos_key, pos_data in initial_positions.items():
            if pos_data.get('trade_id') == trade_id:
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
        order_id = master_bot.broker.submit_order(order)
        
        # Verify position was actually closed by checking updated positions
        updated_positions = master_bot.broker.get_tracked_positions()
        position_still_exists = False
        for pos_data in updated_positions.values():
            if pos_data.get('trade_id') == trade_id:
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

@app.route('/cancel-order/<order_id>', methods=['POST'])
async def cancel_order(order_id):
    """Cancel a pending order"""
    global master_bot
    
    try:
        # Always ensure we have a master bot instance for broker access
        if not master_bot:
            master_bot = MasterTradingBot()
        
        logger.info(f"Canceling order: {order_id}")
        success = master_bot.broker.cancel_order(order_id)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': 'Order canceled successfully'
            }), 200
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to cancel order'
            }), 400
            
    except Exception as e:
        logger.error(f"Error canceling order: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/modify-position', methods=['POST'])
async def modify_position():
    """Modify a position's take profit and stop loss"""
    global master_bot
    
    try:
        # Always ensure we have a master bot instance for broker access
        if not master_bot:
            master_bot = MasterTradingBot()
        
        data = await request.get_json()
        trade_id = data.get('trade_id')
        take_profit = data.get('take_profit')
        stop_loss = data.get('stop_loss')
        
        logger.info(f"Modifying position {trade_id}: TP={take_profit}, SL={stop_loss}")
        
        # Get current positions to find the trade
        positions = master_bot.broker.get_tracked_positions()
        position = None
        for pos_data in positions.values():
            if pos_data.get('trade_id') == trade_id:
                position = pos_data
                break
                
        if not position:
            return jsonify({
                'status': 'error',
                'message': 'Position not found'
            }), 404
        
        # Note: OANDA doesn't support modifying TP/SL on existing positions directly
        # This would require closing the position and opening a new one with TP/SL
        # For now, return a message indicating this limitation
        return jsonify({
            'status': 'error',
            'message': 'Position modification not supported by OANDA. Please close the position and create a new one with desired TP/SL.'
        }), 400
        
    except Exception as e:
        logger.error(f"Error modifying position: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# Register existing blueprints (for backward compatibility)
app.register_blueprint(ai_analysis_bp)

@app.route('/ai-chat', methods=['POST'])
async def ai_chat():
    """AI Finance Expert Chat with web search capabilities"""
    global master_bot
    
    try:
        data = await request.get_json()
        user_message = data.get('message', '')
        conversation_history = data.get('conversation_history', [])
        
        logger.info(f"AI Chat request: {user_message}")
        
        # Always ensure we have a master bot instance for market data access
        if not master_bot:
            master_bot = MasterTradingBot()
        
        # Initialize Google Search Service
        from services.google_search_service import GoogleSearchService
        search_service = GoogleSearchService()
        
        # Check if we need to perform web search
        search_results = []
        web_context = ""
        
        if search_service.should_search_web(user_message):
            logger.info(f"Performing web search for query: {user_message}")
            search_results = search_service.search_financial_news(user_message)
            if search_results:
                web_context = search_service.format_search_context(search_results)
                logger.info(f"Found {len(search_results)} search results")
        
        # Always get user's position data for personalized advice
        user_positions = {}
        account_balance = 0
        total_unrealized_pl = 0
        
        try:
            if master_bot and hasattr(master_bot, 'broker'):
                # Get account balance using the correct method
                cash, unrealized_pl, total_value = master_bot.broker._get_balances_at_broker()
                account_balance = cash
                total_unrealized_pl = unrealized_pl
                
                # Get current positions
                positions = master_bot.broker.get_tracked_positions()
                if positions:
                    user_positions = positions
        except Exception as e:
            logger.warning(f"Could not fetch position data: {e}")
        
        logger.info(f"AI Chat request with user positions: {len(user_positions)} positions")
        
        # Create enhanced finance expert prompt with web context
        base_prompt = f"""You are a highly experienced and knowledgeable financial advisor and market strategist. You provide decisive, specific analysis and actionable recommendations based on your expertise.

RESPONSE REQUIREMENTS:
1. **Be Decisive**: Give specific recommendations based on your analysis - no "if you think" or "monitor upcoming" language
2. **Actionable**: State what you would specifically do and why
3. **Professional**: Provide confident expert analysis
4. **Current**: When web search results are provided, integrate them into your analysis for the most up-to-date insights

QUESTION TYPES:
- **General Market Questions** (e.g., "What's happening in markets today?"): Focus on BROAD analysis across ALL major asset classes (stocks, bonds, currencies, commodities, crypto). Only briefly mention user's position if directly relevant.
- **Position-Specific Questions** (e.g., "Should I hold my EUR/USD trade?"): Focus primarily on their actual trades and P/L with detailed analysis.
- **Educational Questions**: Focus on teaching concepts without necessarily referencing positions.

Recent Conversation:
{format_conversation_history(conversation_history)}

{web_context}

Question: {user_message}

Account Status (reference only when relevant to their specific question):
- Balance: {account_balance:,.2f}€
- Unrealized P/L: {total_unrealized_pl:+.2f}€
- Active Trades: {len(user_positions)}
{format_user_positions(user_positions) if user_positions else 'No open positions currently'}

Provide specific, actionable analysis based on your financial expertise and any current market information provided above:"""

        # Call AI service for response
        try:
            # Create a simple AI service for chat responses
            from services.simple_ai_service import SimpleAIService
            ai_service = SimpleAIService()
            
            # Use OpenAI directly for chat response
            if ai_service.client:
                response = ai_service.client.chat.completions.create(
                    model="gpt-4o",  # Use full gpt-4o for enhanced capabilities
                    messages=[{
                        "role": "user", 
                        "content": base_prompt
                    }],
                    max_tokens=1000,  # Increased for more detailed responses
                    temperature=0.3
                )
                response_text = response.choices[0].message.content
                
                # Use search results as sources instead of extracting from response
                extracted_sources = search_service.extract_sources_for_frontend(search_results) if search_results else []
                
            else:
                # Simple error message
                response_text = "I apologize, but I'm unable to access the AI service at the moment. Please try again in a few moments."
                extracted_sources = []
                
        except Exception as e:
            logger.error(f"AI service error: {e}")
            response_text = "I apologize, but I'm experiencing technical difficulties. Please try again in a moment."
            extracted_sources = []
        
        # Format response
        response_data = {
            'response': response_text,
            'sources': extracted_sources,
            'timestamp': datetime.now().isoformat(),
            'web_search_performed': len(search_results) > 0
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in AI chat: {e}", exc_info=True)
        return jsonify({
            'response': "I apologize, but I'm experiencing technical difficulties. Please try again in a moment.",
            'sources': [],
            'timestamp': datetime.now().isoformat(),
            'web_search_performed': False
        }), 500



def format_conversation_history(history):
    """Format conversation history for AI context"""
    if not history:
        return "This is the start of the conversation."
    
    formatted = []
    for msg in history[-3:]:  # Last 3 messages for context
        role = "User" if msg.get('type') == 'user' else "AI"
        content = msg.get('content', '')[:200]  # Limit length
        formatted.append(f"{role}: {content}")
    
    return '\n'.join(formatted)

def format_user_positions(positions):
    """Format user positions for AI context"""
    if not positions:
        return "No open positions currently"
    
    formatted = []
    for pos_id, pos in positions.items():
        symbol = pos.get('symbol', 'Unknown')
        side = pos.get('side', 'Unknown')
        quantity = pos.get('quantity', 0)
        entry_price = pos.get('entry_price', 0)
        current_price = pos.get('current_price', 0)
        pl_euro = pos.get('pl_euro', 0)
        profit_pct = pos.get('profit_pct', 0)
        
        formatted.append(f"• {symbol} {side}: {quantity:,.0f} units @ {entry_price:.5f} → {current_price:.5f} | P/L: {pl_euro:+.2f}€ ({profit_pct:+.1f}%)")
    
    return '\n'.join(formatted)



# Google CSE integration functions moved to services/google_search_service.py

# ================================
# CUSTOM BOT ROUTES (MULTI-USER)
# ================================

@app.route('/api/custom-bots/', methods=['GET'])
# @require_auth  # Temporarily disabled for testing
async def get_all_bots():
    """Get all custom bots for the authenticated user"""
    try:
        if not supabase_service:
            return jsonify({
                'status': 'error',
                'message': 'Database service not available'
            }), 503
        
        # Temporary test user ID for development (proper UUID format)
        user_id = getattr(request, 'user_id', '11111111-1111-1111-1111-111111111111')
        bots = supabase_service.get_user_bots(user_id)
        
        # Convert to JSON-serializable format with performance data
        bots_data = []
        for bot in bots:
            # Get performance metrics
            performance = supabase_service.get_bot_performance(user_id, bot['id'])
            
            bot_data = {
                'id': bot['id'],
                'name': bot['name'],
                'description': bot['description'],
                'instruments': bot['instruments'],
                'risk_level': bot['risk_level'],
                'execution_interval': bot['execution_interval'],
                'trailing_stop_type': bot['trailing_stop_type'],
                'trailing_stop_pips': bot['trailing_stop_pips'],
                'status': bot['status'],
                'created_at': bot['created_at'],
                'updated_at': bot['updated_at'],
                'last_run': bot['last_run'],
                'performance': performance or {'total_trades': 0, 'win_rate': 0, 'total_pnl': 0},
                'error_count': bot['error_count'],
                'last_error': bot['last_error']
            }
            bots_data.append(bot_data)
        
        return jsonify({
            'status': 'success',
            'bots': bots_data,
            'count': len(bots_data)
        })
        
    except Exception as e:
        logger.error(f"Error getting bots: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/custom-bots/', methods=['POST'])
# @require_auth  # Temporarily disabled for testing
async def create_bot():
    """Create a new custom bot"""
    try:
        if not supabase_service:
            return jsonify({
                'status': 'error',
                'message': 'Database service not available'
            }), 503
        
        data = await request.get_json()
        
        # Validate required fields
        required_fields = ['name', 'code', 'instruments', 'risk_level', 'execution_interval']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'status': 'error',
                    'message': f'Missing required field: {field}'
                }), 400
        
        # Set defaults for optional fields
        bot_data = {
            'name': data['name'],
            'description': data.get('description', ''),
            'code': data['code'],
            'instruments': data['instruments'],
            'risk_level': data['risk_level'],
            'execution_interval': data['execution_interval'],
            'trailing_stop_type': data.get('trailing_stop_type', 'none'),
            'trailing_stop_pips': data.get('trailing_stop_pips', 20)
        }
        
        # Temporary test user ID for development (proper UUID format)
        user_id = getattr(request, 'user_id', '11111111-1111-1111-1111-111111111111')
        bot_id = supabase_service.create_bot(user_id, bot_data)
        
        return jsonify({
            'status': 'success',
            'message': 'Bot created successfully',
            'bot_id': bot_id
        }), 201
        
    except Exception as e:
        logger.error(f"Error creating bot: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/custom-bots/<bot_id>', methods=['GET'])
# @require_auth  # Temporarily disabled for testing
async def get_bot(bot_id):
    """Get a specific bot"""
    try:
        if not supabase_service:
            return jsonify({
                'status': 'error',
                'message': 'Database service not available'
            }), 503
        
        # Temporary test user ID for development (proper UUID format)
        user_id = getattr(request, 'user_id', '11111111-1111-1111-1111-111111111111')
        bot = supabase_service.get_bot(user_id, bot_id)
        
        if not bot:
            return jsonify({
                'status': 'error',
                'message': 'Bot not found'
            }), 404
        
        # Get performance metrics
        performance = supabase_service.get_bot_performance(user_id, bot_id)
        
        bot_data = {
            'id': bot['id'],
            'name': bot['name'],
            'description': bot['description'],
            'code': bot['code'],
            'instruments': bot['instruments'],
            'risk_level': bot['risk_level'],
            'execution_interval': bot['execution_interval'],
            'trailing_stop_type': bot['trailing_stop_type'],
            'trailing_stop_pips': bot['trailing_stop_pips'],
            'status': bot['status'],
            'created_at': bot['created_at'],
            'updated_at': bot['updated_at'],
            'last_run': bot['last_run'],
            'performance': performance or {'total_trades': 0, 'win_rate': 0, 'total_pnl': 0},
            'error_count': bot['error_count'],
            'last_error': bot['last_error']
        }
        
        return jsonify({
            'status': 'success',
            'bot': bot_data
        })
        
    except Exception as e:
        logger.error(f"Error getting bot {bot_id}: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/custom-bots/<bot_id>', methods=['PUT'])
# @require_auth  # Temporarily disabled for testing
async def update_bot(bot_id):
    """Update a bot"""
    try:
        if not supabase_service:
            return jsonify({
                'status': 'error',
                'message': 'Database service not available'
            }), 503
        
        data = await request.get_json()
        
        # Temporary test user ID for development (proper UUID format)
        user_id = getattr(request, 'user_id', '11111111-1111-1111-1111-111111111111')
        success = supabase_service.update_bot(user_id, bot_id, data)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': 'Bot updated successfully'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to update bot'
            }), 500
            
    except Exception as e:
        logger.error(f"Error updating bot {bot_id}: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/custom-bots/<bot_id>', methods=['DELETE'])
# @require_auth  # Temporarily disabled for testing
async def delete_bot(bot_id):
    """Delete a bot"""
    try:
        if not supabase_service:
            return jsonify({
                'status': 'error',
                'message': 'Database service not available'
            }), 503
        
        # Temporary test user ID for development (proper UUID format)
        user_id = getattr(request, 'user_id', '11111111-1111-1111-1111-111111111111')
        success = supabase_service.delete_bot(user_id, bot_id)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': 'Bot deleted successfully'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to delete bot'
            }), 500
            
    except Exception as e:
        logger.error(f"Error deleting bot {bot_id}: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/custom-bots/<bot_id>/start', methods=['POST'])
# @require_auth  # Temporarily disabled for testing
async def start_bot(bot_id):
    """Start a bot with actual execution engine"""
    try:
        if not supabase_service:
            return jsonify({
                'status': 'error',
                'message': 'Database service not available'
            }), 503
        
        # Initialize CustomBotService on-demand
        try:
            bot_service = get_or_create_custom_bot_service()
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Failed to initialize execution service: {str(e)}'
            }), 503
        
        # Temporary test user ID for development (proper UUID format)
        user_id = getattr(request, 'user_id', '11111111-1111-1111-1111-111111111111')
        
        # Get bot details from database
        bot_data = supabase_service.get_bot(user_id, bot_id)
        if not bot_data:
            return jsonify({
                'status': 'error',
                'message': 'Bot not found'
            }), 404
        
        # Load bot into execution service
        load_success = bot_service.load_bot_from_database(bot_data)
        if not load_success:
            return jsonify({
                'status': 'error',
                'message': 'Failed to load bot into execution service'
            }), 500
        
        # Start bot in execution engine
        success = bot_service.start_bot(bot_id)
        
        if success:
            # Update database status
            supabase_service.update_bot_status(user_id, bot_id, 'running')
            supabase_service.log_bot_activity(user_id, bot_id, 'INFO', 'Bot started by user - Execution engine active')
            
            return jsonify({
                'status': 'success',
                'message': 'Bot started successfully - Now actively trading!'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to start bot in execution engine'
            }), 500
            
    except Exception as e:
        logger.error(f"Error starting bot {bot_id}: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/custom-bots/<bot_id>/stop', methods=['POST'])
# @require_auth  # Temporarily disabled for testing
async def stop_bot(bot_id):
    """Stop a bot from execution engine"""
    try:
        if not supabase_service:
            return jsonify({
                'status': 'error',
                'message': 'Database service not available'
            }), 503
        
        # Initialize CustomBotService on-demand
        try:
            bot_service = get_or_create_custom_bot_service()
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Failed to initialize execution service: {str(e)}'
            }), 503
        
        # Temporary test user ID for development (proper UUID format)
        user_id = getattr(request, 'user_id', '11111111-1111-1111-1111-111111111111')
        
        # Stop bot in execution engine
        success = bot_service.stop_bot(bot_id)
        
        if success:
            # Update database status
            supabase_service.update_bot_status(user_id, bot_id, 'stopped')
            supabase_service.log_bot_activity(user_id, bot_id, 'INFO', 'Bot stopped by user - Execution engine stopped')
            
            return jsonify({
                'status': 'success',
                'message': 'Bot stopped successfully'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to stop bot in execution engine'
            }), 500
            
    except Exception as e:
        logger.error(f"Error stopping bot {bot_id}: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/custom-bots/<bot_id>/positions', methods=['GET'])
@require_auth
async def get_bot_positions(bot_id):
    """Get positions for a specific bot"""
    try:
        if not supabase_service:
            return jsonify({
                'status': 'error',
                'message': 'Database service not available'
            }), 503
        
        # Temporary test user ID for development (proper UUID format)
        user_id = getattr(request, 'user_id', '11111111-1111-1111-1111-111111111111')
        positions = supabase_service.get_bot_positions(user_id, bot_id)
        
        return jsonify({
            'status': 'success',
            'positions': positions,
            'count': len(positions)
        })
        
    except Exception as e:
        logger.error(f"Error getting bot positions {bot_id}: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/custom-bots/<bot_id>/positions/<symbol>/close', methods=['POST'])
@require_auth
async def close_bot_position(bot_id, symbol):
    """Close a specific position for a bot"""
    try:
        if not supabase_service:
            return jsonify({
                'status': 'error',
                'message': 'Database service not available'
            }), 503
        
        data = await request.get_json() or {}
        reason = data.get('reason', 'Manual close from UI')
        
        # Temporary test user ID for development (proper UUID format)
        user_id = getattr(request, 'user_id', '11111111-1111-1111-1111-111111111111')
        success = supabase_service.close_position(user_id, bot_id, symbol)
        
        if success:
            # Log the activity
            supabase_service.log_bot_activity(user_id, bot_id, 'INFO', f'Position {symbol} closed manually', {'reason': reason})
            
            return jsonify({
                'status': 'success',
                'message': f'Position {symbol} closed successfully'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': f'No position found for {symbol}'
            }), 404
            
    except Exception as e:
        logger.error(f"Error closing bot position {bot_id}/{symbol}: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/custom-bots/<bot_id>/trades', methods=['GET'])
@require_auth
async def get_bot_trades(bot_id):
    """Get trade history for a specific bot"""
    try:
        if not supabase_service:
            return jsonify({
                'status': 'error',
                'message': 'Database service not available'
            }), 503
        
        # Temporary test user ID for development (proper UUID format)
        user_id = getattr(request, 'user_id', '11111111-1111-1111-1111-111111111111')
        trades = supabase_service.get_bot_trades(user_id, bot_id)
        
        return jsonify({
            'status': 'success',
            'trades': trades,
            'count': len(trades)
        })
        
    except Exception as e:
        logger.error(f"Error getting bot trades {bot_id}: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/custom-bots/<bot_id>/logs', methods=['GET'])
@require_auth
async def get_bot_logs(bot_id):
    """Get logs for a specific bot"""
    try:
        if not supabase_service:
            return jsonify({
                'status': 'error',
                'message': 'Database service not available'
            }), 503
        
        # Temporary test user ID for development (proper UUID format)
        user_id = getattr(request, 'user_id', '11111111-1111-1111-1111-111111111111')
        logs = supabase_service.get_bot_logs(user_id, bot_id)
        
        return jsonify({
            'status': 'success',
            'logs': logs,
            'count': len(logs)
        })
        
    except Exception as e:
        logger.error(f"Error getting bot logs {bot_id}: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/custom-bots/<bot_id>/validate', methods=['POST'])
# @require_auth  # Temporarily disabled for testing
async def validate_bot_code(bot_id):
    """Validate bot code without saving"""
    try:
        data = await request.get_json()
        code = data.get('code', '')
        
        if not code:
            return jsonify({
                'status': 'error',
                'message': 'No code provided'
            }), 400
        
        # Simple validation - check for basic Python syntax
        try:
            compile(code, '<string>', 'exec')
            is_valid = True
        except SyntaxError as e:
            is_valid = False
            error_message = f'Syntax error: {str(e)}'
        except Exception as e:
            is_valid = False
            error_message = f'Compilation error: {str(e)}'
        
        return jsonify({
            'status': 'success',
            'valid': is_valid,
            'message': 'Code is valid' if is_valid else error_message
        })
        
    except Exception as e:
        logger.error(f"Error validating bot code: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'valid': False
        }), 500

# ================================
# USER PROFILE ROUTES
# ================================

@app.route('/api/user/profile', methods=['GET'])
@require_auth
async def get_user_profile():
    """Get user profile"""
    try:
        if not supabase_service:
            return jsonify({
                'status': 'error',
                'message': 'Database service not available'
            }), 503
        
        # Temporary test user ID for development (proper UUID format)
        user_id = getattr(request, 'user_id', '11111111-1111-1111-1111-111111111111')
        profile = supabase_service.get_user_profile(user_id)
        
        if not profile:
            return jsonify({
                'status': 'error',
                'message': 'Profile not found'
            }), 404
        
        return jsonify({
            'status': 'success',
            'profile': profile
        })
        
    except Exception as e:
        logger.error(f"Error getting user profile: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/user/subscription', methods=['GET'])
@require_auth
async def get_user_subscription():
    """Get user subscription info"""
    try:
        if not supabase_service:
            return jsonify({
                'status': 'error',
                'message': 'Database service not available'
            }), 503
        
        # Temporary test user ID for development (proper UUID format)
        user_id = getattr(request, 'user_id', '11111111-1111-1111-1111-111111111111')
        subscription = supabase_service.get_user_subscription(user_id)
        
        return jsonify({
            'status': 'success',
            'subscription': subscription or {'tier': 'free', 'status': 'active'}
        })
        
    except Exception as e:
        logger.error(f"Error getting user subscription: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# ================================
# DASHBOARD ROUTES
# ================================

@app.route('/api/dashboard', methods=['GET'])
@require_auth
async def get_dashboard_data():
    """Get dashboard data for authenticated user"""
    try:
        if not supabase_service:
            return jsonify({
                'status': 'error',
                'message': 'Database service not available'
            }), 503
        
        # Temporary test user ID for development (proper UUID format)
        user_id = getattr(request, 'user_id', '11111111-1111-1111-1111-111111111111')
        
        # Get user bots
        bots = supabase_service.get_user_bots(user_id)
        
        # Calculate aggregated statistics
        total_bots = len(bots)
        running_bots = len([b for b in bots if b['status'] == 'running'])
        total_trades = 0
        total_pnl = 0
        active_positions = 0
        
        for bot in bots:
            performance = supabase_service.get_bot_performance(user_id, bot['id'])
            if performance:
                total_trades += performance.get('total_trades', 0)
                total_pnl += float(performance.get('total_pnl', 0))
            
            positions = supabase_service.get_bot_positions(user_id, bot['id'])
            active_positions += len(positions)
        
        return jsonify({
            'status': 'success',
            'data': {
                'total_bots': total_bots,
                'running_bots': running_bots,
                'stopped_bots': total_bots - running_bots,
                'total_trades': total_trades,
                'total_pnl': total_pnl,
                'active_positions': active_positions,
                'bots': [{
                    'id': bot['id'],
                    'name': bot['name'],
                    'status': bot['status'],
                    'created_at': bot['created_at']
                } for bot in bots[:5]]  # Latest 5 bots for preview
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# ================================
# BOT TEMPLATES ROUTES
# ================================

@app.route('/api/bot-templates', methods=['GET'])
async def get_bot_templates():
    """Get all bot templates with optional filters"""
    try:
        if not supabase_service:
            return jsonify({
                'status': 'error',
                'message': 'Database service not available'
            }), 503
        
        language = request.args.get('language')
        category = request.args.get('category')
        strategy_type = request.args.get('strategy_type')
        
        templates = supabase_service.get_bot_templates(
            language=language,
            category=category,
            strategy_type=strategy_type
        )
        
        return jsonify({
            'status': 'success',
            'templates': templates,
            'count': len(templates)
        })
        
    except Exception as e:
        logger.error(f"Error getting bot templates: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/bot-templates/<template_id>', methods=['GET'])
async def get_bot_template(template_id):
    """Get a specific bot template"""
    try:
        if not supabase_service:
            return jsonify({
                'status': 'error',
                'message': 'Database service not available'
            }), 503
        
        template = supabase_service.get_bot_template(template_id)
        
        if not template:
            return jsonify({
                'status': 'error',
                'message': 'Template not found'
            }), 404
        
        # Increment usage count
        supabase_service.increment_template_usage(template_id)
        
        return jsonify({
            'status': 'success',
            'template': template
        })
        
    except Exception as e:
        logger.error(f"Error getting bot template {template_id}: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/bot-templates', methods=['POST'])
@require_auth
async def create_bot_template():
    """Create a new bot template (admin only for now)"""
    try:
        if not supabase_service:
            return jsonify({
                'status': 'error',
                'message': 'Database service not available'
            }), 503
        
        data = await request.get_json()
        
        if not data or not data.get('name') or not data.get('code') or not data.get('language'):
            return jsonify({
                'status': 'error',
                'message': 'Missing required fields: name, code, language'
            }), 400
        
        template_id = supabase_service.create_bot_template(data)
        
        if template_id:
            return jsonify({
                'status': 'success',
                'id': template_id,
                'message': 'Template created successfully'
            }), 201
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to create template'
            }), 500
            
    except Exception as e:
        logger.error(f"Error creating bot template: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/bot-templates/<template_id>', methods=['PUT'])
@require_auth
async def update_bot_template(template_id):
    """Update a bot template (admin only for now)"""
    try:
        if not supabase_service:
            return jsonify({
                'status': 'error',
                'message': 'Database service not available'
            }), 503
        
        data = await request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No data provided'
            }), 400
        
        success = supabase_service.update_bot_template(template_id, data)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': 'Template updated successfully'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to update template'
            }), 500
            
    except Exception as e:
        logger.error(f"Error updating bot template {template_id}: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/bot-templates/<template_id>', methods=['DELETE'])
@require_auth
async def delete_bot_template(template_id):
    """Delete a bot template (admin only for now)"""
    try:
        if not supabase_service:
            return jsonify({
                'status': 'error',
                'message': 'Database service not available'
            }), 503
        
        success = supabase_service.delete_bot_template(template_id)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': 'Template deleted successfully'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to delete template'
            }), 500
            
    except Exception as e:
        logger.error(f"Error deleting bot template {template_id}: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


if __name__ == '__main__':
    print("""
    🚀 MASTER TRADING BOT SERVER 🚀
    
    ================================
    PROFIT MAXIMIZER BOT
    ================================
    
    🔥 Features:
    - Analyzes ALL assets simultaneously
    - AI-powered profit scoring
    - Automatic trade execution
    - Risk-adjusted position sizing
    - Real-time performance tracking
    
    📊 Endpoints:
    - POST /api/master-bot/start - Start the bot
    - POST /api/master-bot/stop - Stop the bot
    - GET  /api/master-bot/status - Get bot status
    - POST /api/master-bot/analyze - Manual analysis
    - POST /api/master-bot/force-trade - Force trade
    - GET  /api/master-bot/performance - Performance metrics
    
    🎯 Ready to make profitable trades!
    """)
    
    app.run(host='0.0.0.0', port=5004, debug=True) 