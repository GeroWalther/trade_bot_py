from quart import Quart, request, jsonify
from quart_cors import cors
import asyncio
import logging
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from master_trading_bot import MasterTradingBot
from routes.trading_routes import trading_bp
from routes.analysis_routes import analysis_bp
from routes.ai_analysis_routes import ai_analysis_bp
from services.market_intelligence_service import MarketIntelligenceService
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
market_intelligence = None

@app.before_serving
async def startup():
    """Initialize services on startup"""
    global market_intelligence
    
    logger.info("🚀 Starting Master Trading Server...")
    
    try:
        # Validate API keys
        validate_api_keys()
        
        # Initialize market intelligence service
        market_intelligence = MarketIntelligenceService()
        
        logger.info("✅ Master Trading Server initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize server: {e}")
        raise

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
    """Get trading status - compatibility endpoint"""
    global master_bot
    
    try:
        if not master_bot:
            return jsonify({
                'status': 'success',
                'data': {
                    'active_positions': [],
                    'account_balance': 0,
                    'today_pnl': 0,
                    'bot_status': 'stopped'
                }
            })
        
        return jsonify({
            'status': 'success',
            'data': {
                'active_positions': [master_bot.current_position] if master_bot.current_position else [],
                'account_balance': 10000,  # Mock data
                'today_pnl': master_bot.performance_metrics['total_profit_loss'],
                'bot_status': 'running' if master_bot.is_running else 'stopped',
                'total_trades': master_bot.performance_metrics['total_trades'],
                'win_rate': master_bot.performance_metrics['win_rate']
            }
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

# Register existing blueprints (for backward compatibility)
app.register_blueprint(trading_bp)
app.register_blueprint(analysis_bp)
app.register_blueprint(ai_analysis_bp)

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
    
    app.run(host='0.0.0.0', port=5003, debug=True) 