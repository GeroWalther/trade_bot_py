from flask import Blueprint, request, jsonify
import logging
from services.custom_bot_service import CustomBotService, BotStatus

logger = logging.getLogger(__name__)

def create_custom_bot_routes(custom_bot_service: CustomBotService):
    """Create Flask Blueprint for custom bot routes"""
    
    bp = Blueprint('custom_bots', __name__, url_prefix='/api/custom-bots')
    
    @bp.route('/', methods=['GET'])
    def get_all_bots():
        """Get all custom bots"""
        try:
            bots = custom_bot_service.get_all_bots()
            
            # Convert to JSON-serializable format
            bots_data = []
            for bot in bots:
                bot_data = {
                    'id': bot.id,
                    'name': bot.name,
                    'description': bot.description,
                    'instruments': bot.instruments,
                    'risk_level': bot.risk_level,
                    'execution_interval': bot.execution_interval,
                    'trailing_stop_type': bot.trailing_stop_type,
                    'trailing_stop_pips': bot.trailing_stop_pips,
                    'status': bot.status.value,
                    'created_at': bot.created_at.isoformat() if bot.created_at else None,
                    'updated_at': bot.updated_at.isoformat() if bot.updated_at else None,
                    'last_run': bot.last_run.isoformat() if bot.last_run else None,
                    'performance': bot.performance or {'total_trades': 0, 'win_rate': 0, 'total_pnl': 0},
                    'error_count': bot.error_count,
                    'last_error': bot.last_error
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
    
    @bp.route('/', methods=['POST'])
    def create_bot():
        """Create a new custom bot"""
        try:
            data = request.get_json()
            
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
            
            bot_id = custom_bot_service.create_bot(bot_data)
            
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
    
    @bp.route('/<bot_id>', methods=['GET'])
    def get_bot(bot_id):
        """Get a specific bot"""
        try:
            bot = custom_bot_service.get_bot(bot_id)
            
            if not bot:
                return jsonify({
                    'status': 'error',
                    'message': 'Bot not found'
                }), 404
            
            bot_data = {
                'id': bot.id,
                'name': bot.name,
                'description': bot.description,
                'code': bot.code,
                'instruments': bot.instruments,
                'risk_level': bot.risk_level,
                'execution_interval': bot.execution_interval,
                'trailing_stop_type': bot.trailing_stop_type,
                'trailing_stop_pips': bot.trailing_stop_pips,
                'status': bot.status.value,
                'created_at': bot.created_at.isoformat() if bot.created_at else None,
                'updated_at': bot.updated_at.isoformat() if bot.updated_at else None,
                'last_run': bot.last_run.isoformat() if bot.last_run else None,
                'performance': bot.performance or {'total_trades': 0, 'win_rate': 0, 'total_pnl': 0},
                'error_count': bot.error_count,
                'last_error': bot.last_error
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
    
    @bp.route('/<bot_id>', methods=['PUT'])
    def update_bot(bot_id):
        """Update a bot"""
        try:
            data = request.get_json()
            
            success = custom_bot_service.update_bot(bot_id, data)
            
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
    
    @bp.route('/<bot_id>', methods=['DELETE'])
    def delete_bot(bot_id):
        """Delete a bot"""
        try:
            success = custom_bot_service.delete_bot(bot_id)
            
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
    
    @bp.route('/<bot_id>/start', methods=['POST'])
    def start_bot(bot_id):
        """Start a bot"""
        try:
            success = custom_bot_service.start_bot(bot_id)
            
            if success:
                return jsonify({
                    'status': 'success',
                    'message': 'Bot started successfully'
                })
            else:
                return jsonify({
                    'status': 'error',
                    'message': 'Failed to start bot'
                }), 500
                
        except Exception as e:
            logger.error(f"Error starting bot {bot_id}: {e}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
    
    @bp.route('/<bot_id>/stop', methods=['POST'])
    def stop_bot(bot_id):
        """Stop a bot"""
        try:
            success = custom_bot_service.stop_bot(bot_id)
            
            if success:
                return jsonify({
                    'status': 'success',
                    'message': 'Bot stopped successfully'
                })
            else:
                return jsonify({
                    'status': 'error',
                    'message': 'Failed to stop bot'
                }), 500
                
        except Exception as e:
            logger.error(f"Error stopping bot {bot_id}: {e}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
    
    @bp.route('/<bot_id>/positions', methods=['GET'])
    def get_bot_positions(bot_id):
        """Get positions for a specific bot"""
        try:
            positions = custom_bot_service.get_bot_positions(bot_id)
            
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
    
    @bp.route('/<bot_id>/positions/<symbol>/close', methods=['POST'])
    def close_bot_position(bot_id, symbol):
        """Close a specific position for a bot"""
        try:
            data = request.get_json() or {}
            reason = data.get('reason', 'Manual close from UI')
            
            success = custom_bot_service.close_bot_position(bot_id, symbol, reason)
            
            if success:
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
    
    @bp.route('/<bot_id>/validate', methods=['POST'])
    def validate_bot_code(bot_id):
        """Validate bot code without saving"""
        try:
            data = request.get_json()
            code = data.get('code', '')
            
            if not code:
                return jsonify({
                    'status': 'error',
                    'message': 'No code provided'
                }), 400
            
            # Use the service's validation method
            is_valid = custom_bot_service._validate_bot_code(code)
            
            return jsonify({
                'status': 'success',
                'valid': is_valid,
                'message': 'Code is valid' if is_valid else 'Code validation failed'
            })
            
        except Exception as e:
            logger.error(f"Error validating bot code: {e}")
            return jsonify({
                'status': 'error',
                'message': str(e),
                'valid': False
            }), 500
    
    return bp 