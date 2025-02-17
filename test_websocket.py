import unittest
from websocket_server import app, socketio
import json

class TestWebSocket(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app_context = app.app_context()
        self.app_context.push()

    def test_health_check(self):
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {'status': 'healthy'})

    def test_websocket_connection(self):
        client = socketio.test_client(app)
        self.assertTrue(client.is_connected())
        
        # Test sending trading status
        test_data = {
            'cash_available': 10000,
            'total_portfolio_value': 10500,
            'daily_return_pct': 5.0,
            'positions': {}
        }
        
        client.emit('trading_status', test_data)
        received = client.get_received()
        self.assertTrue(len(received) > 0)
        
        client.disconnect()
        self.assertFalse(client.is_connected())

if __name__ == '__main__':
    unittest.main() 