from flask import Flask, jsonify, make_response
from werkzeug.serving import run_simple
from trading_data import store

app = Flask(__name__)

@app.route('/health')
def health_check():
    return {'status': 'healthy'}

@app.route('/trading-status')
def get_trading_status():
    data = store.get_data()
    
    response = make_response(jsonify(data))
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = '*'
    response.headers['Access-Control-Allow-Headers'] = '*'
    
    return response

@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = '*'
    response.headers['Access-Control-Allow-Headers'] = '*'
    return response

def start_server():
    run_simple('127.0.0.1', 5001, app, use_reloader=True, use_debugger=True)

if __name__ == '__main__':
    start_server() 