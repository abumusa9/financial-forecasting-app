import os
import sys

# --- Monkey patch for eventlet ---
import eventlet
eventlet.monkey_patch()

# DON'T CHANGE THIS !!!
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, send_from_directory, request
from flask_cors import CORS
from flask_socketio import SocketIO
from src.database import db
from src.models.user import User
from src.models.stock_data import StockData, ModelPrediction
from src.routes.user import user_bp
from src.routes.stock_api import stock_bp
from src.websocket_handler import WebSocketHandler, setup_websocket_events

# Flask app setup
app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), 'static'))
app.config['SECRET_KEY'] = 'asdf#FGSgvasgf$5$WGT'

# Enable CORS for all routes
CORS(app, origins="*")

# Initialize SocketIO with eventlet
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# WebSocket handler
websocket_handler = WebSocketHandler(socketio)
setup_websocket_events(socketio, websocket_handler)

# Register blueprints
app.register_blueprint(user_bp, url_prefix='/api')
app.register_blueprint(stock_bp, url_prefix='/api/stock')

# Database config
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(os.path.dirname(__file__), 'database', 'app.db')}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)
with app.app_context():
    db.create_all()

# Serve React/Frontend build
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    static_folder_path = app.static_folder
    if static_folder_path is None:
        return "Static folder not configured", 404

    if path != "" and os.path.exists(os.path.join(static_folder_path, path)):
        return send_from_directory(static_folder_path, path)
    else:
        index_path = os.path.join(static_folder_path, 'index.html')
        if os.path.exists(index_path):
            return send_from_directory(static_folder_path, 'index.html')
        else:
            return "index.html not found", 404


# Run with eventlet WSGI server
if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5001, debug=False)
