from flask_socketio import SocketIO, emit
from src.data_ingestion import DataIngestion
import threading
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebSocketHandler:
    """
    WebSocket handler for real-time data streaming
    """
    
    def __init__(self, socketio: SocketIO):
        self.socketio = socketio
        self.data_ingestion = DataIngestion()
        self.active_streams = {}
        self.stream_threads = {}
        
    def start_price_stream(self, symbol: str, interval: int = 30):
        """
        Start streaming real-time price data for a symbol
        
        Args:
            symbol (str): Stock symbol
            interval (int): Update interval in seconds
        """
        if symbol in self.active_streams:
            logger.info(f"Stream already active for {symbol}")
            return
        
        self.active_streams[symbol] = True
        
        def stream_worker():
            logger.info(f"Starting price stream for {symbol}")
            
            while self.active_streams.get(symbol, False):
                try:
                    # Fetch real-time data
                    data = self.data_ingestion.fetch_real_time_data(symbol)
                    
                    if data:
                        # Emit data to connected clients
                        self.socketio.emit('price_update', {
                            'symbol': symbol,
                            'data': data
                        }, room=f'stream_{symbol}')
                        
                        logger.debug(f"Emitted price update for {symbol}: ${data['current_price']}")
                    
                    time.sleep(interval)
                    
                except Exception as e:
                    logger.error(f"Error in price stream for {symbol}: {str(e)}")
                    time.sleep(interval)
            
            logger.info(f"Price stream stopped for {symbol}")
        
        # Start stream thread
        thread = threading.Thread(target=stream_worker, daemon=True)
        thread.start()
        self.stream_threads[symbol] = thread
        
    def stop_price_stream(self, symbol: str):
        """
        Stop streaming real-time price data for a symbol
        
        Args:
            symbol (str): Stock symbol
        """
        if symbol in self.active_streams:
            self.active_streams[symbol] = False
            logger.info(f"Stopping price stream for {symbol}")
            
            # Wait for thread to finish
            if symbol in self.stream_threads:
                thread = self.stream_threads[symbol]
                thread.join(timeout=5)
                del self.stream_threads[symbol]
    
    def stop_all_streams(self):
        """Stop all active price streams"""
        symbols_to_stop = list(self.active_streams.keys())
        for symbol in symbols_to_stop:
            self.stop_price_stream(symbol)

def setup_websocket_events(socketio: SocketIO, websocket_handler: WebSocketHandler):
    """
    Set up WebSocket event handlers
    
    Args:
        socketio (SocketIO): SocketIO instance
        websocket_handler (WebSocketHandler): WebSocket handler instance
    """
    
    @socketio.on('connect')
    def handle_connect():
        logger.info(f"Client connected: {request.sid}")
        emit('connected', {'message': 'Connected to financial data stream'})
    
    @socketio.on('disconnect')
    def handle_disconnect():
        logger.info(f"Client disconnected: {request.sid}")
    
    @socketio.on('subscribe_price')
    def handle_subscribe_price(data):
        """Handle price subscription request"""
        try:
            symbol = data.get('symbol', '').upper()
            interval = data.get('interval', 30)
            
            if not websocket_handler.data_ingestion.validate_symbol(symbol):
                emit('error', {'message': f'Invalid symbol: {symbol}'})
                return
            
            # Join room for this symbol
            join_room(f'stream_{symbol}')
            
            # Start price stream if not already active
            websocket_handler.start_price_stream(symbol, interval)
            
            emit('subscribed', {
                'symbol': symbol,
                'interval': interval,
                'message': f'Subscribed to {symbol} price updates'
            })
            
            logger.info(f"Client subscribed to {symbol} price updates")
            
        except Exception as e:
            logger.error(f"Error handling price subscription: {str(e)}")
            emit('error', {'message': 'Failed to subscribe to price updates'})
    
    @socketio.on('unsubscribe_price')
    def handle_unsubscribe_price(data):
        """Handle price unsubscription request"""
        try:
            symbol = data.get('symbol', '').upper()
            
            # Leave room for this symbol
            leave_room(f'stream_{symbol}')
            
            emit('unsubscribed', {
                'symbol': symbol,
                'message': f'Unsubscribed from {symbol} price updates'
            })
            
            logger.info(f"Client unsubscribed from {symbol} price updates")
            
        except Exception as e:
            logger.error(f"Error handling price unsubscription: {str(e)}")
            emit('error', {'message': 'Failed to unsubscribe from price updates'})
    
    @socketio.on('get_symbols')
    def handle_get_symbols():
        """Handle request for supported symbols"""
        try:
            symbols = websocket_handler.data_ingestion.get_supported_symbols()
            emit('symbols', {'symbols': symbols})
        except Exception as e:
            logger.error(f"Error getting symbols: {str(e)}")
            emit('error', {'message': 'Failed to get supported symbols'})

