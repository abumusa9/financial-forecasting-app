from flask import Blueprint, request, jsonify
from flask_cors import cross_origin
from src.data_ingestion import DataIngestion
from src.lstm_model import LSTMForecaster
from src.database import db
from src.models.stock_data import StockData, ModelPrediction
from datetime import datetime, date
import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
stock_bp = Blueprint('stock', __name__)

# Initialize components
data_ingestion = DataIngestion()
lstm_forecaster = LSTMForecaster()

# Model paths
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models', 'saved')
os.makedirs(MODEL_DIR, exist_ok=True)

@stock_bp.route('/symbols', methods=['GET'])
@cross_origin()
def get_supported_symbols():
    """Get list of supported stock symbols"""
    try:
        symbols = data_ingestion.get_supported_symbols()
        return jsonify({
            'success': True,
            'symbols': symbols
        })
    except Exception as e:
        logger.error(f"Error getting symbols: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@stock_bp.route('/historical/<symbol>', methods=['GET'])
@cross_origin()
def get_historical_data(symbol):
    """Get historical stock data for a symbol"""
    try:
        # Validate symbol
        if not data_ingestion.validate_symbol(symbol):
            return jsonify({
                'success': False,
                'error': f'Symbol {symbol} is not supported'
            }), 400
        
        # Get period from query parameters
        period = request.args.get('period', '1y')
        
        # Check if data exists in database
        existing_data = StockData.query.filter_by(symbol=symbol.upper()).order_by(StockData.date.desc()).all()
        
        if existing_data:
            # Convert to list of dictionaries
            data_list = [record.to_dict() for record in existing_data]
            return jsonify({
                'success': True,
                'data': data_list,
                'source': 'database'
            })
        
        # Fetch from API if not in database
        data = data_ingestion.fetch_historical_data(symbol.upper(), period)
        
        if data is None:
            return jsonify({
                'success': False,
                'error': f'No data found for symbol {symbol}'
            }), 404
        
        # Store in database
        for _, row in data.iterrows():
            stock_record = StockData(
                symbol=row['symbol'],
                date=row['date'],
                open_price=row['open_price'],
                high_price=row['high_price'],
                low_price=row['low_price'],
                close_price=row['close_price'],
                volume=row['volume']
            )
            db.session.add(stock_record)
        
        db.session.commit()
        
        # Convert to list of dictionaries
        data_list = data.to_dict('records')
        
        return jsonify({
            'success': True,
            'data': data_list,
            'source': 'api'
        })
        
    except Exception as e:
        logger.error(f"Error getting historical data for {symbol}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@stock_bp.route('/realtime/<symbol>', methods=['GET'])
@cross_origin()
def get_realtime_data(symbol):
    """Get real-time stock data for a symbol"""
    try:
        # Validate symbol
        if not data_ingestion.validate_symbol(symbol):
            return jsonify({
                'success': False,
                'error': f'Symbol {symbol} is not supported'
            }), 400
        
        # Fetch real-time data
        data = data_ingestion.fetch_real_time_data(symbol.upper())
        
        if data is None:
            return jsonify({
                'success': False,
                'error': f'No real-time data found for symbol {symbol}'
            }), 404
        
        return jsonify({
            'success': True,
            'data': data
        })
        
    except Exception as e:
        logger.error(f"Error getting real-time data for {symbol}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@stock_bp.route('/train/<symbol>', methods=['POST'])
@cross_origin()
def train_model(symbol):
    """Train LSTM model for a specific symbol"""
    try:
        # Validate symbol
        if not data_ingestion.validate_symbol(symbol):
            return jsonify({
                'success': False,
                'error': f'Symbol {symbol} is not supported'
            }), 400
        
        # Get training parameters
        data_request = request.get_json() or {}
        epochs = data_request.get('epochs', 50)
        batch_size = data_request.get('batch_size', 32)
        
        # Get historical data from database
        stock_records = StockData.query.filter_by(symbol=symbol.upper()).order_by(StockData.date.asc()).all()
        
        if len(stock_records) < 100:  # Need sufficient data for training
            return jsonify({
                'success': False,
                'error': f'Insufficient data for training. Need at least 100 records, found {len(stock_records)}'
            }), 400
        
        # Convert to DataFrame
        data_list = [record.to_dict() for record in stock_records]
        df = pd.DataFrame(data_list)
        df['date'] = pd.to_datetime(df['date'])
        
        # Preprocess data
        processed_data = data_ingestion.preprocess_data(df)
        
        # Train model
        training_results = lstm_forecaster.train(
            processed_data,
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Save model
        model_path = os.path.join(MODEL_DIR, f'{symbol.lower()}_model.h5')
        scaler_path = os.path.join(MODEL_DIR, f'{symbol.lower()}_scalers.pkl')
        lstm_forecaster.save_model(model_path, scaler_path)
        
        return jsonify({
            'success': True,
            'training_results': training_results,
            'message': f'Model trained successfully for {symbol}'
        })
        
    except Exception as e:
        logger.error(f"Error training model for {symbol}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@stock_bp.route('/predict/<symbol>', methods=['POST'])
@cross_origin()
def predict_prices(symbol):
    """Generate price predictions for a symbol"""
    try:
        # Validate symbol
        if not data_ingestion.validate_symbol(symbol):
            return jsonify({
                'success': False,
                'error': f'Symbol {symbol} is not supported'
            }), 400
        
        # Get prediction parameters
        data_request = request.get_json() or {}
        days_ahead = data_request.get('days_ahead', 30)
        
        # Load model if not already loaded
        model_path = os.path.join(MODEL_DIR, f'{symbol.lower()}_model.h5')
        scaler_path = os.path.join(MODEL_DIR, f'{symbol.lower()}_scalers.pkl')
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return jsonify({
                'success': False,
                'error': f'No trained model found for {symbol}. Please train the model first.'
            }), 404
        
        # Load model
        lstm_forecaster.load_model(model_path, scaler_path)
        
        # Get recent data for prediction
        stock_records = StockData.query.filter_by(symbol=symbol.upper()).order_by(StockData.date.desc()).limit(200).all()
        stock_records.reverse()  # Order by date ascending
        
        if len(stock_records) < lstm_forecaster.sequence_length:
            return jsonify({
                'success': False,
                'error': f'Insufficient recent data for prediction. Need at least {lstm_forecaster.sequence_length} records.'
            }), 400
        
        # Convert to DataFrame
        data_list = [record.to_dict() for record in stock_records]
        df = pd.DataFrame(data_list)
        df['date'] = pd.to_datetime(df['date'])
        
        # Preprocess data
        processed_data = data_ingestion.preprocess_data(df)
        
        # Make predictions
        predictions = lstm_forecaster.predict(processed_data, days_ahead)
        
        # Store predictions in database
        for i, (pred_date, pred_price, (conf_lower, conf_upper)) in enumerate(
            zip(predictions['dates'], predictions['predictions'], predictions['confidence_intervals'])
        ):
            prediction_record = ModelPrediction(
                symbol=symbol.upper(),
                prediction_date=datetime.strptime(pred_date, '%Y-%m-%d').date(),
                predicted_price=pred_price,
                confidence_lower=conf_lower,
                confidence_upper=conf_upper,
                model_version=predictions['model_version']
            )
            db.session.add(prediction_record)
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'predictions': predictions
        })
        
    except Exception as e:
        logger.error(f"Error predicting prices for {symbol}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@stock_bp.route('/predictions/<symbol>', methods=['GET'])
@cross_origin()
def get_stored_predictions(symbol):
    """Get stored predictions for a symbol"""
    try:
        # Validate symbol
        if not data_ingestion.validate_symbol(symbol):
            return jsonify({
                'success': False,
                'error': f'Symbol {symbol} is not supported'
            }), 400
        
        # Get predictions from database
        predictions = ModelPrediction.query.filter_by(symbol=symbol.upper()).order_by(ModelPrediction.prediction_date.asc()).all()
        
        if not predictions:
            return jsonify({
                'success': False,
                'error': f'No predictions found for {symbol}'
            }), 404
        
        # Convert to list of dictionaries
        predictions_list = [pred.to_dict() for pred in predictions]
        
        return jsonify({
            'success': True,
            'predictions': predictions_list
        })
        
    except Exception as e:
        logger.error(f"Error getting predictions for {symbol}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@stock_bp.route('/evaluate/<symbol>', methods=['GET'])
@cross_origin()
def evaluate_model(symbol):
    """Evaluate model performance for a symbol"""
    try:
        logger.info(f"Starting evaluation for {symbol}")
        
        # Validate symbol
        if not data_ingestion.validate_symbol(symbol):
            return jsonify({
                'success': False,
                'error': f'Symbol {symbol} is not supported'
            }), 400
        
        # Load model
        model_path = os.path.join(MODEL_DIR, f'{symbol.lower()}_model.h5')
        scaler_path = os.path.join(MODEL_DIR, f'{symbol.lower()}_scalers.pkl')
        
        logger.info(f"Checking model files: {model_path}, {scaler_path}")
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return jsonify({
                'success': False,
                'error': f'No trained model found for {symbol}'
            }), 404
        
        # Load model
        logger.info("Loading model...")
        lstm_forecaster.load_model(model_path, scaler_path)
        
        # Get test data (use more data if needed for evaluation)
        logger.info("Fetching test data...")
        stock_records = StockData.query.filter_by(symbol=symbol.upper()).order_by(StockData.date.asc()).all()
        
        if len(stock_records) < 100:
            return jsonify({
                'success': False,
                'error': 'Insufficient data for evaluation (minimum 100 records required)'
            }), 400
        
        # Use last 50% for evaluation, but ensure we have at least 80 records
        test_size = max(int(len(stock_records) * 0.5), 80)
        test_records = stock_records[-test_size:]
        logger.info(f"Using {len(test_records)} records for evaluation")
        
        # Convert to DataFrame
        logger.info("Converting to DataFrame...")
        data_list = [record.to_dict() for record in test_records]
        df = pd.DataFrame(data_list)
        df['date'] = pd.to_datetime(df['date'])
        
        # Preprocess data
        logger.info("Preprocessing data...")
        processed_data = data_ingestion.preprocess_data(df)
        logger.info(f"Processed data shape: {processed_data.shape}")
        
        # Evaluate model
        logger.info("Evaluating model...")
        metrics = lstm_forecaster.evaluate(processed_data)
        
        logger.info(f"Evaluation completed successfully: {metrics}")
        return jsonify({
            'success': True,
            'metrics': metrics
        })
        
    except Exception as e:
        logger.error(f"Error evaluating model for {symbol}: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

