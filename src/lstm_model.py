import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import os
import logging
from typing import Tuple, List, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LSTMForecaster:
    """
    LSTM Neural Network for Financial Time Series Forecasting
    """
    
    def __init__(self, sequence_length: int = 60, features: List[str] = None):
        """
        Initialize LSTM Forecaster
        
        Args:
            sequence_length (int): Number of time steps to look back
            features (List[str]): List of feature columns to use
        """
        self.sequence_length = sequence_length
        self.features = features or ['close_price', 'volume', 'sma_5', 'sma_10', 'sma_20', 'rsi', 'volatility']
        self.model = None
        self.scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        self.is_trained = False
        self.model_version = "v1.0"
        
    def prepare_data(self, data: pd.DataFrame, target_column: str = 'close_price') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM training
        
        Args:
            data (pd.DataFrame): Input data
            target_column (str): Target column name
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: X (features) and y (targets)
        """
        try:
            logger.info("Preparing data for LSTM training")
            
            # Select features
            feature_data = data[self.features].values
            target_data = data[target_column].values.reshape(-1, 1)
            
            # Scale features and target
            scaled_features = self.scaler.fit_transform(feature_data)
            scaled_target = self.target_scaler.fit_transform(target_data)
            
            # Create sequences
            X, y = [], []
            for i in range(self.sequence_length, len(scaled_features)):
                X.append(scaled_features[i-self.sequence_length:i])
                y.append(scaled_target[i, 0])
            
            X, y = np.array(X), np.array(y)
            
            logger.info(f"Prepared data shapes - X: {X.shape}, y: {y.shape}")
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise
    
    def build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """
        Build LSTM model architecture
        
        Args:
            input_shape (Tuple[int, int]): Input shape (sequence_length, n_features)
        
        Returns:
            Sequential: Compiled LSTM model
        """
        try:
            logger.info(f"Building LSTM model with input shape: {input_shape}")
            
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                LSTM(50, return_sequences=True),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mean_squared_error',
                metrics=['mean_absolute_error']
            )
            
            logger.info("LSTM model built successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise
    
    def train(self, data: pd.DataFrame, validation_split: float = 0.2, epochs: int = 100, batch_size: int = 32) -> Dict:
        """
        Train the LSTM model
        
        Args:
            data (pd.DataFrame): Training data
            validation_split (float): Validation split ratio
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
        
        Returns:
            Dict: Training history and metrics
        """
        try:
            logger.info("Starting LSTM model training")
            
            # Prepare data
            X, y = self.prepare_data(data)
            
            # Build model
            self.model = self.build_model((X.shape[1], X.shape[2]))
            
            # Define callbacks
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.0001
            )
            
            # Train model
            history = self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )
            
            self.is_trained = True
            
            # Calculate final metrics
            train_loss = history.history['loss'][-1]
            val_loss = history.history['val_loss'][-1]
            
            training_results = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'epochs_trained': len(history.history['loss']),
                'model_version': self.model_version
            }
            
            logger.info(f"Training completed - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            return training_results
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
    
    def predict(self, data: pd.DataFrame, days_ahead: int = 30) -> Dict:
        """
        Make predictions using the trained model
        
        Args:
            data (pd.DataFrame): Input data for prediction
            days_ahead (int): Number of days to predict ahead
        
        Returns:
            Dict: Predictions with confidence intervals
        """
        try:
            if not self.is_trained or self.model is None:
                raise ValueError("Model must be trained before making predictions")
            
            logger.info(f"Making predictions for {days_ahead} days ahead")
            
            # Prepare the last sequence for prediction
            feature_data = data[self.features].values
            scaled_features = self.scaler.transform(feature_data)
            
            # Get the last sequence
            last_sequence = scaled_features[-self.sequence_length:].reshape(1, self.sequence_length, len(self.features))
            
            predictions = []
            confidence_intervals = []
            
            # Make predictions for each day
            current_sequence = last_sequence.copy()
            
            for day in range(days_ahead):
                # Predict next value
                pred_scaled = self.model.predict(current_sequence, verbose=0)
                pred_price = self.target_scaler.inverse_transform(pred_scaled)[0, 0]
                
                # Calculate confidence interval (simplified approach)
                # In a more sophisticated implementation, you would use techniques like
                # Monte Carlo Dropout or ensemble methods for better uncertainty quantification
                uncertainty = abs(pred_price * 0.05)  # 5% uncertainty
                confidence_lower = pred_price - uncertainty
                confidence_upper = pred_price + uncertainty
                
                predictions.append(float(pred_price))
                confidence_intervals.append((float(confidence_lower), float(confidence_upper)))
                
                # Update sequence for next prediction
                # For simplicity, we'll use the predicted price and repeat other features
                next_features = current_sequence[0, -1, :].copy()
                next_features[0] = pred_scaled[0, 0]  # Update close_price with prediction
                
                # Shift sequence and add new prediction
                current_sequence = np.roll(current_sequence, -1, axis=1)
                current_sequence[0, -1, :] = next_features
            
            # Generate prediction dates
            last_date = pd.to_datetime(data['date'].iloc[-1])
            prediction_dates = [last_date + pd.Timedelta(days=i+1) for i in range(days_ahead)]
            
            results = {
                'predictions': predictions,
                'confidence_intervals': confidence_intervals,
                'dates': [date.strftime('%Y-%m-%d') for date in prediction_dates],
                'model_version': self.model_version
            }
            
            logger.info(f"Generated {len(predictions)} predictions")
            return results
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def evaluate(self, data: pd.DataFrame) -> Dict:
        """
        Evaluate model performance on test data
        
        Args:
            data (pd.DataFrame): Test data
        
        Returns:
            Dict: Evaluation metrics
        """
        try:
            if not self.is_trained or self.model is None:
                raise ValueError("Model must be trained before evaluation")
            
            logger.info("Evaluating model performance")
            
            # Prepare test data
            X, y = self.prepare_data(data)
            
            # Make predictions
            y_pred_scaled = self.model.predict(X, verbose=0)
            
            # Inverse transform predictions and actual values
            y_pred = self.target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            y_actual = self.target_scaler.inverse_transform(y.reshape(-1, 1)).flatten()
            
            # Calculate metrics
            mse = mean_squared_error(y_actual, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_actual, y_pred)
            mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
            
            metrics = {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'mape': float(mape),
                'model_version': self.model_version
            }
            
            logger.info(f"Evaluation completed - RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%")
            return metrics
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise
    
    def save_model(self, model_path: str, scaler_path: str):
        """
        Save the trained model and scalers
        
        Args:
            model_path (str): Path to save the model
            scaler_path (str): Path to save the scalers
        """
        try:
            if not self.is_trained or self.model is None:
                raise ValueError("Model must be trained before saving")
            
            # Save model
            self.model.save(model_path)
            
            # Save scalers
            scalers = {
                'feature_scaler': self.scaler,
                'target_scaler': self.target_scaler,
                'sequence_length': self.sequence_length,
                'features': self.features,
                'model_version': self.model_version
            }
            joblib.dump(scalers, scaler_path)
            
            logger.info(f"Model saved to {model_path}, scalers saved to {scaler_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, model_path: str, scaler_path: str):
        """
        Load a trained model and scalers
        
        Args:
            model_path (str): Path to the saved model
            scaler_path (str): Path to the saved scalers
        """
        try:
            # Load model
            self.model = load_model(model_path)
            
            # Load scalers
            scalers = joblib.load(scaler_path)
            self.scaler = scalers['feature_scaler']
            self.target_scaler = scalers['target_scaler']
            self.sequence_length = scalers['sequence_length']
            self.features = scalers['features']
            self.model_version = scalers['model_version']
            
            self.is_trained = True
            
            logger.info(f"Model loaded from {model_path}, scalers loaded from {scaler_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

