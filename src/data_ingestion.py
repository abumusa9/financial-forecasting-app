import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataIngestion:
    """
    Data ingestion module for fetching financial data from Yahoo Finance
    """
    
    def __init__(self):
        self.supported_symbols = [
            'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 
            'META', 'NVDA', 'NFLX', 'ADBE', 'CRM'
        ]
    
    def fetch_historical_data(self, symbol: str, period: str = "2y") -> Optional[pd.DataFrame]:
        """
        Fetch historical stock data for a given symbol
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL')
            period (str): Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        
        Returns:
            pd.DataFrame: Historical stock data or None if error
        """
        try:
            logger.info(f"Fetching historical data for {symbol} with period {period}")
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                logger.warning(f"No data found for symbol {symbol}")
                return None
            
            # Reset index to make Date a column
            data.reset_index(inplace=True)
            
            # Rename columns to match our database schema
            data.rename(columns={
                'Open': 'open_price',
                'High': 'high_price',
                'Low': 'low_price',
                'Close': 'close_price',
                'Volume': 'volume'
            }, inplace=True)
            
            # Add symbol column
            data['symbol'] = symbol
            
            # Convert Date to date only (remove time component)
            data['date'] = data['Date'].dt.date
            
            # Select only the columns we need
            data = data[['symbol', 'date', 'open_price', 'high_price', 'low_price', 'close_price', 'volume']]
            
            logger.info(f"Successfully fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def fetch_real_time_data(self, symbol: str) -> Optional[Dict]:
        """
        Fetch real-time stock data for a given symbol
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL')
        
        Returns:
            Dict: Real-time stock data or None if error
        """
        try:
            logger.info(f"Fetching real-time data for {symbol}")
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get the most recent trading data
            hist = ticker.history(period="1d", interval="1m")
            
            if hist.empty:
                logger.warning(f"No real-time data found for symbol {symbol}")
                return None
            
            latest = hist.iloc[-1]
            
            real_time_data = {
                'symbol': symbol,
                'current_price': float(latest['Close']),
                'open_price': float(latest['Open']),
                'high_price': float(latest['High']),
                'low_price': float(latest['Low']),
                'volume': int(latest['Volume']),
                'timestamp': latest.name.isoformat(),
                'previous_close': info.get('previousClose', 0),
                'change': float(latest['Close'] - info.get('previousClose', latest['Close'])),
                'change_percent': float((latest['Close'] - info.get('previousClose', latest['Close'])) / info.get('previousClose', latest['Close']) * 100) if info.get('previousClose') else 0
            }
            
            logger.info(f"Successfully fetched real-time data for {symbol}")
            return real_time_data
            
        except Exception as e:
            logger.error(f"Error fetching real-time data for {symbol}: {str(e)}")
            return None
    
    def get_supported_symbols(self) -> List[str]:
        """
        Get list of supported stock symbols
        
        Returns:
            List[str]: List of supported symbols
        """
        return self.supported_symbols
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a symbol is supported
        
        Args:
            symbol (str): Stock symbol to validate
        
        Returns:
            bool: True if symbol is supported, False otherwise
        """
        return symbol.upper() in self.supported_symbols
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data for LSTM model training
        
        Args:
            data (pd.DataFrame): Raw stock data
        
        Returns:
            pd.DataFrame: Preprocessed data
        """
        try:
            # Sort by date
            data = data.sort_values('date').reset_index(drop=True)
            
            # Calculate technical indicators
            data['sma_5'] = data['close_price'].rolling(window=5).mean()
            data['sma_10'] = data['close_price'].rolling(window=10).mean()
            data['sma_20'] = data['close_price'].rolling(window=20).mean()
            
            # Calculate RSI
            delta = data['close_price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['rsi'] = 100 - (100 / (1 + rs))
            
            # Calculate price change percentage
            data['price_change'] = data['close_price'].pct_change()
            
            # Calculate volatility (rolling standard deviation)
            data['volatility'] = data['close_price'].rolling(window=20).std()
            
            # Drop rows with NaN values
            data = data.dropna().reset_index(drop=True)
            
            logger.info(f"Preprocessed data: {len(data)} records with technical indicators")
            return data
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            return data

