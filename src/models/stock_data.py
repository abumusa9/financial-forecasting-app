from src.database import db
from datetime import datetime

class StockData(db.Model):
    __tablename__ = 'stock_data'
    
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(10), nullable=False, index=True)
    date = db.Column(db.Date, nullable=False, index=True)
    open_price = db.Column(db.Float, nullable=False)
    high_price = db.Column(db.Float, nullable=False)
    low_price = db.Column(db.Float, nullable=False)
    close_price = db.Column(db.Float, nullable=False)
    volume = db.Column(db.BigInteger, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'symbol': self.symbol,
            'date': self.date.isoformat(),
            'open_price': self.open_price,
            'high_price': self.high_price,
            'low_price': self.low_price,
            'close_price': self.close_price,
            'volume': self.volume,
            'created_at': self.created_at.isoformat()
        }

class ModelPrediction(db.Model):
    __tablename__ = 'model_predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(10), nullable=False, index=True)
    prediction_date = db.Column(db.Date, nullable=False, index=True)
    predicted_price = db.Column(db.Float, nullable=False)
    confidence_lower = db.Column(db.Float, nullable=False)
    confidence_upper = db.Column(db.Float, nullable=False)
    model_version = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'symbol': self.symbol,
            'prediction_date': self.prediction_date.isoformat(),
            'predicted_price': self.predicted_price,
            'confidence_lower': self.confidence_lower,
            'confidence_upper': self.confidence_upper,
            'model_version': self.model_version,
            'created_at': self.created_at.isoformat()
        }

