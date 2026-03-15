import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-very-secret-key-change-in-production'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///realestate.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    MODEL_PATH = 'model/best_model.pkl'
    SCALER_PATH = 'model/scaler.pkl'
    ENCODER_PATH = 'model/encoder.pkl'
