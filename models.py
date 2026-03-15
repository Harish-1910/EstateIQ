"""
models.py
---------
SQLAlchemy database models for EstateIQ.
Imported by app.py inside create_app() to avoid circular imports.

Tables:
  users       → registered accounts
  predictions → every valuation made by each user
"""
from extensions import db
from flask_login import UserMixin
from datetime import datetime


class User(db.Model, UserMixin):
    __tablename__ = 'users'

    id         = db.Column(db.Integer,     primary_key=True)
    username   = db.Column(db.String(80),  unique=True, nullable=False)
    email      = db.Column(db.String(120), unique=True, nullable=False)
    password   = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime,    default=datetime.utcnow)

    predictions = db.relationship('Prediction', backref='user', lazy=True)

    def __repr__(self):
        return f'<User {self.username}>'


class Prediction(db.Model):
    __tablename__ = 'predictions'

    id              = db.Column(db.Integer,     primary_key=True)
    user_id         = db.Column(db.Integer,     db.ForeignKey('users.id'), nullable=False)
    location        = db.Column(db.String(100))
    city            = db.Column(db.String(50),  default='')
    total_sqft      = db.Column(db.Float)
    bhk             = db.Column(db.Integer)
    bath            = db.Column(db.Integer)
    predicted_price = db.Column(db.Float)
    created_at      = db.Column(db.DateTime,    default=datetime.utcnow)

    def __repr__(self):
        return f'<Prediction {self.id} {self.city} {self.predicted_price}L>'