from flask_sqlalchemy import SQLAlchemy
from flask import Flask

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///detections.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Detection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    emotion = db.Column(db.String(50))
    head_movement = db.Column(db.String(50))
    objects = db.Column(db.String(255))
    timestamp = db.Column(db.DateTime, server_default=db.func.now())

# Create tables if they don't exist
with app.app_context():
    db.create_all()
