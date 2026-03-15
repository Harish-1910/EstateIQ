from flask import Flask
from extensions import db, bcrypt, login_manager
from config import Config


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    db.init_app(app)
    bcrypt.init_app(app)
    login_manager.init_app(app)

    with app.app_context():
        from models import User

        @login_manager.user_loader
        def load_user(user_id):
            # Use Session.get() — avoids SQLAlchemy 2.0 LegacyAPIWarning
            return db.session.get(User, int(user_id))

        from auth import auth_bp
        from routes import main_bp
        app.register_blueprint(auth_bp)
        app.register_blueprint(main_bp)

        db.create_all()

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)