"""
tests.py
Basic test suite for EstateIQ Flask app.

Run:
    pytest tests.py -v
"""
import pytest
from app import create_app, db
from models import User


@pytest.fixture
def app():
    """Create test Flask app with in-memory SQLite."""
    app = create_app()
    app.config.update({
        'TESTING': True,
        'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:',
        'WTF_CSRF_ENABLED': False,
        'SECRET_KEY': 'test-secret-key',
    })
    with app.app_context():
        db.create_all()
        yield app
        db.drop_all()


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def runner(app):
    return app.test_cli_runner()


# ── Helper ────────────────────────────────────────────────────────────────────
def register_user(client, username='testuser', email='test@example.com', password='testpass123'):
    return client.post('/register', data={
        'username': username,
        'email': email,
        'password': password,
        'confirm_password': password,
    }, follow_redirects=True)


def login_user(client, email='test@example.com', password='testpass123'):
    return client.post('/login', data={
        'email': email,
        'password': password,
    }, follow_redirects=True)


# ── Auth Tests ────────────────────────────────────────────────────────────────
class TestAuth:
    def test_register_page_loads(self, client):
        res = client.get('/register')
        assert res.status_code == 200
        assert b'Create account' in res.data

    def test_login_page_loads(self, client):
        res = client.get('/login')
        assert res.status_code == 200
        assert b'Welcome back' in res.data

    def test_register_new_user(self, client):
        res = register_user(client)
        assert res.status_code == 200
        assert b'log in' in res.data.lower() or b'sign in' in res.data.lower()

    def test_register_duplicate_email(self, client):
        register_user(client)
        res = register_user(client, username='other')
        assert b'already' in res.data.lower()

    def test_register_password_mismatch(self, client):
        res = client.post('/register', data={
            'username': 'user2',
            'email': 'user2@example.com',
            'password': 'abc123',
            'confirm_password': 'wrong',
        }, follow_redirects=True)
        assert b'match' in res.data.lower()

    def test_login_valid(self, client):
        register_user(client)
        res = login_user(client)
        assert b'dashboard' in res.data.lower() or b'welcome' in res.data.lower()

    def test_login_invalid_password(self, client):
        register_user(client)
        res = client.post('/login', data={
            'email': 'test@example.com',
            'password': 'wrongpassword',
        }, follow_redirects=True)
        assert b'invalid' in res.data.lower()

    def test_logout(self, client):
        register_user(client)
        login_user(client)
        res = client.get('/logout', follow_redirects=True)
        assert res.status_code == 200


# ── Route Access Tests ────────────────────────────────────────────────────────
class TestRoutes:
    def test_dashboard_requires_login(self, client):
        res = client.get('/dashboard', follow_redirects=False)
        assert res.status_code == 302  # redirect to login

    def test_predict_requires_login(self, client):
        res = client.get('/predict', follow_redirects=False)
        assert res.status_code == 302

    def test_history_requires_login(self, client):
        res = client.get('/history', follow_redirects=False)
        assert res.status_code == 302

    def test_dashboard_accessible_after_login(self, client):
        register_user(client)
        login_user(client)
        res = client.get('/dashboard')
        assert res.status_code == 200

    def test_predict_page_loads_after_login(self, client):
        register_user(client)
        login_user(client)
        res = client.get('/predict')
        assert res.status_code == 200

    def test_history_page_loads_after_login(self, client):
        register_user(client)
        login_user(client)
        res = client.get('/history')
        assert res.status_code == 200


# ── API Tests ─────────────────────────────────────────────────────────────────
class TestAPI:
    def test_api_predict_requires_auth(self, client):
        res = client.post('/api/predict', json={
            'location': 'Whitefield',
            'total_sqft': 1200,
            'bhk': 3,
            'bath': 2,
        })
        assert res.status_code == 302  # redirect to login

    def test_api_locations_requires_auth(self, client):
        res = client.get('/api/locations')
        assert res.status_code == 302

    def test_api_predict_authenticated(self, client):
        register_user(client)
        login_user(client)
        res = client.post('/api/predict', json={
            'location': 'Whitefield',
            'total_sqft': 1200,
            'bhk': 3,
            'bath': 2,
        })
        # Will return error if model not trained — that's expected in test env
        assert res.status_code in (200, 400)


# ── Model Tests ───────────────────────────────────────────────────────────────
class TestDatabase:
    def test_user_creation(self, app):
        with app.app_context():
            from app import bcrypt
            u = User(
                username='alice',
                email='alice@test.com',
                password=bcrypt.generate_password_hash('pass').decode('utf-8')
            )
            db.session.add(u)
            db.session.commit()
            found = User.query.filter_by(email='alice@test.com').first()
            assert found is not None
            assert found.username == 'alice'

    def test_user_password_hashing(self, app):
        with app.app_context():
            from app import bcrypt
            pw = 'secretpass'
            hashed = bcrypt.generate_password_hash(pw).decode('utf-8')
            assert bcrypt.check_password_hash(hashed, pw)
            assert not bcrypt.check_password_hash(hashed, 'wrongpass')