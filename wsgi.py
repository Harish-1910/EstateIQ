"""
wsgi.py
Production entry point — used by Gunicorn.

Run locally:
    gunicorn wsgi:app

Or use the Procfile which calls create_app() directly.
"""
from app import create_app

app = create_app()

if __name__ == '__main__':
    app.run()