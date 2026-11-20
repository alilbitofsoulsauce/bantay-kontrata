"""WSGI entrypoint for Gunicorn, Waitress, and cloud PaaS deployments."""

from app import create_app

app = create_app()
