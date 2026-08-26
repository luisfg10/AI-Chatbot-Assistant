"""
Backend app entrypoint called by the chatbot's UI.

Notes
-----
    - App uses sessions to maintain state across different user interactions,
    but these are lost on restart.
    TODO: Log user sessions on databases for session persistence.
"""
from src.backend.main import app

__all__ = ["app"]
