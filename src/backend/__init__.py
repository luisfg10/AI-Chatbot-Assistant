"""
Entry point for the backend application, which is called by the chatbot's UI.

Notes
-----
    * The application uses sessions to maintain state across different user interactions,
    but these are lost on restart.

    TODO: Implement persistent session management to allow users to save and load their
    chat history and agent state across sessions, even after restarting the application.

"""
from src.backend.main import app

__all__ = ["app"]
