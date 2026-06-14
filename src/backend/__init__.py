"""
Entry point for the backend application, which is called by the chatbot's UI.

Notes
-----
    * The application uses sessions to maintain state across different user interactions,
    but these are lost on restart.

    TODO: Reduce the verbosity of the application logs to essential information, or at
    least control behavior from an environment variable.

"""
from src.backend.main import app

__all__ = ["app"]
