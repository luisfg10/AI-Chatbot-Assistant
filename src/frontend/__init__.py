"""
Entry point for the application's frontend logic.

The frontend is now served as static files from the FastAPI backend.
"""
from src.frontend.streamlit_ui import build_streamlit_ui

__all__ = ["build_streamlit_ui"]
