"""
Module for the AI Chatbot Assistant application.

Notes
-----
    TODO: Implement persistent session management to allow users to save and load their
    chat history across sessions, even after restarting the application.
    Existing sessions should be visible on a sidetab in the UI, and on each new chat
    an async parallel task should be created to name and save the session.

    TODO: For multi-turn responses, like those requiring tool calls, show the user
    a brief summary in the UI of the tools that were called by the LLM in order for
    it to generate a response.

    TODO: When an error happens in the application, show a user-friendly error
    message, possibly with an error ID, instead of a full-stack trace. On the developer's
    side, the full error should be logged in stderr, as well as in a log file.

    TODO: Stream LLM responses in the UI for a more natural conversational experience.

    TODO: Allow users to go back in the flow of the conversation and edit their message,
    which restarts the conversation from that point.

    TODO: Allow users to upload images, files and voice messages.

    TODO: Allow users to highlight a part of an LLM's response and ask a follow-up
    question about it, while keeping the context of the conversation.

    TODO: Deep Research capability using an ideator-reviewer-critic agent architecture.
    To do this, the agent must at least have a web search tool available, and optionally
    be able to spin up several sub-agents in parallel to reduce latency.

    TODO: Add a means for user feedback, which can also serve for LLM evaluation:
    on each message, users should be able to rate the quality of the message
    (thumbs up or thumbs down). This feedback should be saved and used to improve the
    agent.

    TODO: Allow users to add/remove models from the models list manually, and
    persist their choices across sessions.

    TODO: On each message, show users whether the agent used any tools, and which.
"""
from src.backend import app

__all__ = ["app"]
