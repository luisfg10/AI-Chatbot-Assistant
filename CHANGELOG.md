# Changelog
All notable changes to this project will be documented in this file.

## [v0.0.1]
Initial version of the assistant within a simple Streamlit UI, including basic functionalities like:
* Option to pick different chatbot personalities (e.g., friendly, professional)
* Ability to select different LLM providers and models from environment variables (not from the UI yet)
* Basic memory management involving storing key information from the conversation in-memory and perodically running summarization on recent conversation turns to keep memory concise and relevant. This information isn't saved in permanent storage, so it is lost when the assistant is restarted.