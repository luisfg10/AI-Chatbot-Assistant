# Changelog
All notable changes to this project will be documented in this file.

## [v0.0.2]  
* Added the capability for the user to select the LLM directly from the UI, allowing for more flexibility and ease of use without needing to modify environment variables. The list of available LLMs is automatically resolved from a combination of `config/llm_config.json` and the API keys provided from environment variables. Models set in the congiguration for which no API key is provided will be hidden from the dropdown list in the UI.  
* Added support for Anthropic and Grok models.  

## [v0.0.1]
Initial version of the assistant within a simple Streamlit UI including basic functionalities like:
* Option to pick different chatbot personalities from the UI (e.g., friendly, professional), as well as manually-resetting the conversation's context.
* Ability to select different LLM providers and models from environment variables. This will eventually be moved to a UI-based configuration.
* Basic memory management involving storing key information from the conversation in-memory and perodically running summarization on recent conversation turns to keep memory concise and relevant. This information isn't saved in permanent storage, so it is lost when the assistant or the webpage are restarted.