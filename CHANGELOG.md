# Changelog
All notable changes to this project will be documented in this file.

## [v0.0.1]
Initial release of the assistant within a simple {framework} UI, including basic functionalities like:
* Option to pick different chatbot personalities (e.g., friendly, professional, humorous)
* Option to pick different LLMs and LLM providers for the chatbot's responses
* Web search tool for performing simple search tasks. This tool isn't selectable by the user, but the assistant can choose to use it when needed.
* Very basic memory management involving storing key information from the conversation in-memory. This isn't persisted across sessions, nor does it have any advanced capabilities like summarization or retrieval. It's just a simple dictionary that the assistant can use to store and retrieve information during the conversation.
* Basic CI pipeline when doing a pull request to the main branch, which attempts to build the project's Docker image and lint the code using `ruff`. Failed builds or linting errors will cause the pipeline to fail, preventing the PR from being merged until the issues are resolved.