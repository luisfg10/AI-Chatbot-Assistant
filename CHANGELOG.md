# Changelog
All notable changes to this project will be documented in this file.  

## [v0.0.5] (WIP)  
* Added basic agent tools: it can fetch today's date and perform basic arithmetic operations (add, substract, multiply, divide, exponentiate).  
* Added a web search tool to the agent using Tavily's search endpoint. This tool is only available and visible to the agent if the environment variable `TAVILY_API_KEY` is provided on service startup.  


## [v0.0.4]  
* Significant change in memory management: The messages list now has an appended message on each significant piece of information (base instructions, user message, chatbot response, tool call decision and execution). This is a necessary change before the introduction of actual tool calls, and aligns the project with commonly-used standards.     
* Added context file caching in `src/chatbot/core/context.py` to avoid loading and parsing the YAML files from scratch on every context fetch.  
* Deprecated Streamlit UI and removed as dependency for the project.  
* Updated instructions in `README.md` when running `Dockerfile.dev`: the container can now only read the project's directory without being able to write to it. Previously, this caused issues when running from a Windows machine due to symlink conflicts between the container (Linux subsystem) and Windows that prevented `uv` from running correctly.  
* Added a new environment variable, `LOG_LEVEL`, by which it's possible to control the minimum hierarchy level in which logs from `loguru.logger` are sent to stderr.  


## [v0.0.3]  
* Replaced the Streamlit UI with a Single-Page-Application (SPA) using HTML, JavaScript and CSS connected to a `FastAPI` backend which handles interactions to the chatbot agent class. This change provides greater flexibility and scalability as the app grows in capabilities going forward. Streamlit is still included in the project but will be deprecated in a future version.   
* Updated AI coding assistant material on `.github/`: Updated `copilot-instructions.md`, and added a new `skills/` directory with two skills: [feature-dev](https://github.com/notedit/happy-skills/blob/main/skills/dev/feature-dev/SKILL.md) and [frontend-design](https://github.com/anthropics/skills/blob/main/skills/frontend-design/SKILL.md), with some minor tweaks for this project's specific implementation.    
* Minor changes to ruff linting rules and code ordering.    

## [v0.0.2]  
* Added the capability for the user to select the LLM directly from the UI, allowing for more flexibility and ease of use without needing to modify environment variables. The list of available LLMs is automatically resolved from a combination of `config/llm_config.json` and the API keys provided from environment variables. Models set in the congiguration for which no API key is provided will be hidden from the dropdown list in the UI.  
* Added support for Anthropic and Grok models.  

## [v0.0.1]
Initial version of the assistant within a simple Streamlit UI including basic functionalities like:
* Option to pick different chatbot personalities from the UI (e.g., friendly, professional), as well as manually-resetting the conversation's context.
* Ability to select different LLM providers and models from environment variables. This will eventually be moved to a UI-based configuration.
* Basic memory management involving storing key information from the conversation in-memory and perodically running summarization on recent conversation turns to keep memory concise and relevant. This information isn't saved in permanent storage, so it is lost when the assistant or the webpage are restarted.