# AI Chatbot Assistant  
This learning project consists of an AI chatbot that can have conversations with users and help them with different tasks, much like the assistants offered by OpenAI (ChatGPT) and Anthropic (Claude).  
The main challenge here is to build a fully-functional chatbot with memory, tools and other advanced capabilities from scratch and without relying on agentic AI frameworks like LangGraph or OpenAI Agents SDK to make the work easier.


## Table of Contents
- [1. Project Structure](#1-project-structure)  
- [2. Running the Project](#2-running-the-project)  
  - [2.1 Running with Docker for Development](#21-running-with-docker-for-development)  
  - [2.2 Running with Docker for Production](#22-running-with-docker-for-production)  
  - [2.3 Run from a Local Virtual Environment](#23-run-from-a-local-virtual-environment)  
- [3. How to Navigate the User Interface](#3-how-to-navigate-the-user-interface)  
- [4. Coding Standards](#4-coding-standards)  
- [5. Future Versions](#5-future-versions)  

## 1. Project Structure
The project was structured in a modular way and each core functionality of the chatbot and webpage were written to be as customizable and extensible as possible, particularly the chatbot's prompt logic.

```
.
├── app/                    # Streamlit UI components
│   ├── __init__.py
│   └── ui.py              # Main UI rendering logic
|
├── chatbot/               # Core chatbot functionality
│   ├── context/           # Prompt templates
│   │   ├── system.yaml    # System prompt configurations
│   │   └── user.yaml      # User prompt templates
│   ├── core/              # Core chatbot logic
│   │   ├── agent.py       # ChatbotAgent implementation
│   │   └── context.py     # Context management utilities
│   └── __init__.py
|
├── config/                # Configuration files
│   ├── __init__.py
│   ├── .env.example       # Template for environment variables
│   ├── app_config.py      # Application configuration
│   └── llm_config.json    # Default LLM model configurations
|
├── tests/                 # Test files (pending implementation)
│   └── __init__.py
|
├── CHANGELOG.md           # Project changelog
├── Dockerfile             # Docker configuration
├── LICENSE                # License information
├── README.md              # This file
├── main.py                # Application entry point
├── pyproject.toml         # Project dependencies and metadata
└── uv.lock                # Dependency lock file
```

### 1.1 Application Configuration  
The app's configuration is managed within the `config` directory. The `app_config.py` file contains the main application configuration logic, which loads settings from environment variables and the `llm_config.json` file. The `llm_config.json` file defines the available LLM providers and models that the chatbot can use, along with their respective configurations. Environment variables may override configurations specified in the JSON file, allowing for dynamic adjustments without modifying the codebase.

## 2. Running the Project  
This project uses `uv` from Astral as dependency manager and build tool. You can build and run the project on your machine either from a Docker container or locally with a virtual environment following the instructions below.

## 2.1 Running with Docker for Development  
This alternative builds a Docker image including the project's `dev` dependencies and current directory mounting into the container, also providing an interactive terminal for development purposes.

```bash
# Build the Docker image
docker build -f Dockerfile.dev -t ai-chatbot-assistant:dev .

# Run the container with volume mounting and port mapping
docker run --rm -v "${PWD}":/app -p 8501:8501 -it ai-chatbot-assistant:dev

# Inside the container, run the Streamlit app
uv run streamlit run main.py

# Exit container
exit
```

The `--rm` flag removes the container after exit, `-v "${PWD}":/app` mounts your current directory for live code updates, and `-p 8501:8501` maps the Streamlit port. Access the app at `http://localhost:8501`.

## 2.2 Running with Docker for Production  
This alternative builds a Docker image with only the production dependencies and runs the Streamlit app directly.

```bash
# Build the Docker image
docker build -t ai-chatbot-assistant:latest .

# Run the container
docker run --rm -p 8501:8501 ai-chatbot-assistant:latest
```

## 2.3 Run from a Local Virtual Environment  

1. Download and install `uv` into your machine by following [the instructions](https://docs.astral.sh/uv/#installation) in their official page

2. Install python **3.13.0** in your machine. You can leverage `uv` to do this by running the following command in your terminal:
```bash
uv python install 3.13.0
```

2. Follow the next steps to set up the virtual environment for the project:

```bash
# Navigate to project directory (for MacOS/Linux)
cd AI-Chatbot-Assistant

# Initiate UV project
uv init

# Pin python version to the project (or pin the interpreter some other way if you prefer)
uv python pin 3.13.0

# Create virtual environment
uv venv

# Activate virtual environment (for MacOS/Linux)
source .venv/bin/activate

# Install dependencies
uv sync

# Add an additional dependency (e.g., pandas)
uv add pandas

# Run the project
uv run streamlit run main.py
```
The application will start and automatically open in your default web browser. If it doesn't open automatically, navigate to the URL shown in the terminal output (typically `http://localhost:8501`).

To stop the server, press `Ctrl+C` in the terminal.

## 3. How to Navigate the User Interface  
**Note**: This project is currently in early stages and this section will be expanded once a stable version of the UI is implemented. For now, the advice is to run this project by yourself and explore the different features available in the UI, as it's also designed to be intuitive and user-friendly.


## 4. Coding Standards  
This project uses `ruff` for linting (and formatting if you wish), and it is included as a dev dependency in the `pyproject.toml` file. You can run it with the following command:
```bash
# Linting
uv run ruff check .

# Formatting
uv run ruff format .
```

## 5. Future Versions  
Here are some ideas to further improve this project:
* Add a web search tool to the chatbot so it can improve and ground its responses upon the user's request from the UI.  
* Add support for more LLM providers and models, including open-source and self-hosted ones. Make model selection available from the UI.  
* Add a test suite for the project using `pytest` and include in a CI pipeline alongside `ruff` checks to ensure code quality on pull requests to the main branch.  
* Add additional chatbot features, like extended thinking (simulate using a ReACT architecture) and deep research tasks.  
* Improve memory management by implementing more advanced information storage and retrieval techniques, such as vector databases or knowledge graphs.  
