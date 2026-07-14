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
├── .dockerignore
├── .github/               # GitHub-specific files and workflows
├── .gitignore
├── .python-version
├── CHANGELOG.md           # Project changelog
├── Dockerfile             # Production Docker configuration
├── Dockerfile.dev         # Development Docker configuration
├── LICENSE                # License information
├── README.md              # Project documentation
├── main.py                # Application entry point
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Dependency lock file
├── config/                # Runtime configuration files
│   ├── __init__.py
│   ├── app_config.py      # Application configuration logic
│   └── llm_config.json    # Default settings
├── src/                   # Application source code
│   ├── __init__.py
│   ├── backend/           # Backend API logic
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── schemas.py
│   ├── chatbot/           # Chatbot implementation
│   │   ├── __init__.py
│   │   ├── context/       # Prompt templates
│   │   │   ├── system.yaml
│   │   │   └── user.yaml
│   │   ├── core/          # Core agent logic
│   │   │   ├── agent.py
│   │   │   ├── base_agent.py
│   │   │   └── context.py
│   │   └── tools/         # Chatbot tools
│   │       ├── __init__.py
│   │       ├── builder.py
│   │       ├── definitions.py
│   │       ├── math.py
│   │       └── web_search.py
│   └── frontend/          # Static frontend assets
│       └── static/
│           ├── app.js
│           ├── index.html
│           └── style.css
└── tests/                 # Automated test suite
    ├── backend/
    └── chatbot/
        ├── core/
        │   ├── test_agent.py
        │   ├── test_base_agent.py
        │   └── test_context_manager.py
        └── tools/
            ├── test_current_date_tool.py
            ├── test_math_tool.py
            └── test_web_search_tool.py
```

Local and generated artifacts such as `.venv`, `.pytest_cache`, `.ruff_cache`, `__pycache__`, and `.env` are intentionally omitted from the tree above.

### 1.1 Application Configuration  
The app's configuration is managed within the `config` directory. The `app_config.py` file contains the main application configuration logic, which loads settings from environment variables and the `llm_config.json` file. The `llm_config.json` file defines the available LLM providers and models that the chatbot can use, along with their respective configurations. Environment variables may override configurations specified in the JSON file, allowing for dynamic adjustments without modifying the codebase.

## 2. Running the Project  
This project uses `uv` from Astral as dependency manager and build tool. You can build and run the project on your machine either from a Docker container or locally with a virtual environment following the instructions below.

## 2.1 Running with Docker for Development  
This alternative builds a Docker image including the project's `dev` dependencies and current directory mounting into the container, also providing an interactive terminal for development purposes.

```bash
# Build the Docker image
docker build -f Dockerfile.dev -t ai-chatbot-assistant:dev .

# Run the container with volume mounting and port mapping (MacOS / Linux / Windows PS)
docker run --rm -v "${PWD}:/app:ro" -v /app/.venv -p 8000:8000 -it ai-chatbot-assistant:dev

# Inside the container, run the app
uv run main.py

# Exit container
exit
```

The `--rm` flag removes the container after exit, `-v "${PWD}":/app:ro` mounts your current directory for live code updates in read-only mode (ro), `-v / app/.venv` creates an anonymous volume that shadows the virtual environment inside the container, so `uv` writes there, and `-p 8000:8000` maps the port. Access the app at `http://localhost:8000`.

## 2.2 Running with Docker for Production  
This alternative builds a Docker image with only the production dependencies and runs the app directly.

```bash
# Build the Docker image
docker build -t ai-chatbot-assistant:latest .

# Run the container
docker run --rm -p 8000:8000 ai-chatbot-assistant:latest
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
# In Windows PS, run .\.venv\Scripts\Activate.ps1
source .venv/bin/activate

# Install dependencies
uv sync

# Optional: add an additional dependency (e.g., pandas)
uv add pandas

# Run the project
uv run main.py
```
The application will start and run on `http://localhost:8000`.

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
