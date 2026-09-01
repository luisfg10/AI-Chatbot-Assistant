# AI Chatbot Assistant  

This project consists of an AI agent served on a graphical user interface that's able to have conversations with users and execute tasks on their behalf, like searching the web or performing math calculations.  

The project does not make use of highly-abstracted agentic frameworks to facilitate tasks like tool use and memory management: the whole point of the exercise is building these capabilities from scratch.  

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
The project was structured in a modular way and each core functionality of the chatbot and webpage were written to be as customizable and extensible as possible.  

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
│   │   │   ├── base_chat_completions.py
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

### 1.1 App Configuration  
The app's configuration is managed within the `config` directory. The `app_config.py` file contains the main application configuration logic, which loads settings from environment variables and the `llm_config.json` file. The `llm_config.json` file defines the available LLM providers and models that the chatbot can use, along with their respective configurations.  

## 2. Running the Project  
This project uses Astral's `uv` as dependency manager and build tool. You can build and run the project on your machine either from a Docker container or locally with a virtual environment following the instructions below.  

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

2. Install python **3.14** in your machine. You can leverage `uv` to do this by running the following command in your terminal:
```bash
uv python install 3.14
```

2. Follow the next steps to set up the virtual environment for the project:

```bash
# Navigate to project directory (for MacOS/Linux)
cd AI-Chatbot-Assistant

# Initiate UV project
uv init

# Pin python version to the project (or pin the interpreter some other way if you prefer)
uv python pin 3.14

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
The application will start and run on `http://localhost:8000`. To stop the server, press `Ctrl+C` in the terminal.

## 3. How to Navigate the User Interface  
**Note**: This section will be provided with further detail in the future.    


## 4. Coding Standards  
This project uses `ruff` for linting and ensuring good coding standards are upheld, and it is included as a dev dependency in the `pyproject.toml` file. You can run it with the following command:

```bash
# Linting
uv run ruff check .
```

The project also has a test suite on the `tests\` directory for ensuring that the app works correctly after changes. You can run this using:  

```bash
uv run pytest
```

The project also has a CI pipeline that runs as a GitHub action on each pull request. This pipeline checks both the test suite and ruff linters pass before allowing PRs to be merged. In order to check consistency locally before pushing, you can set up a pre-push hook to your local git directory using the command:

```bash
./config/hooks/setup.sh
```
