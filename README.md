# AI Chatbot Assistant  
This learning project consists of an AI chatbot that can have conversations with users and help them with different tasks, much like the assistants offered by OpenAI (ChatGPT) and Anthropic (Claude).  
The main challenge here is to build a fully-functional chatbot with memory, tools and other advanced capabilities from scratch without relying on agentic AI frameworks like OpenAI Agents SDK or LangGraph to make the work easier.

## Table of Contents
[pending]

## Creating a Virtual Environment  
This project uses `uv` from Astral as dependency manager and build tool. Follow these steps to setting up your virtual environment locally:
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

# Add a dependency (e.g., pandas)
uv add pandas
```

## Linting and Formatting
This project uses `ruff` for linting (and formatting if you wish), and it is included as a dev dependency in the `pyproject.toml` file. You can run it with the following command:
```bash
# Linting
uv run ruff check .

# Formatting
uv run ruff format .
```


## Future Versions  
Here are some ideas to further improve this project:
* Add comprehensive memory management capabilities, allowing the chatbot to remember past interactions that are persisted across sessions in a database, and use that information to provide more personalized and context-aware responses.
* Add multimodal capabilities to the chatbot, allowing it to process (and maybe generate) content in non-text formats like images, audio and video.  
* Add open-source observability with tools like Langfuse and Arize Phoenix.  
* Add a "Reasoning" option for the chatbot to think longer and reflect on a given task.
* Add a "Deep Research" option for conducting extensive, thorough research on a given project. This would involve the chatbot using multiple tools and resources to gather information, analyze it, and present comprehensive findings to the user.
* Add an option for connecting the chatbot to external MCP servers for common tasks, providing additional functionality.