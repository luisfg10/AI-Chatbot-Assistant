FROM python:3.13.0-slim

# Install uv as dep manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency files first (for layer caching)
COPY pyproject.toml uv.lock ./

# Install production dependencies only
RUN uv sync --frozen --no-dev

# Copy application code
COPY . .

# Run the application using uv
CMD ["uv", "run", "streamlit", "run", "main.py"]