from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.backend.main import agent_store, app, get_agent
from src.chatbot import ChatbotAssistant

# ------------------------------------------------------------------
# Fixtures


@pytest.fixture(autouse=True)  # activate for all tests that see it
def _clear_agent_store() -> Any:
    """
    Ensure the global session store starts and ends each test empty.

    Notes
    -----
    `agent_store` is a plain module-level dict in `src.backend.main`.
    Because the `app` object (and therefore this dict) is imported once
    and reused for the whole test session, any session entries left behind
    by one test would otherwise still be visible to the next test.

    The clear after `yield` runs as teardown, after the test body has
    finished, so the next test starts from an empty store even if the
    current test failed partway through (pytest still runs code after
    `yield` on failure, similar to a `finally` block).

    `yield` is what turns this into a "setup/teardown" fixture: code before
    `yield` runs before the test, the test then runs, and code after `yield`
    runs once the test is done, regardless of its result.
    """
    yield
    agent_store.clear()


@pytest.fixture
def mock_agent() -> MagicMock:
    """
    Build a mocked `ChatbotAssistant` with realistic response attributes.

    Notes
    -----
    `MagicMock` (from `unittest.mock`, the standard library) creates a
    fake object that accepts any attribute access or method call and
    records how it was used, without running any real code.
    This is different from the `mocker` fixture provided by `pytest-mock`:
    `mocker.patch(...)` *replaces* an existing attribute/function/class
    on a real object for the duration of a test, whereas
    `MagicMock(spec=ChatbotAssistant)` here builds a brand new
    fake object that stands in for a whole `ChatbotAssistant` instance,
    so no real agent (and therefore no real `OpenAI` client) ever gets
    constructed.

    `spec=ChatbotAssistant` restricts the mock to only the attributes and
    methods that actually exist on `ChatbotAssistant`. Without `spec`, a
    typo like `agent.chatbot_cal(...)` would silently succeed and return
    another `MagicMock` instead of raising an `AttributeError`.

    Attributes like `models` and `default_model` are set to plain values
    because the endpoints read them directly. `__call__` is a dunder method,
    so its behavior is configured via `.return_value`: any call
    to `mock_agent(...)` in a test returns this canned string, and the
    call arguments can later be asserted with
    `mock_agent().assert_called_once_with(...)`.
    """
    agent = MagicMock(spec=ChatbotAssistant)
    agent.models = {"gpt-5-mini": {}, "gpt-5.5": {}}
    agent.default_model = "gpt-5-mini"
    agent.supported_personalities = ["friendly", "professional"]
    agent.default_personality = "friendly"
    agent.available_tools = [
        "evaluate_math_expression",
        "get_current_date"
    ]
    agent.tool_registry = {"evaluate_math_expression": MagicMock()}
    # return_value is a special term to replace __call__ on MagicMock
    agent.return_value = "Hello, how can I help?"
    return agent


@pytest.fixture
def authed_client(mock_agent: MagicMock) -> Any:
    """
    Build a client where `get_agent` is overridden to return `mock_agent`.

    Parameters
    ----------
        mock_agent: MagicMock
            The MagicMock for tne agent, also a fixture.

    Notes
    -----
        `app.dependency_overrides` is a dict FastAPI checks before resolving
        any `Depends(...)`. Its keys are the original dependency callables
        and its values are the replacement callables to use instead.

        Setting `app.dependency_overrides[get_agent] = lambda: mock_agent`
        tells FastAPI: "whenever a route depends on `get_agent`, call this lambda
        instead and use its return value."
        This bypasses the real `get_agent` (the cookie/`agent_store` lookup
        and the 401 check), so every route in this client behaves as if a valid,
        already-authenticated session were present, without needing to drive
        a real cookie through the homepage endpoint first.

        `yield` is used (instead of `return`) so that code can run *after*
        the test that uses this fixture finishes, i.e. the line after
        `yield` is teardown code, executed once the test body completes
        (pass or fail).

        `app.dependency_overrides.clear()` in that teardown step is
        necessary because `app` is the same singleton `FastAPI` instance
        shared by every test. Without clearing it, an override set by one
        test would still be active for the next test (which might expect the
        real `get_agent` behavior, e.g. the 401 tests in
        `TestAuthRequired`), causing tests to interfere with each other
        depending on execution order.
    """
    # Mock "get_agent" dependency
    app.dependency_overrides[get_agent] = lambda: mock_agent

    # Run test app
    yield TestClient(app)

    # Teardown after running tests
    app.dependency_overrides.clear()


@pytest.fixture
def client() -> TestClient:
    """Build a plain client with no session cookie or dependency overrides."""
    return TestClient(app)


# ------------------------------------------------------------------
# Testing Classes


class TestHomepage:
    """Test the `/` endpoint's session and cookie bookkeeping."""

    def test_serves_index_html_with_200(self, client: TestClient) -> None:
        """Test the homepage responds successfully with HTML content."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_no_cookie_creates_session_and_agent_and_sets_cookie(
        self, client: TestClient, mocker: Any
    ) -> None:
        """Test a first-time visit sets a session cookie and stores a new agent."""
        # Patch chatbot to avoid actual init
        mocker.patch("src.backend.main.ChatbotAssistant")

        # Call root
        response = client.get("/")

        # Validate response
        assert "session_id" in response.cookies
        assert len(agent_store) == 1

    def test_known_session_cookie_reuses_existing_agent(
        self, client: TestClient, mocker: Any, mock_agent: MagicMock
    ) -> None:
        """Test a known session cookie reuses the stored agent unchanged."""
        # Avoid chabot init on root call
        mocker.patch("src.backend.main.ChatbotAssistant")

        # Add mock agent to store
        agent_store["known-session"] = mock_agent

        # Set fake session id and call
        client.cookies.set("session_id", "known-session")
        response = client.get("/")

        # Validate mock agent is set to store
        assert agent_store["known-session"] is mock_agent
        assert "session_id" not in response.cookies


class TestAuthRequired:
    """Test that `/api/*` routes reject requests without a valid session."""

    @pytest.mark.parametrize(
        ("method", "path", "json_body"),
        [
            ("get", "/api/models", None),
            ("get", "/api/personalities", None),
            ("get", "/api/tools", None),
            ("post", "/api/reset", None),
            ("post", "/api/chat", {"message": "Hi"}),
            ("post", "/api/config/model", {"model": "gpt-5.5"}),
            ("post", "/api/config/personality", {"personality": "professional"}),
            ("post", "/api/config/tools", {"tool_names": ["get_current_date"]}),
        ],
    )
    def test_missing_cookie_returns_401(
        self,
        client: TestClient,
        method: str,
        path: str,
        json_body: dict | None
    ) -> None:
        """
        Test every protected route rejects requests without a session cookie.

        Parameters
        ----------
            client: TestClient
                Fixture with the test FastAPI app.
            method: str
                HTTP method to use on the endpoint.
            path: str
                The endpoint to call for the test API.
            json_body: dict | None
                Optional body to send as part of the request.
        """
        kwargs = {"json": json_body} if json_body is not None else {}
        response = getattr(client, method)(path, **kwargs)
        assert response.status_code == 401

    def test_unknown_session_cookie_returns_401(self, client: TestClient) -> None:
        """Test a session cookie not present in the store is rejected."""
        client.cookies.set("session_id", "unknown-session")
        response = client.get("/api/models")
        assert response.status_code == 401


class TestGetModels:
    """Test the `/api/models` endpoint."""

    def test_returns_agent_models_and_default(
            self,
            authed_client: TestClient
    ) -> None:
        """Test the response reflects the agent's available and default models."""
        response = authed_client.get("/api/models")
        assert response.status_code == 200
        assert response.json() == {  # Should match the "mock_agent" fixture's data
            "models": ["gpt-5-mini", "gpt-5.5"],
            "default_model": "gpt-5-mini",
        }


class TestGetPersonalities:
    """Test the `/api/personalities` endpoint."""

    def test_returns_agent_personalities_and_default(
            self,
            authed_client: TestClient
    ) -> None:
        """Test the response reflects the agent's personality metadata."""
        response = authed_client.get("/api/personalities")
        assert response.status_code == 200
        assert response.json() == {
            "personalities": ["friendly", "professional"],
            "default_personality": "friendly",
        }


class TestGetTools:
    """Test the `/api/tools` endpoint."""

    def test_returns_available_and_selected_tools(
            self,
            authed_client: TestClient,
            mock_agent: MagicMock
    ) -> None:
        """Test the response reflects the agent's available and selected tools."""
        response = authed_client.get("/api/tools")
        assert response.status_code == 200
        assert response.json() == {  # Should match the "mock_agent" fixture's data
            "tools": mock_agent.available_tools,
            "selected_tools": list(mock_agent.tool_registry.keys()),
        }


class TestResetMemory:
    """Test the `/api/reset` endpoint."""

    def test_reset_calls_agent_reset_memory(
        self, authed_client: TestClient, mock_agent: MagicMock
    ) -> None:
        """Test the endpoint delegates to the agent's `reset_memory` method."""
        # Call reset endpoint
        response = authed_client.post("/api/reset")

        # Check response is ok
        assert response.json() == {"ok": True}

        # Check reset_memory method was called inside mock agent
        mock_agent.reset_memory.assert_called_once()


class TestChat:
    """Test the `/api/chat` endpoint."""

    def test_chat_returns_llm_response(
        self,
        authed_client: TestClient,
        mock_agent: MagicMock
    ) -> None:
        """Test the endpoint returns the agent's response for a valid message."""
        response = authed_client.post("/api/chat", json={"message": "Hi"})
        assert response.status_code == 200
        assert response.json() == {"response": "Hello, how can I help?"}
        mock_agent.assert_called_once_with("Hi")

    def test_chat_missing_message_field_returns_422(
        self, authed_client: TestClient
    ) -> None:
        """Test a request body missing `message` fails schema validation."""
        response = authed_client.post("/api/chat", json={})
        assert response.status_code == 422


class TestConfigModel:
    """Test the `/api/config/model` endpoint."""

    def test_set_model_valid_calls_set_client(
        self,
        authed_client: TestClient,
        mock_agent: MagicMock
    ) -> None:
        """Test the endpoint delegates the requested model to `set_client`."""
        response = authed_client.post(
            "/api/config/model",
            json={"model": "gpt-5.5"}
        )
        assert response.json() == {"ok": True}
        mock_agent.set_client.assert_called_once_with("gpt-5.5")

    def test_set_model_invalid_model_returns_400(
        self,
        authed_client: TestClient,
        mock_agent: MagicMock
    ) -> None:
        """Test an unknown model code surfaces as a 400 Bad Request."""
        # Mimick KeyError on agent's side
        mock_agent.set_client.side_effect = KeyError("bad-model")

        # Call endpoint
        response = authed_client.post(
            "/api/config/model",
            json={"model": "unexistent-model"}
        )

        # Validate andpoint response
        mock_agent.set_client.assert_called_once_with("unexistent-model")
        assert response.status_code == 400
        assert "not found" in response.json().get("detail")


class TestConfigPersonality:
    """Test the `/api/config/personality` endpoint."""

    def test_set_personality_calls_agent(
        self,
        authed_client: TestClient,
        mock_agent: MagicMock
    ) -> None:
        """Test endpoint delegates call to `agent.set_personality`."""
        response = authed_client.post(
            "/api/config/personality",
            json={"personality": "professional"}
        )
        assert response.json() == {"ok": True}
        mock_agent.set_personality.assert_called_once_with("professional")


class TestConfigTools:
    """Test the `/api/config/tools` endpoint."""

    def test_set_tools_calls_agent_with_tool_names(
        self,
        authed_client: TestClient,
        mock_agent: MagicMock
    ) -> None:
        """Test endpoint delegates the requested tool names to `agent.set_tools`."""
        response = authed_client.post(
            "/api/config/tools",
            json={"tool_names": ["get_current_date"]}
        )
        assert response.json() == {"ok": True}
        mock_agent.set_tools.assert_called_once_with(
            tool_names=["get_current_date"]
        )

    def test_set_tools_missing_field_defaults_to_empty_list(
        self,
        authed_client: TestClient,
        mock_agent: MagicMock
    ) -> None:
        """Test a missing `tool_names` field disables all tools instead of erroring."""
        response = authed_client.post("/api/config/tools", json={})
        assert response.json() == {"ok": True}
        mock_agent.set_tools.assert_called_once_with(tool_names=[])
