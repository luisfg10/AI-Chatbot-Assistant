"""Backend app endpoint definitions."""
from uuid import uuid4

from fastapi import (
    APIRouter,
    Cookie,
    Depends,
    FastAPI,
    HTTPException,
)
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.backend.schemas import (
    AvailableModelsResponse,
    AvailablePersonalitiesResponse,
    ChatRequest,
    ChatResponse,
)
from src.chatbot import ChatbotAssistant

app = FastAPI()
# Store agents being used per each specific session id (cookie)
agent_store: dict[str, ChatbotAssistant] = {}

# ------------------------------------------------------------------
# Homepage and Dependencies

# Mount static frontend files
app.mount(
    "/static",
    StaticFiles(directory="src/frontend/static"),
    name="static"
)


@app.get("/")
async def homepage(
    session_id: str = Cookie(default=None)
) -> FileResponse:
    """Serve the static frontend and save unique session IDs using cookies."""
    response = FileResponse("src/frontend/static/index.html")
    # Set cookie if it doesn't exist
    if not session_id:
        session_id = str(uuid4())
        response.set_cookie(
            key="session_id",
            value=session_id,
            path="/",
            httponly=True
        )
        agent_store[session_id] = ChatbotAssistant()
    # Assign agent to session
    elif session_id not in agent_store:
        agent_store[session_id] = ChatbotAssistant()

    return response


def get_agent(session_id: str = Cookie(default=None)) -> ChatbotAssistant:
    """
    Retrieve the chatbot agent instance for the current session.

    Used as a dependency for the chat API endpoints.

    Parameters
    ----------
    session_id : str
        The unique session ID from the user's cookie.

    Returns
    -------
    ChatbotAssistant
        The chatbot agent instance associated with the session.
    """
    # Check invariant: Session must be started
    if not session_id or session_id not in agent_store:
        raise HTTPException(status_code=401, detail="No valid session")
    return agent_store[session_id]

# ------------------------------------------------------------------
# Backend Endpoints


# API router for backend endpoints
router = APIRouter(
    prefix="/api",
    dependencies=[Depends(get_agent)]
)


@router.get("/models")
async def get_models(
    agent: ChatbotAssistant = Depends(get_agent)
) -> AvailableModelsResponse:
    """
    Get the list of available LLMs for use.

    Parameters
    ----------
    agent : ChatbotAssistant
        The chatbot agent instance for the current session.

    Returns
    -------
    AvailableModelsResponse
        A response object containing the list of available models
        and the default model.
    """
    return AvailableModelsResponse(
        models=list(agent.models.keys()),
        default_model=agent.default_model
    )


@router.get("/personalities")
async def get_personalities(
    agent: ChatbotAssistant = Depends(get_agent)
) -> AvailablePersonalitiesResponse:
    """
    Get the list of supported chatbot personalities.

    Parameters
    ----------
    agent : ChatbotAssistant
        The chatbot agent instance for the current session.

    Returns
    -------
    AvailablePersonalitiesResponse
        List of supported personalities and default personality.
    """
    return AvailablePersonalitiesResponse(
        personalities=list(agent.supported_personalities),
        default_personality=agent.default_personality
    )


@router.post("/reset")
async def reset_memory(
    agent: ChatbotAssistant = Depends(get_agent)
) -> dict:
    """
    Reset the agent's memory.

    Parameters
    ----------
    agent : ChatbotAssistant
        The chatbot agent instance for the current session.
    """
    agent.reset_memory()
    return {"ok": True}


@router.post("/chat")
async def chat(
    chat_request: ChatRequest,
    agent: ChatbotAssistant = Depends(get_agent)
) -> ChatResponse:
    """
    Receive a user message and return the agent's response.

    Parameters
    ----------
    chat_request : ChatRequest
        The request body containing the user's message.
    agent : ChatbotAssistant
        The chatbot agent instance for the current session.

    Returns
    -------
    ChatResponse
        The agent's response message.
    """
    response = agent(chat_request.message)
    return ChatResponse(response=response)


@router.post("/config/model")
async def set_model(
    body: dict,
    agent: ChatbotAssistant = Depends(get_agent)
) -> dict:
    """
    Update the LLM to be used for the agent.

    Parameters
    ----------
    body : dict
        The request body containing the new model name.
    agent : ChatbotAssistant
        The chatbot agent instance for the current session.

    Notes
    -----
    `agent_store` already references the in-memory object,
    so the update on the agent instance suffices.
    """
    model = body.get("model")
    try:
        agent.set_client(model)
        return {"ok": True}
    except KeyError:
        raise HTTPException(
            status_code=400,
            detail=f"Provided model '{model}' not found."
        ) from None


@router.post("/config/personality")
async def set_personality(
    body: dict,
    agent: ChatbotAssistant = Depends(get_agent)
) -> dict:
    """
    Update the agent's personality system prompt.

    Parameters
    ----------
    body : dict
        The request body containing the new personality traits.
    agent : ChatbotAssistant
        The chatbot agent instance for the current session.
    """
    agent.set_personality(body["personality"])
    return {"ok": True}


# Add router to FastAPI app
app.include_router(router)
