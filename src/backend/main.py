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
from src.chatbot import ChatbotAgent

app = FastAPI()
agent_store: dict[str, ChatbotAgent] = {}

# ------------------------------------------------------------------
# Homepage

# Mount static files
app.mount("/static", StaticFiles(directory="src/frontend/static"), name="static")


@app.get("/")
async def homepage(session_id: str = Cookie(default=None)) -> FileResponse:
    """Serve the static SPA frontend and create/save unique session IDs using cookies."""
    response = FileResponse("src/frontend/static/index.html")
    if not session_id:
        session_id = str(uuid4())
        response.set_cookie(key="session_id", value=session_id, path="/", httponly=True)
        agent_store[session_id] = ChatbotAgent()
    elif session_id not in agent_store:
        agent_store[session_id] = ChatbotAgent()
    return response

# ------------------------------------------------------------------
# Chat and Config Endpoints


def get_agent(session_id: str = Cookie(default=None)) -> ChatbotAgent:
    """
    Retrieve the chatbot agent instance for the current session.

    This is defined as a dependency for the chat API endpoints, not to be
    called directly.

    Parameters
    ----------
        session_id : str
            The unique session ID from the user's cookie.

    Returns
    -------
        ChatbotAgent
            The chatbot agent instance associated with the session.
    """
    # Check invariant: Session must be started
    if not session_id or session_id not in agent_store:
        raise HTTPException(status_code=401, detail="No valid session")
    return agent_store[session_id]


# Define API router for chat and config endpoints
router = APIRouter(
    prefix="/api",
    dependencies=[Depends(get_agent)]
)


@router.get("/models")
async def get_models(
    agent: ChatbotAgent = Depends(get_agent)
) -> AvailableModelsResponse:
    """
    Get the list of available LLMs supported by the chatbot agent.

    Parameters
    ----------
        agent : ChatbotAgent
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
    agent: ChatbotAgent = Depends(get_agent)
) -> AvailablePersonalitiesResponse:
    """
    Get the list of supported chatbot personalities.

    Parameters
    ----------
        agent : ChatbotAgent
            The chatbot agent instance for the current session.

    Returns
    -------
        AvailablePersonalitiesResponse
            A response object containing the list of supported personalities
            and the default personality.
    """
    return AvailablePersonalitiesResponse(
        personalities=list(agent.supported_chatbot_personalities),
        default_personality=agent.default_personality
    )


@router.post("/reset")
async def reset_memory(
    agent: ChatbotAgent = Depends(get_agent)
) -> dict:
    """
    Reset the chatbot agent's memory.

    Parameters
    ----------
        agent : ChatbotAgent
            The chatbot agent instance for the current session.

    Returns
    -------
        dict
            A simple response indicating success.
    """
    agent.reset_memory()
    return {"ok": True}


@router.post("/chat")
async def chat(
    chat_request: ChatRequest,
    agent: ChatbotAgent = Depends(get_agent)
) -> ChatResponse:
    """
    Receive a user message and return the chatbot's response.

    Parameters
    ----------
        chat_request : ChatRequest
            The request body containing the user's message.
        agent : ChatbotAgent
            The chatbot agent instance for the current session.

    Returns
    -------
        ChatResponse
            The chatbot's response message.
    """
    response = agent.chatbot_call(chat_request.message)
    return ChatResponse(response=response)


@router.post("/config/model")
async def set_model(
    body: dict,
    agent: ChatbotAgent = Depends(get_agent)
) -> dict:
    """
    Update the LLM to be called by the chatbot agent.

    Notice the agent_store object already references the in-memory object,
    so it is not necessary to update it after changing the model.

    Parameters
    ----------
        body : dict
            The request body containing the new model name.
        agent : ChatbotAgent
            The chatbot agent instance for the current session.

    Returns
    -------
        dict
    """
    agent.set_client(body["model"])
    return {"ok": True}


@router.post("/config/personality")
async def set_personality(
    body: dict,
    agent: ChatbotAgent = Depends(get_agent)
) -> dict:
    """
    Update the personality traits of the chatbot agent.

    Notice the agent_store object already references the in-memory object,
    so it is not necessary to update it after changing the personality.

    Parameters
    ----------
        body : dict
            The request body containing the new personality traits.
        agent : ChatbotAgent
            The chatbot agent instance for the current session.

    Returns
    -------
        dict
    """
    agent.set_personality(body["personality"])
    return {"ok": True}


app.include_router(router)
