import streamlit as st
from loguru import logger

from chatbot import ChatbotAgent


def init_session_state() -> None:
    """Initialize Streamlit session state with default values on first run."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "agent" not in st.session_state:
        agent = ChatbotAgent()
        agent.set_client(agent.default_model)
        agent.set_personality(agent.default_personality)
        st.session_state.agent = agent

    if "model" not in st.session_state:
        st.session_state.model = st.session_state.agent.default_model

    if "personality" not in st.session_state:
        st.session_state.personality = st.session_state.agent.default_personality


def render_sidebar() -> None:
    """Render the sidebar with model selector, personality selector, and clear chat button."""
    with st.sidebar:
        st.header("Settings")

        agent = st.session_state.agent
        available_models = list(agent.models.keys())
        supported_personalities = list(agent.supported_chatbot_personalities)

        new_model = st.selectbox(
            "Model",
            options=available_models,
            index=available_models.index(st.session_state.model),
        )

        if new_model != st.session_state.model:
            st.session_state.model = new_model
            agent.set_client(new_model)
            logger.debug(f"Model changed to '{new_model}'")

        new_personality = st.selectbox(
            "Chatbot Personality",
            options=supported_personalities,
            index=supported_personalities.index(
                st.session_state.personality
            ),
        )

        if new_personality != st.session_state.personality:
            st.session_state.personality = new_personality
            agent.set_personality(new_personality)
            logger.debug(f"Personality changed to '{new_personality}'")

        st.markdown("---")

        if st.button("Clear Chat", type="primary", use_container_width=True):
            st.session_state.agent.reset_memory()
            st.session_state.messages = []
            logger.debug("Chat cleared and agent memory reset.")
            st.rerun()


def render_chat() -> None:
    """Render the chat message history and handle new user input."""
    # Welcome message when chat is empty
    if not st.session_state.messages:
        st.markdown(
            "<div style='text-align:center; color:grey; padding-top:2rem;'>"
            "Welcome! Type something to get started."
            "</div>",
            unsafe_allow_html=True,
        )

    # Display existing messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if user_input := st.chat_input("Say something..."):
        # Show user message immediately
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate assistant response with loading indicator
        with st.chat_message("assistant"):
            with st.spinner("Loading..."):
                response = st.session_state.agent.chatbot_call(user_input)

            st.markdown(response)

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )


def build_chatbot_ui() -> None:
    """Entry point for the Streamlit chatbot application."""
    st.set_page_config(page_title="AI Chatbot Assistant", page_icon="💬")

    init_session_state()
    render_sidebar()
    render_chat()
