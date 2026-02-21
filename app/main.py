import streamlit as st
from loguru import logger

from config import AppConfig
from chatbot import ChatbotAgent


def init_session_state() -> None:
    """Initialize Streamlit session state with default values on first run."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "personality" not in st.session_state:
        st.session_state.personality = AppConfig.SUPPORTED_CHATBOT_PERSONALITIES[0]

    if "agent" not in st.session_state:
        agent = ChatbotAgent()
        agent.set_prompt_templates(st.session_state.personality)
        st.session_state.agent = agent


def render_sidebar() -> None:
    """Render the sidebar with personality selector and clear chat button."""
    with st.sidebar:
        st.header("Settings")

        new_personality = st.selectbox(
            "Chatbot Personality",
            options=AppConfig.SUPPORTED_CHATBOT_PERSONALITIES,
            index=AppConfig.SUPPORTED_CHATBOT_PERSONALITIES.index(
                st.session_state.personality
            ),
        )

        if new_personality != st.session_state.personality:
            st.session_state.personality = new_personality
            st.session_state.agent.set_prompt_templates(new_personality)
            logger.info(f"Personality changed to '{new_personality}'")

        st.markdown("---")

        if st.button("Clear Chat", type="primary", use_container_width=True):
            st.session_state.agent.reset_memory()
            st.session_state.messages = []
            logger.info("Chat cleared and agent memory reset.")
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


def main() -> None:
    """Entry point for the Streamlit chatbot application."""
    st.set_page_config(page_title="AI Chatbot Assistant", page_icon="💬")

    init_session_state()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
