import json

from loguru import logger
from openai import OpenAI

from config import AppConfig
from src.chatbot.core.context import ChatbotContextHelper
from src.chatbot.tools import tool_registry, tool_schema


class ChatbotAgent(ChatbotContextHelper):
    """
    Class for chatbot agent that interacts with a user via a chat interface.

    Inherits from ChatbotContextHelper for managing the chatbot's context.
    """

    def __init__(
            self,
            available_models: dict = AppConfig.AVAILABLE_MODELS,
            provider_api_keys: dict = AppConfig.PROVIDER_API_KEYS,
            default_chatbot_config: dict = AppConfig.DEFAULT_CONFIG,
            supported_chatbot_personalities: list | tuple = AppConfig.SUPPORTED_CHATBOT_PERSONALITIES
    ) -> None:
        """
        Initialize the ChatbotAgent.

        Parameters
        ----------
            available_models: dict
                A dictionary of available LLM providers and their corresponding
                models, as defined in the app config.
            provider_api_keys: dict
                A dictionary mapping LLM providers to their corresponding API keys,
                as defined in the app config.
            default_chatbot_config: dict
                A dictionary containing the default configuration for the chatbot,
                including default LLM provider, model code, and other settings.
            supported_chatbot_personalities: list
                A list of supported chatbot personalities.
                Each personality string should have matching system prompt
                templates in the directory path AppConfig.CHATBOT_CONTEXT_DIR

        Attributes
        ----------
            models: dict
                A dictionary for conveniently storing connection parameters for the
                different available LLM providers so different Chatbot clients can
                be created as needed.
                e.g.,
                    {
                        "gemini-flash-2.5": {
                            "base url": "https://generativelanguage.googleapis.com/v1beta/openai/",
                            "api key": {GEMINI_API_KEY from .env file},
                        }, ...
                    }
            supported_chatbot_personalities: list
                A list of supported chatbot personalities, as defined in the app config.
        """
        # Raise if no models are available based on the app config
        if not isinstance(available_models, dict) or len(available_models) == 0:
            raise ValueError(
                f"Value for `available_models` is invalid: {available_models}"
            )

        # Create an object for easier lookup of available models
        self.models = {}
        for provider, details in available_models.items():
            base_url = details["urls"]["api"]
            api_key = provider_api_keys[provider]
            self.models.update({
                model: {
                    "base url": base_url,
                    "api key": api_key
                } for model in details.get("available models", [])
            })

        # Create an object for storing personality-specific API call parameters
        self.supported_chatbot_personalities = supported_chatbot_personalities
        if (
            not isinstance(self.supported_chatbot_personalities, (list, tuple))
            or not self.supported_chatbot_personalities
        ):
            raise ValueError(
                "Didn't receive a valid value for "
                "`supported_chatbot_personalities`."
            )

        # Set default model to show on Chatbot startup
        default_model = default_chatbot_config.get("model")
        if default_model not in self.models:
            default_model = list(self.models.keys())[0]
        self.default_model = default_model

        # Set default personality to show on Chatbot startup
        default_personality = default_chatbot_config.get("personality")
        if default_personality not in self.supported_chatbot_personalities:
            default_personality = self.supported_chatbot_personalities[0]
        self.default_personality = default_personality

        # Set max completion tokens: if not valid, no limit is set
        max_completion_tokens = default_chatbot_config.get("max completion tokens")
        self.max_completion_tokens = (
                max_completion_tokens if isinstance(max_completion_tokens, int)
                and max_completion_tokens > 0 else None
        )

        # Set total messages in messages list before memory compacting
        self.compacting_msg_limit = int(
            default_chatbot_config.get("compacting message limit", 30)
        )

        # Init messages list
        self.messages: list[dict] = []

        # Initialize context helper
        super().__init__()

        # Initialize the client and personality
        self.set_client()
        self.set_personality()

    def set_client(
            self,
            model_code: str | None = None,
    ) -> None:
        """
        Set and store an OpenAI client and model code to use.

        This model may be called at any time to update the chatbot's client parameters.

        Parameters
        ----------
            model_code: str | None
                The code of the LLM model for which to create the client.
                If not provided, defaults to the default model saved to the self.

        Returns
        -------
            None
                Sets the self.client attribute to an instance of the OpenAI client.
        """
        if not model_code:
            model_code = self.default_model
        self.model_code = model_code
        model_data = self.models[model_code]
        self.client = OpenAI(
            base_url=model_data["base url"],
            api_key=model_data["api key"]
        )

    def set_personality(
            self,
            personality: str | None = None
    ) -> None:
        """
        Set the chatbot's instructions depending on its selected personality.

        This method may be called during initialization or at any other time
        to update the chatbot's prompts based on a new input.
        Modifies the first message in the messages list to the self.

        Parameters
        ----------
            personality: str | None
                The personality for which to set the chatbot's system prompts.
                If not provided, defaults to the default personality saved to
                the self.

        Returns
        -------
            None
                Updates the first message in the messages list containing the
                chatbot's isntructions.
                In the case of the instructions for compacting, saves the instructions
                to use to the self.
        """
        if personality not in self.supported_chatbot_personalities:
            personality = self.default_personality

        # Chatbot instructions
        chatbot_instructions = self.get_chatbot_instructions(personality)
        instructions_message = {
            "role": "system",
            "content": chatbot_instructions
        }

        if len(self.messages) > 0:
            self.messages[0] = instructions_message
        else:
            self.messages.append(instructions_message)

        # Compacting
        self.compacting_instructions = self.get_compacting_instructions(personality)

    def llm_api_call(
        self,
        messages: list[dict],
        tools: list[dict] | None = None
    ) -> list | None:
        r"""
        Call the LLM API and return the generated response.

        Returns a list of messages containing the LLM's response.
        This list will have len 1 for simple answers, but for answers
        involving tool calls it will have a larger length.

        Notes
        -----
            * Includes special considerations for certain OpenAI models for
            which parameter names have changed.

            * Includes several debugging statements. Control whether these are
            printed to stderr using the env var LOG_LEVEL.

        Parameters
        ----------
            messages: list[dict]
                A list of messages containing the current conversation history.
            tools: Optional[list[dict]]
                An optional list of tools to provide to the LLM for enhanced
                capabilities.

        Returns
        -------
            list
                The generated list of messages from the LLM's response.

        >>> response = ChatbotAgent().llm_api_call(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant."
                    },
                    {
                        "role": "user",
                        "content": "What's today's date?"
                    }
                ],
                tools=[
                    {
                        'type': 'function',
                        'function': {
                            'name': 'get_current_date',
                            'description': 'Get today's date in YYYY-MM-DD.',
                            'parameters': {
                                'type': 'object',
                                'properties': {},
                                'required': []
                            }
                        }
                    }
                ]
        >>> print(response)
            [
                {
                    'role': 'assistant',
                    'content': None,
                    'tool_calls': [
                        {
                            'id': 'function-call-67023',
                            'type': 'function',
                            'function': {
                                'name': 'get_current_date',
                                'arguments': '{}'
                            }
                        }
                    ]
                },
                {
                    'role': 'tool',
                    'tool_call_id': 'function-call-67023',
                    'content': '2026-06-29'
                },
                {
                    'role': 'assistant',
                    'content': "Hello there! Today's date is June 29, 2026. \n"
                }
            ]

        """
        # ------------------------------------------------------------------
        # Build API call params

        params = {
            "model": self.model_code,
            "messages": messages
        }
        if (
            hasattr(self, "max_completion_tokens")
            and self.max_completion_tokens
        ):
            if self.model_code in [
                "gpt-5-mini",
                "gpt-5-nano"
            ]:
                param_name = "max_completion_tokens"
            else:
                param_name = "max_tokens"
            params[param_name] = self.max_completion_tokens
        if isinstance(tools, list) and len(tools) > 0:
            params["tools"] = tools

        # ------------------------------------------------------------------
        # Make call, parse and return response

        # Init return value
        response_messages = []

        # Make API call
        logger.debug(f"Making LLM API Call - Params: {params}")
        response = self.client.chat.completions.create(**params)

        # Parse choice
        choice = response.choices[0]
        logger.debug(f"Response choice: {choice}")

        # Get response depending on finish reason
        if choice.finish_reason == "stop":  # regular text response
            response_messages.append({
                "role": "assistant",
                "content": choice.message.content
            })
        elif choice.finish_reason == "tool_calls":  # tool calling required
            # Append tool-calling request to messages
            choice.message.content = None
            response_messages.append(
                self.serialize_chat_completions_response(
                    choice.message
                )
            )
            # Call tools and add results to messages
            response_messages.extend(
                self.llm_tool_call(choice.message.tool_calls)
            )
            # Call LLM again with tool results (no looping allowed for now)
            messages.extend(response_messages)
            params["messages"] = messages
            logger.debug(
                f"Making LLM API Call after tool calling - Params: {params}"
            )
            new_response = self.client.chat.completions.create(**params)
            new_choice = new_response.choices[0]
            logger.debug(
                f"Response choice after tool call: {new_choice}"
            )
            response_messages.append({
                "role": "assistant",
                "content": new_choice.message.content
            })

        return response_messages

    @staticmethod
    def llm_tool_call(
            tool_calls: list,
            tool_registry: dict[str, callable] = tool_registry
    ) -> list:
        """
        Execute a tool call and return its results.

        Compatible with OpenAI's chat completions endpoint.

        Parameters
        ----------
            tool_calls: list
                List with the tool calls to be made from the OpenAI
                chat completions endpoint.
            tool_registry: dict
                A dictionary mapping tool names to the actual Python
                functions, aka tools.

        Returns
        -------
            list
                A list with the results of each made tool call.

        >>> tool_results = ChatbotAgent().llm_tool_call(
            [
                ChatCompletionMessageFunctionToolCall(
                    id='function-call-6050399',
                    function=Function(
                        arguments='{}',
                        name='get_current_date'
                    ),
                    type='function'),
                ...
            ]
            )
        >>> print(tool_results)
            [
                {
                    'role': 'tool',
                    'tool_call_id': 'function-call-6050399',
                    'content': '2026-06-27'
                },
                ...
            ]
        """
        results = []
        for tool_call in tool_calls:
            # Unpack the elements to call
            tool_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            function_to_run = tool_registry.get(tool_name)
            if function_to_run is None:
                raise ValueError(f"Unknown tool: {function_to_run}")

            result = function_to_run(**args)
            results.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result)
            })

        return results

    def chatbot_call(
            self,
            user_query: str,
            tools: dict = tool_schema
    ) -> str | None:
        """
        Make a chatbot API call with included context management.

        Parameters
        ----------
            user_query: str
                The user's input query to the chatbot.
            tools: dict
                Dictionary containing the tools the LLM can call.

        Returns
        -------
            str | None
                The generated response from the chatbot as a string,
                or None if the API call fails.
        """
        # Append user query to messages
        self.messages.append({
            "role": "user",
            "content": user_query
        })

        new_messages = self.llm_api_call(
            messages=self.messages,
            tools=tools
        )
        llm_response = None

        # Update messages and save LLM's response
        llm_response = new_messages[-1]["content"]
        self.messages.extend(new_messages)

        # Evaluate compacting
        self._compact_messages()

        return llm_response

    def _compact_messages(self) -> None:
        """
        Run compacting to summarize the key details in the current conversation.

        Makes a transcript of the current messages list as a single string,
        then sends as a user message along with the compacting instruction.

        Returns
        -------
            None
                Updates the chatbot's messages list attribute.
                Keeps two messages in the messages list:
                    1. The chatbot instructions
                    2. The conversation summary so far
        """
        # Skip if memory update is not needed
        if len(self.messages) % self.compacting_msg_limit != 0:
            return

        # Check if long-term memory exists
        long_term_memory = (
            self.long_term_memory
            if hasattr(self, "long_term_memory")
            and isinstance(self.long_term_memory, str)
            and len(self.long_term_memory) > 0
            else None
        )

        # Get user prompt and format
        conversation_history = self.messages[1:]  # exclude chatbot instructions
        compacting_user_prompt = self.get_compacting_user_prompt(
            recent_conversation=conversation_history,
            long_term_memory=long_term_memory
        )

        # Make API call
        messages = [
            {"role": "system", "content": self.compacting_instructions},
            {"role": "user", "content": compacting_user_prompt}
        ]
        conversation_summary = self.llm_api_call(
            messages=messages
        )

        # Update messages list
        summary_message = self.get_conversation_summary_prompt(
            conversation_summary
        )
        self.messages[1] = {
            "role": "system",
            "content": summary_message
        }
        self.messages = self.messages[: 2]  # remove short-term messages from list
        self.long_term_memory = conversation_summary

    def reset_memory(self) -> None:
        """
        Reset the chatbot's memory, clearing both long-term and short-term memory.

        This can be useful for starting a new conversation or clearing any
        accumulated context.

        Returns
        -------
            None
                Resets the chatbot's memory to an empty state, outside of its
                instructions based on personality.
        """
        self.messages = self.messages[:1]
