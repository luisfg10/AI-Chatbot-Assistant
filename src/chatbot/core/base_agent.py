import json

from loguru import logger
from openai import OpenAI

from src.chatbot.tools import tool_registry


class ChatCompletionsBaseAgent:
    """
    Base class for defining LLM capabilities and interactions.

    Compatible with OpenAI's chat completions endpoint.

    Includes all API call-related logic, but does not handle abstractions
    like prompt fetching and memory management.
    """

    def __init__(
            self,
            models: dict[str, dict],
            default_model: str,
            max_recursive_tool_calls: int | None = None,
            max_completion_tokens: int | None = None,
    ) -> None:
        """
        Initialize the instance for the class.

        Parameters
        ----------
            models: dict
                A dictionary of model codes to store in the self
                and use for setting clients. e.g.,
                    {
                        "gpt-5.5": {
                            "base url": "example_url.com",
                            "api key": "example-api-key"
                        },
                        ...
                    }
            default_model: str
                The default model to use. Checks that this model is
                inside the models dict.
            max_recusrive_tool_calls: int | None = None
                Max allowed number of tool calls for the Agent before
                forcing a response. Optional but recommended for preventing
                infinite tool call loops, which are unlikely but possible.
            max_completion_tokens: int | None = None
                Optional parameter for specifying max tokens.
        """
        # Resolve max completion tokens
        self.max_completion_tokens = (
            max_completion_tokens if isinstance(max_completion_tokens, int)
            and max_completion_tokens > 0 else None
        )

        # Init recursive tool calls attribute
        self.recursive_tool_calls = {
            "current": 0,
            "max": (
                max_recursive_tool_calls
                if isinstance(max_recursive_tool_calls, int)
                and max_recursive_tool_calls > 0 else 50
            )
        }

        # Save models: assumes it has the correct structure
        self.models = models

        if default_model not in self.models:
            default_model = list(self.models.keys())[0]
        self.default_model = default_model

        # Set client
        self.set_client()

    def set_client(
            self,
            model_code: str | None = None,
    ) -> None:
        """
        Set and store an OpenAI client and model code to use.

        This model may be called at any time to update the chatbot's
        client parameters.

        Parameters
        ----------
            model_code: str | None
                The code of the LLM model for which to create the client.
                If not provided, defaults to self.default model.

        Returns
        -------
            None
                Updates or creates self.client
        """
        if not model_code:
            model_code = self.default_model
        self.model_code = model_code
        model_data = self.models[model_code]
        self.client = OpenAI(
            base_url=model_data["base url"],
            api_key=model_data["api key"]
        )

    def llm_api_call(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        recursive: bool = False,
        existing_response_messages: list[dict] | None = None,
        tool_limit_reached_prompt: str | None = None
    ) -> list | None:
        r"""
        Call the LLM API and return the generated response.

        Returns a list of messages containing the LLM's response.
        This list will have len 1 for simple answers, but for multi-step
        answers involving tool calls it'll have a larger length.

        For multi-step answers that require tool calls, calls itself
        recursively while updating the messages list and while monitoring
        the number of recursive tool calls being done without answering the
        user. In order to enforce a max limit of recursive tool calls, whenever
        the limit is reached the next LLM call eliminates the possibility of
        tool calling (tools=None) and also adds a context message to the LLM
        explaining the must answer the user in their next message.

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
            recursive: bool
                Boolean flag indicating whether the method was called from itself.
                This is used internally when tool calling to select the correct
                list of messages to build on the API request.
            existing_response_messages: list[dict] | None
                In case it is a recursive call, the current list of messages that
                will make part of the final return value.
            tool_limit_reached_prompt: str | None = None
                Message to send to the LLM in case it exceeds the max recursive
                tool calls allowed for a single interaction.
                e.g.,
                "You have exceeded your tool calls, please respond to the user now."

        Returns
        -------
            list
                The generated list of messages from the LLM's response.

        Examples
        --------
        >>> response = ChatbotAssistant().llm_api_call(
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
        response_messages = existing_response_messages if recursive else []

        # Log API call values
        logger.debug(f"LLM API Call - Messages: {messages}")
        other_params = {
            k: v for k, v in params.items()
            if k != "messages"
        }
        logger.trace(f"LLM API Call - Other Params: {other_params}")

        # Make API call
        response = self.client.chat.completions.create(**params)

        # Parse choice
        choice = response.choices[0]
        logger.debug(f"Response choice: {choice}")

        # Get response depending on finish reason
        if choice.finish_reason == "stop":  # regular text response
            self.recursive_tool_calls["current"] = 0
            response_messages.append({
                "role": "assistant",
                "content": choice.message.content
            })
            return response_messages

        elif choice.finish_reason == "tool_calls":  # tool calling required
            self.recursive_tool_calls["current"] += 1
            choice.message.content = None

            new_messages = [self.serialize_chat_completions_response(choice.message)]
            new_messages.extend(self.llm_tool_call(choice.message.tool_calls))

            response_messages.extend(new_messages)

            # Build the next request's message list without mutating original list
            next_messages = messages + new_messages

            if self.recursive_tool_calls["current"] < self.recursive_tool_calls["max"]:
                return self.llm_api_call(
                    messages=next_messages,
                    tools=tools,
                    recursive=True,
                    existing_response_messages=response_messages,
                    tool_limit_reached_prompt=tool_limit_reached_prompt
                )
            else:  # tool calling exceeded: demand response next and take away tools
                if (
                    not isinstance(tool_limit_reached_prompt, str)
                    or len(tool_limit_reached_prompt) == 0
                ):
                    raise ValueError(
                        "`tool_limit_reached_prompt` is not provided, "
                        "cannot proceed. Please double-check implementation."
                    )
                next_messages = next_messages + [{
                    "role": "system",
                    "content": tool_limit_reached_prompt
                }]
                return self.llm_api_call(
                    messages=next_messages,
                    tools=None,
                    recursive=True,
                    existing_response_messages=response_messages,
                    tool_limit_reached_prompt=tool_limit_reached_prompt
                )

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

        Examples
        --------
        >>> tool_results = ChatbotAssistant().llm_tool_call(
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
            if function_to_run is None:  # Check invariant: known tools only
                raise ValueError(f"Unknown tool: {function_to_run}")

            result = function_to_run(**args)
            results.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result)
            })

        return results

    @staticmethod
    def serialize_chat_completions_response(message: dict) -> dict:
        """
        Serialize an OpenAI's chat completions message into a Python dict.

        Parameters
        ----------
            message: ChatCompletionMessage
                An object from OpenAI's chat completions endpoint
                indicating to call a tool.

        Returns
        -------
            dict
                A serialized dict to be appended to the messages list.

        Examples
        --------
        >>> example = ChatbotContextHelper().serialize_tool_calls_response(
                message=ChatCompletionMessage(
                    content=None,
                    refusal=None,
                    role='assistant',
                    annotations=None,
                    audio=None,
                    function_call=None,
                    tool_calls=[
                        ChatCompletionMessageFunctionToolCall(
                            id='function-call-6907',
                            function=Function(
                                arguments='{}',
                                name='get_current_date'
                            ),
                            type='function'
                        )
                    ]
                )
            )
        >>> print(example)
        {
            'role': 'assistant',
            'content': None,
            'tool_calls': [
                {
                    'id': 'function-call-6907',
                    'type': 'function',
                    'function': {
                        'name': 'get_current_date',
                        'arguments': '{}'
                    }
                }
            ]
        }
        """
        msg = {
            "role": message.role,
            "content": message.content
        }
        if message.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in message.tool_calls
                ]
            return msg
