from typing import Any
from pathlib import Path

import yaml
from loguru import logger

from config import AppConfig


class BaseContextHelper:
    """General utility class for managing text from .yaml files."""

    def __init__(self) -> None:
        pass

    @staticmethod
    def load_yaml_file(file_path: str) -> dict | list:
        """
        Load a YAML file and return its contents as a JSON-type object.

        Parameters
        ----------
        file_path : str
            The path to the YAML file to load.

        Returns
        -------
        dict | list
            The contents of the YAML file as a dictionary or list.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        yaml.YAMLError
            If the file contains invalid YAML syntax.
        """
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"YAML file not found: {file_path}")

        try:
            logger.debug(f"Loading YAML file: {file_path}")
            with file_path_obj.open('r', encoding='utf-8') as file:
                content = yaml.safe_load(file)

            if content is None:
                logger.warning(f"YAML file is empty or contains only null values: {file_path}")

            return content or {}

        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML syntax in file {file_path}: {e}") from None
        except Exception as e:
            raise ValueError(f"Unexpected error loading YAML file {file_path}: {e}") from e

    def load_and_format_context(
            self,
            file_path: str,
            key_name: str,
            **kwargs: Any
    ) -> dict | list | str:
        """
        Load, retrieve and format a given key in a YAML file.

        Parameters
        ----------
            file_path: str
                The path to the YAML file to load.
            key_name: str
                The key in the YAML file whose value should be retrieved and formatted.
            **kwargs:
                Additional keyword arguments to use for formatting, in case the retrieved
                value corresponds to a string with placeholders.

        Returns
        -------
            dict | list | str
                The retrieved value from the YAML file, formatted if applicable.
        """
        file = self.load_yaml_file(file_path)
        if not isinstance(file, dict):
            raise ValueError(
                f"Expected YAML file to contain a dictionary at the top level, "
                f"but got {type(file).__name__} in file {file_path}"
            )

        if key_name not in file:
            raise KeyError(f"Key '{key_name}' not found in YAML file {file_path}")

        value = file[key_name]
        if kwargs and isinstance(value, str):
            value = value.format(**kwargs)

        return value


class ChatbotContextHelper(BaseContextHelper):
    """Helper class specialized for managing chatbot context."""

    def __init__(
            self,
            context_dir: str  = AppConfig.CHATBOT_CONTEXT_DIR,
            system_prompts_file: str = "system.yaml",
            user_prompts_file: str = "user.yaml"
    ) -> None:
        """
        Initialize the ChatbotContextHelper.

        Assumes a two-file distribution of chatbot context: one for system prompts and one for
        user prompts. If this structure changes, this class will need to be updated accordingly.

        Parameters
        ----------
            context_dir: str
                The directory where the chatbot context YAML files are located.
            system_prompts_file: str
                The filename for the system prompts YAML file.
            user_prompts_file: str
                The filename for the user prompts YAML file.
        """
        if not isinstance(context_dir, str):
            raise ValueError("'context_dir' must be a string.")
        if not context_dir.endswith("/"):
            context_dir += "/"
        if not system_prompts_file.endswith(".yaml"):
            system_prompts_file += ".yaml"
        if not user_prompts_file.endswith(".yaml"):
            user_prompts_file += ".yaml"

        self.context_dir = context_dir
        self.system_prompts_file = system_prompts_file
        self.user_prompts_file = user_prompts_file

        super().__init__()

    def get_chatbot_system_prompt(
            self,
            personality: str,
    ) -> str:
        """Get the system prompt for the chatbot based on the specified personality."""
        outer_key = self.load_and_format_context(
                file_path=self.context_dir + self.system_prompts_file,
                key_name="chatbot"
            )
        return outer_key[personality]

    def get_chatbot_user_prompt(
            self,
            user_query: str,
            conversation_history: str | None = None
    ) -> str:
        """
        Get the user prompt for the chatbot, optionally including conversation history.

        Parameters
        ----------
            user_query: str
                The current user's query to include in the prompt.
                e.g., "What is the capital of France?"
            conversation_history: str | None
                Optional conversation history to include in the user prompt.

        Returns
        -------
            str
                The formatted user prompt.
        """
        outer_key = self.load_and_format_context(
            file_path=self.context_dir + self.user_prompts_file,
            key_name="chatbot"
        )
        user_prompt_template = outer_key.get("template")

        conversation_prompt = ""
        if conversation_history:
            memory_template = outer_key.get("memory template")
            conversation_prompt = memory_template.format(history=conversation_history)

        return user_prompt_template.format(**{
            "user query": user_query,
            "conversation history": conversation_prompt
        })

    def get_memory_manager_system_prompt(
            self,
            personality: str
    ) -> str:
        """
        Get the system prompt for the memory manager based on the specified personality.

        Note the personality is used to refine which details of the conversation are important
        for the memory manager to retain.
        """
        # Fetch base key
        outer_key = self.load_and_format_context(
            file_path=self.context_dir + self.system_prompts_file,
            key_name="memory manager"
        )
        system_prompt_template = outer_key.get("template")
        personality_prompt = outer_key.get("personalities", {}).get(personality, "")

        return system_prompt_template.format(**{
            "chatbot personality prompt": personality_prompt
        })

    def get_memory_manager_user_prompt(
            self,
            recent_conversation: list[dict],
            long_term_memory: str | None = None,
            user_input_key: str = "user",
            chatbot_response_key: str = "chatbot"
    ) -> str:
        """
        Get the user prompt for the memory manager, optionally including long-term memory.

        Parameters
        ----------
            recent_conversation: list[dict]
                A list of recent conversation turns, where each turn is represented as a dictionary.
                e.g.,
                    [{
                        "user": "What is the capital of France?",
                        "chatbot": "The capital of France is Paris."
                    }, ...
                    ]
            long_term_memory: str | None
                Optional long-term memory to include in the user prompt, which may contain key facts about the
                user and the conversation that should be retained.
                e.g., "The user's name is John and they seem to be interested in geography."
            user_input_key: str
                The key in the conversation turn dictionaries that corresponds to the user's input.
                Defaults to "user".
            chatbot_response_key: str
                The key in the conversation turn dictionaries that corresponds to the chatbot's response.
                Defaults to "chatbot".

        Returns
        -------
            str
                The formatted user prompt for the memory manager.
        """
        # TODO: Complete method
        outer_key = self.load_and_format_context(
            file_path=self.context_dir + self.user_prompts_file,
            key_name="memory manager"
        )
        # Format recent conversation into string representation
        short_term_memory = ""

        # Return results
        return self.load_and_format_context(
            file_path=self.context_dir + self.user_prompts_file,
            key_name="memory manager",
            **{
                "long term memory": long_term_memory or "Not available.",
                "short term memory": short_term_memory
            }
        )
