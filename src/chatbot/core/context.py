from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from config import AppConfig


class BaseContextHelper:
    """Access text from YAML files."""

    def __init__(
            self,
            file_caching: bool = True
    ) -> None:
        """
        Initialize the BaseContextHelper class instance.

        Parameters
        ----------
        file_caching: bool = True
            Whether to save loaded files to memory to eliminate
            fetching on future calls.
        """
        self.file_caching = file_caching
        if self.file_caching:
            self.file_store = {}

    def load_yaml_file(
            self,
            file_path: str
    ) -> dict | list:
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
        # Evaluate file caching
        if self.file_caching and file_path in self.file_store:
            return self.file_store[file_path]

        # Check filepath exists
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"YAML file not found: {file_path}")

        try:
            logger.debug(f"Loading YAML file: {file_path}")
            with file_path_obj.open('r', encoding='utf-8') as file:
                content = yaml.safe_load(file)

            if content is None:
                logger.warning(
                    f"YAML file is empty or contains only null values: {file_path}"
                )

            if self.file_caching:
                self.file_store[file_path] = content

            return content or {}

        except yaml.YAMLError as e:
            raise yaml.YAMLError(
                f"Invalid YAML syntax in file {file_path}: {e}"
            ) from None
        except Exception as e:
            raise ValueError(
                f"Unexpected error loading YAML file {file_path}: {e}"
            ) from e

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
            The key in the YAML file whose value should be
            retrieved and formatted.
        **kwargs:
            Additional keyword arguments to use for formatting,
            in case the retrieved value corresponds to a string
            with placeholders.

        Returns
        -------
        dict | list | str
            The retrieved value from the YAML file, formatted if
            applicable.
        """
        file = self.load_yaml_file(file_path)
        if not isinstance(file, dict):
            raise ValueError(
                f"Expected YAML file to contain a dictionary at the top level, "
                f"but got {type(file).__name__} in file {file_path}"
            )

        if key_name not in file:
            raise KeyError(
                f"Key '{key_name}' not found in YAML file {file_path}"
            )

        value = file[key_name]
        if kwargs and isinstance(value, str):
            value = value.format(**kwargs)

        return value


class ChatbotContextHelper(BaseContextHelper):
    """Manage the agent's context from YAML files."""

    def __init__(
            self,
            context_dir: str = AppConfig.CHATBOT_CONTEXT_DIR,
            system_prompts_filename: str = "system.yaml",
            user_prompts_filename: str = "user.yaml",
            file_caching: bool = True
    ) -> None:
        """
        Initialize the class instance.

        Parameters
        ----------
        context_dir: str
            The directory where the agent's context files are.
        system_prompts_filename: str
            The filename for the system prompts file.
        user_prompts_filename: str
            The filename for the user prompts file.
        file_caching: bool = True
            Whether to loaded context files to memory to avoid
            repeat loading operations.

        Notes
        -----
        Assumes a two-file distribution of chatbot context: one for system
        prompts and one for user prompts.
        """
        # Check directory is a valid path string
        if not isinstance(context_dir, str):
            raise ValueError("'context_dir' must be a string.")
        if not context_dir.endswith("/"):
            context_dir += "/"
        self.context_dir = context_dir

        # Validate context files have correct extension
        if not system_prompts_filename.endswith(".yaml"):
            system_prompts_filename += ".yaml"
        self.system_prompts_filename = system_prompts_filename
        if not user_prompts_filename.endswith(".yaml"):
            user_prompts_filename += ".yaml"
        self.user_prompts_filename = user_prompts_filename

        # Init parent class
        super().__init__(file_caching=file_caching)

    def get_agent_instructions(
            self, personality: str,
    ) -> str:
        """Get the agent's instructions based on its personality."""
        outer_key = self.load_and_format_context(
            file_path=self.context_dir + self.system_prompts_filename,
            key_name="personalities"
        )
        return outer_key[personality]

    def get_compacting_instructions(
            self,
            personality: str
    ) -> str:
        """
        Get the instructions for compacting the conversation.

        The agent's personality is used to decide which details of the
        conversation are important to retain.
        """
        # Fetch base key
        outer_key = self.load_and_format_context(
            file_path=self.context_dir + self.system_prompts_filename,
            key_name="memory compacting"
        )
        system_prompt_template = outer_key["template"]
        personality_prompt = outer_key["personalities"][personality]

        return system_prompt_template.format(**{
            "chatbot personality prompt": personality_prompt
        })

    @staticmethod
    def transcribe_messages_list(messages: list) -> str:
        """
        Convert a list of chatbot-user messages into a readable transcript.

        Used as part of memory compacting in order to convert the messages list
        into a digestible format by the LLM.
        """
        transcript = ""
        for message in messages:
            role = message["role"]
            if role == "tool":
                transcript += f"tool result: {message['content']}\n"

            elif role == "assistant" and message.get("tool_calls"):
                for tool_call in message["tool_calls"]:
                    transcript += (
                        "assistant called tool: "
                        f"{tool_call['function']['name']}\n"
                    )
            else:
                transcript += f"{role}: {message['content']}\n"
        return transcript

    def get_compacting_user_prompt(
            self,
            recent_conversation: list[dict],
            long_term_memory: str | None = None
    ) -> str:
        """
        Get the user prompt for compacting.

        The list of recent conversation messages is transcribed to a
        single string before formatting into the prompt.

        Parameters
        ----------
        recent_conversation: list[dict]
            A list of recent conversation messages.
            e.g.,
                [
                    {
                        "role": "user",
                        "content": "What is the capital of France?"
                    },
                    {
                        "role": "assistant",
                        "content": "The capital of France is Paris."
                    },
                    ...
                ]
        long_term_memory: str | None
            Optional long-term memory to include in the user prompt,
            which may contain key facts about the user and the conversation
            that should be retained.
            e.g., "The user's name is John and they like cars."

        Returns
        -------
        str
            The formatted user prompt for the memory manager.
        """
        outer_key = self.load_and_format_context(
            file_path=self.context_dir + self.user_prompts_filename,
            key_name="memory compacting"
        )
        memory_template = outer_key["template"]

        # Transcribe messages list
        messages_transcription = self.transcribe_messages_list(
            recent_conversation
        )

        return memory_template.format(**{
            "short term memory": messages_transcription,
            "long term memory": long_term_memory or "Empty."
        })

    def get_conversation_summary_prompt(
            self,
            summary: str
    ) -> str:
        """Get the prompt template summarizing the conversation."""
        summary_template = self.load_and_format_context(
            file_path=self.context_dir + self.system_prompts_filename,
            key_name="conversation summary"
        )
        return summary_template.format(**{
            "summary": summary
        })

    def get_rec_tool_lim_prompt(self) -> str:
        """Get the system message for exhausted recursive tool calls."""
        return self.load_and_format_context(
            file_path=self.context_dir + self.system_prompts_filename,
            key_name="recursive tool limit reached"
        )
