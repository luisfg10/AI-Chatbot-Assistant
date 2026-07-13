from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from config import AppConfig


class BaseContextHelper:
    """General utility class for managing text from .yaml files."""

    def __init__(
            self,
            file_caching: bool = True
    ) -> None:
        """
        Initialize the BaseContextHelper class instance.

        Parameters
        ----------
            file_caching: bool = True
                Whether to save to memory already-loaded files to avoid
                having to load them again on new calls.
                This is useful in projects with few files that need to be
                accessed many times, but might consume greater memory when
                several files (100+) need to be accessed in a single run.
                Defaults to True.

        Returns
        -------
            None
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

        If self.file_caching is set to True, it will first check to see if
        the file is loaded to the file store and loaded from there. If it is
        not already load it, it will do so and avoid doing that next time around.

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
        # Optional: load from cache
        if self.file_caching and file_path in self.file_store:
            return self.file_store[file_path]

        # Invariant: Filepath must exist
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
    """Helper class specialized in managing chatbot context."""

    def __init__(
            self,
            context_dir: str = AppConfig.CHATBOT_CONTEXT_DIR,
            system_prompts_filename: str = "system.yaml",
            user_prompts_filename: str = "user.yaml",
            file_caching: bool = True
    ) -> None:
        """
        Initialize the ChatbotContextHelper.

        Assumes a two-file distribution of chatbot context: one for system
        prompts and one for user prompts.

        Parameters
        ----------
            context_dir: str
                The directory where the context YAML files are located.
            system_prompts_filename: str
                The filename for the system prompts YAML file.
            user_prompts_filename: str
                The filename for the user prompts YAML file.
            file_caching: bool = True
                Whether to save to memory already-loaded files to avoid
                having to load them again on new calls.
        """
        # Invariant check
        if not isinstance(context_dir, str):
            raise ValueError("'context_dir' must be a string.")

        if not context_dir.endswith("/"):
            context_dir += "/"
        self.context_dir = context_dir

        if not system_prompts_filename.endswith(".yaml"):
            system_prompts_filename += ".yaml"
        self.system_prompts_filename = system_prompts_filename

        if not user_prompts_filename.endswith(".yaml"):
            user_prompts_filename += ".yaml"
        self.user_prompts_filename = user_prompts_filename

        super().__init__(file_caching=file_caching)

    def get_chatbot_instructions(
            self,
            personality: str,
    ) -> str:
        """Get the chatbot's instructions based on its personality."""
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
        Get and format the instructions for compacting the messages list.

        The chatbot's personality is used to decide which details of the
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

        This method is used as part of memory compacting for the ChatbotAssistant,
        and its output is to be passed on as a user message in the messages list.
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
        Get the user prompt for the memory manager.

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
        """Get and format the prompt template summarizing the conversation."""
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
