from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.chatbot.core.context import BaseContextHelper

# ------------------------------------------------------------------
# Test BaseContextHelper

# Example YAML cases

VALID_YAML_CONTENT = (
    "plain_key: plain_value\n"
    "greeting: 'Hello, {name}!'\n"
)
LIST_YAML_CONTENT = "- first\n- second\n"
EMPTY_YAML_CONTENT = ""
INVALID_YAML_CONTENT = "key: [1, 2\n"  # unclosed flow sequence

# Testing Class


class TestBaseContextHelper:
    """Test `BaseContextHelper`."""

    @pytest.fixture
    def helper(self) -> BaseContextHelper:
        """Build a fresh, cache-enabled BaseContextHelper instance."""
        return BaseContextHelper(file_caching=True)

    @pytest.fixture
    def write_yaml_file(self, tmp_path: Path) -> Callable[..., Path]:
        """
        Build a factory for writing arbitrary content to a YAML file.

        Parameters
        ----------
            tmp_path: Path
                Built-in pytest fixture for creating and managing
                temp directories during tests.

        Returns
        -------
            Callable[..., Path]
                A function taking `content` and an optional `filename`,
                writing `content` to `tmp_path / filename` and returning
                its path.
        """
        def _write(content: str, filename: str = "test.yaml") -> Path:
            file_path = tmp_path / filename
            file_path.write_text(content, encoding="utf-8")
            return file_path
        return _write

    def test_file_caching_enabled_creates_store(self) -> None:
        """Test `file_caching=True` initializes a file store."""
        helper = BaseContextHelper(file_caching=True)
        assert hasattr(helper, "file_store")

    def test_file_caching_disabled_skips_store(self) -> None:
        """Test `file_caching=False` does not initialize a file store."""
        helper = BaseContextHelper(file_caching=False)
        assert not hasattr(helper, "file_store")

    def test_loads_dict_yaml(
            self,
            helper: BaseContextHelper,
            write_yaml_file: Callable[..., Path]
    ) -> None:
        """Test a dict-based YAML file is loaded with its expected content."""
        file_path = write_yaml_file(VALID_YAML_CONTENT)
        content = helper.load_yaml_file(str(file_path))
        assert content == {
            "plain_key": "plain_value",
            "greeting": "Hello, {name}!"
        }

    def test_loads_list_yaml(
            self,
            helper: BaseContextHelper,
            write_yaml_file: Callable[..., Path]
    ) -> None:
        """Test a list-based YAML file is loaded with its expected content."""
        file_path = write_yaml_file(LIST_YAML_CONTENT)
        content = helper.load_yaml_file(str(file_path))
        assert content == ["first", "second"]

    def test_missing_file_raises(self, helper: BaseContextHelper) -> None:
        """Test a nonexistent file path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            helper.load_yaml_file("does/not/exist.yaml")

    def test_invalid_syntax_raises(
            self,
            helper: BaseContextHelper,
            write_yaml_file: Callable[..., Path]
    ) -> None:
        """Test malformed YAML syntax raises yaml.YAMLError."""
        file_path = write_yaml_file(INVALID_YAML_CONTENT)
        with pytest.raises(yaml.YAMLError, match="Invalid YAML syntax"):
            helper.load_yaml_file(str(file_path))

    def test_empty_file_returns_empty_dict_and_warns(
            self,
            helper: BaseContextHelper,
            write_yaml_file: Callable[..., Path]
    ) -> None:
        """Test an empty YAML file returns an empty dict."""
        file_path = write_yaml_file(EMPTY_YAML_CONTENT)
        content = helper.load_yaml_file(str(file_path))
        assert content == {}

    def test_caching_avoids_second_file_read(
            self,
            mocker: Any,
            helper: BaseContextHelper,
            write_yaml_file: Callable[..., Path]
    ) -> None:
        """
        Test repeated calls with caching enabled parse the file only once.

        Parameters
        ----------
            mocker: Any
                A pytest mocker, which is resolved automatically
                when invoking the tests.
                Requires the dependency `pytest-mock`.

        Notes
        -----
        `mocker.spy` wraps the real `yaml.safe_load` instead of replacing it,
        so parsing happens normally while the call count is recorded.
        """
        file_path = write_yaml_file(VALID_YAML_CONTENT)
        spy = mocker.spy(yaml, "safe_load")
        helper.load_yaml_file(str(file_path))
        helper.load_yaml_file(str(file_path))
        helper.load_yaml_file(str(file_path))
        assert spy.call_count == 1

    def test_caching_disabled_reads_file_each_time(
            self,
            mocker: Any,
            write_yaml_file: Callable[..., Path]
    ) -> None:
        """Test files are read each time with disabled caching."""
        file_path = write_yaml_file(VALID_YAML_CONTENT)
        helper = BaseContextHelper(file_caching=False)
        spy = mocker.spy(yaml, "safe_load")
        helper.load_yaml_file(str(file_path))
        helper.load_yaml_file(str(file_path))
        assert spy.call_count == 2

    def test_retrieves_key_value(
            self,
            helper: BaseContextHelper,
            write_yaml_file: Callable[..., Path]
    ) -> None:
        """Test a non-string value is retrieved unchanged without kwargs."""
        file_path = write_yaml_file(VALID_YAML_CONTENT)
        value = helper.load_and_format_context(
            file_path=str(file_path),
            key_name="plain_key"
        )
        assert value == "plain_value"

    def test_formats_string_value_with_kwargs(
            self,
            helper: BaseContextHelper,
            write_yaml_file: Callable[..., Path]
    ) -> None:
        """Test a string value with placeholders is formatted using kwargs."""
        file_path = write_yaml_file(VALID_YAML_CONTENT)
        value = helper.load_and_format_context(
            file_path=str(file_path),
            key_name="greeting",
            name="Ada"
        )
        assert value == "Hello, Ada!"

    def test_non_dict_top_level_raises_value_error(
            self,
            helper: BaseContextHelper,
            write_yaml_file: Callable[..., Path]
    ) -> None:
        """Test a list top-level YAML file raises ValueError."""
        file_path = write_yaml_file(LIST_YAML_CONTENT)
        with pytest.raises(
            ValueError,
            match="Expected YAML file to contain a dictionary"
        ):
            helper.load_and_format_context(
                file_path=str(file_path),
                key_name="any_key"
            )

    def test_missing_key_raises_key_error(
            self,
            helper: BaseContextHelper,
            write_yaml_file: Callable[..., Path]
    ) -> None:
        """Test a key absent from the YAML file raises KeyError."""
        file_path = write_yaml_file(VALID_YAML_CONTENT)
        with pytest.raises(KeyError, match="not found in YAML file"):
            helper.load_and_format_context(
                file_path=str(file_path),
                key_name="missing_key"
            )
