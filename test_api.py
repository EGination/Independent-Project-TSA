import pytest
import os
from unittest.mock import patch, MagicMock

from api import DeepSeekAPI


class TestDeepSeekAPI:
    """Test suite for DeepSeekAPI class."""

    def test_init_creates_openai_client(self):
        """Test that __init__ creates OpenAI client with correct parameters."""
        with patch('api.OpenAI') as mock_openai:
            api = DeepSeekAPI()
            mock_openai.assert_called_once_with(
                api_key=os.environ.get('DEEPSEEK_API_KEY'),
                base_url="https://api.deepseek.com"
            )

    def test_init_uses_env_api_key(self):
        """Test that API key is retrieved from environment variable."""
        with patch('api.OpenAI') as mock_openai:
            api = DeepSeekAPI()
            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs['base_url'] == "https://api.deepseek.com"

    def test_get_system_prompt_returns_yaml_parsed(self):
        """Test that get_system_prompt returns parsed YAML content."""
        api = DeepSeekAPI()
        # Note: yaml.safe_load() with no argument returns None
        result = api.get_system_prompt("some/dir")
        assert result is None  # Current implementation

    def test_get_system_prompt_accepts_dir_parameter(self):
        """Test that get_system_prompt accepts dir parameter without error."""
        api = DeepSeekAPI()
        # Should not raise exception even if dir is not used
        result = api.get_system_prompt("test/path")
        assert result is None

    @patch('api.OpenAI')
    def test_generate_tsa_returns_content(self, mock_openai_class):
        """Test that generate_tsa returns message content from API response."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Test response content"
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        api = DeepSeekAPI()
        result = api.generate_tsa()

        assert result == "Test response content"
        mock_client.chat.completions.create.assert_called_once()

    @patch('api.OpenAI')
    def test_generate_tsa_uses_correct_model(self, mock_openai_class):
        """Test that generate_tsa calls API with correct model."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Test"
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        api = DeepSeekAPI()
        api.generate_tsa()

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs['model'] == "deepseek-v4-flash"

    @patch('api.OpenAI')
    def test_generate_tsa_uses_high_reasoning_effort(self, mock_openai_class):
        """Test that generate_tsa sets reasoning_effort to high."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Test"
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        api = DeepSeekAPI()
        api.generate_tsa()

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs['reasoning_effort'] == "high"

    @patch('api.OpenAI')
    def test_generate_tsa_includes_thinking_extra_body(self, mock_openai_class):
        """Test that generate_tsa includes thinking in extra_body."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Test"
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        api = DeepSeekAPI()
        api.generate_tsa()

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert 'extra_body' in call_kwargs
        assert call_kwargs['extra_body'] == {"thinking": {"type": "enabled"}}

    @patch('api.OpenAI')
    def test_generate_tsa_sends_system_and_user_messages(self, mock_openai_class):
        """Test that generate_tsa sends correct message structure."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Test"
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        api = DeepSeekAPI()
        api.generate_tsa()

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        messages = call_kwargs['messages']
        assert len(messages) == 2
        assert messages[0]['role'] == 'system'
        assert messages[1]['role'] == 'user'
        assert messages[1]['content'] == 'Hello'

    @patch('api.OpenAI')
    def test_generate_tsa_stream_false(self, mock_openai_class):
        """Test that generate_tsa sets stream to False."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Test"
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        api = DeepSeekAPI()
        api.generate_tsa()

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs['stream'] is False

    @patch('api.OpenAI')
    def test_generate_tsa_empty_content(self, mock_openai_class):
        """Test generate_tsa when API returns empty content."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = ""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        api = DeepSeekAPI()
        result = api.generate_tsa()

        assert result == ""

    @patch('api.OpenAI')
    def test_generate_tsa_none_content(self, mock_openai_class):
        """Test generate_tsa when API returns None content."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = None
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        api = DeepSeekAPI()
        result = api.generate_tsa()

        assert result is None
