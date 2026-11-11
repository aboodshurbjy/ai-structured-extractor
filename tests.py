"""
Tests for the structured extractor pipeline.

Covers:
  - Prompt template rendering
  - JSON parsing & validation (the parse_response layer)
  - Error handling for malformed LLM output
  - End-to-end extraction with a mock provider

Run:  pytest tests.py -v
"""

import json
import pytest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
from extractor import (
    _text_from_anthropic_response,
    _text_from_openai_response,
    TruncatedResponseError,
    parse_response,
    ParseError,
    extract,
    MODE_PROMPTS,
    REQUIRED_KEYS,
    SYSTEM_PREAMBLE,
)


# -- parse_response -----------------------------------------------------------

class TestParseResponse:
    """Validate the JSON parsing and schema-checking layer."""

    def test_valid_summarize(self):
        raw = json.dumps({
            "title": "Q3 Update",
            "summary": "Revenue grew 12% QoQ.",
            "key_points": ["Revenue up", "Hiring paused"],
            "word_count_original": 340,
            "sentiment": "positive",
        })
        result = parse_response(raw, "summarize")
        assert result["title"] == "Q3 Update"
        assert isinstance(result["key_points"], list)

    def test_valid_entities(self):
        raw = json.dumps({
            "entities": [
                {"name": "Acme Corp", "type": "ORG", "context": "Acme Corp announced"},
            ],
            "entity_count": 1,
        })
        result = parse_response(raw, "entities")
        assert result["entity_count"] == 1

    def test_valid_actions(self):
        raw = json.dumps({
            "actions": [
                {
                    "task": "Review proposal",
                    "owner": "Jane",
                    "deadline": "Friday",
                    "priority": "high",
                },
            ],
            "action_count": 1,
        })
        result = parse_response(raw, "actions")
        assert result["actions"][0]["owner"] == "Jane"

    def test_valid_classify(self):
        raw = json.dumps({
            "classifications": [
                {"category": "technical", "confidence": 0.9},
                {"category": "business", "confidence": 0.4},
            ],
            "primary_category": "technical",
            "language": "en",
        })
        result = parse_response(raw, "classify")
        assert result["primary_category"] == "technical"

    def test_strips_markdown_fences(self):
        """LLMs sometimes wrap JSON in ```json ... ``` -- parser should handle it."""
        inner = json.dumps({
            "title": "Test",
            "summary": "A test.",
            "key_points": [],
            "word_count_original": 5,
            "sentiment": "neutral",
        })
        raw = f"```json\n{inner}\n```"
        result = parse_response(raw, "summarize")
        assert result["title"] == "Test"

    def test_missing_keys_raises(self):
        raw = json.dumps({"title": "Incomplete"})
        with pytest.raises(ParseError, match="missing required keys"):
            parse_response(raw, "summarize")

    def test_invalid_json_raises(self):
        with pytest.raises(ParseError, match="invalid JSON"):
            parse_response("This is not JSON at all.", "summarize")

    def test_empty_string_raises(self):
        with pytest.raises(ParseError):
            parse_response("", "summarize")


# -- Prompt templates ---------------------------------------------------------

class TestPromptTemplates:
    """Ensure every mode has a prompt and the text placeholder works."""

    @pytest.mark.parametrize("mode", MODE_PROMPTS.keys())
    def test_template_has_text_placeholder(self, mode):
        assert "{text}" in MODE_PROMPTS[mode]

    @pytest.mark.parametrize("mode", MODE_PROMPTS.keys())
    def test_template_renders(self, mode):
        rendered = MODE_PROMPTS[mode].format(text="Hello world")
        assert "Hello world" in rendered

    @pytest.mark.parametrize("mode", REQUIRED_KEYS.keys())
    def test_required_keys_defined(self, mode):
        assert len(REQUIRED_KEYS[mode]) > 0


# -- End-to-end with mocked API ----------------------------------------------

class TestExtractE2E:
    """Integration tests using a mocked LLM provider."""

    MOCK_SUMMARIZE_RESPONSE = json.dumps({
        "title": "Project Kickoff",
        "summary": "The team aligned on scope and timeline for Phase 1.",
        "key_points": [
            "Phase 1 targets June delivery",
            "Budget approved at $50K",
            "Weekly syncs start Monday",
        ],
        "word_count_original": 187,
        "sentiment": "positive",
    })

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key-123"})
    @patch("extractor._call_anthropic")
    def test_summarize_anthropic(self, mock_call):
        mock_call.return_value = self.MOCK_SUMMARIZE_RESPONSE
        result = extract("Some meeting notes here.", mode="summarize", provider="anthropic")

        assert result.mode == "summarize"
        assert result.provider == "anthropic"
        assert result.data["title"] == "Project Kickoff"
        assert len(result.data["key_points"]) == 3

        # Verify the API was called with the right structure
        mock_call.assert_called_once()
        call_args = mock_call.call_args
        assert call_args[0][1] == SYSTEM_PREAMBLE  # system prompt

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key-456"})
    @patch("extractor._call_openai")
    def test_summarize_openai(self, mock_call):
        mock_call.return_value = self.MOCK_SUMMARIZE_RESPONSE
        result = extract("Some meeting notes here.", mode="summarize", provider="openai")

        assert result.provider == "openai"
        assert result.data["sentiment"] == "positive"

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key-123"})
    @patch("extractor._call_anthropic")
    def test_actions_mode(self, mock_call):
        mock_call.return_value = json.dumps({
            "actions": [
                {
                    "task": "Submit RFI response",
                    "owner": "Mike",
                    "deadline": "2025-04-15",
                    "priority": "high",
                },
                {
                    "task": "Update drawings",
                    "owner": "unassigned",
                    "deadline": "none",
                    "priority": "medium",
                },
            ],
            "action_count": 2,
        })
        result = extract("Project notes...", mode="actions", provider="anthropic")
        assert result.data["action_count"] == 2
        assert result.data["actions"][1]["owner"] == "unassigned"

    def test_missing_api_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
                extract("text", mode="summarize", provider="anthropic")

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            extract("text", mode="nonexistent")

    def test_invalid_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            extract("text", provider="gemini")

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key-123"})
    @patch("extractor._call_anthropic")
    def test_result_to_json(self, mock_call):
        mock_call.return_value = self.MOCK_SUMMARIZE_RESPONSE
        result = extract("Notes.", mode="summarize")
        json_str = result.to_json()
        reparsed = json.loads(json_str)
        assert reparsed["title"] == "Project Kickoff"


# -- Anthropic response handling ----------------------------------------------

class TestAnthropicResponseText:
    """The text must be read from text blocks, wherever they are in the list."""

    @staticmethod
    def _response(*blocks, stop_reason="end_turn"):
        return SimpleNamespace(content=list(blocks), stop_reason=stop_reason)

    def test_text_only_response(self):
        response = self._response(SimpleNamespace(type="text", text='{"a": 1}'))
        assert _text_from_anthropic_response(response) == '{"a": 1}'

    def test_thinking_block_before_text_is_skipped(self):
        """Regression: content[0] can be a thinking block with no .text attribute."""
        response = self._response(
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text='{"a": 1}'),
        )
        assert _text_from_anthropic_response(response) == '{"a": 1}'

    def test_multiple_text_blocks_are_joined(self):
        response = self._response(
            SimpleNamespace(type="text", text='{"a": '),
            SimpleNamespace(type="text", text="1}"),
        )
        assert _text_from_anthropic_response(response) == '{"a": 1}'

    def test_no_text_block_raises_parse_error(self):
        response = self._response(
            SimpleNamespace(type="thinking", thinking=""), stop_reason="refusal"
        )
        with pytest.raises(ParseError, match="refusal"):
            _text_from_anthropic_response(response)


# -- Truncated responses ------------------------------------------------------

class TestTruncatedResponses:
    """A response cut off at max_tokens must be reported as truncated, not as
    generic invalid JSON."""

    def test_anthropic_max_tokens_stop_raises_truncated(self):
        response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text='{"title": "Cut off he')],
            stop_reason="max_tokens",
        )
        with pytest.raises(TruncatedResponseError, match="max_tokens"):
            _text_from_anthropic_response(response)

    def test_openai_length_finish_raises_truncated(self):
        response = SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="length",
            message=SimpleNamespace(content='{"title": "Cut off he'),
        )])
        with pytest.raises(TruncatedResponseError, match="max_tokens"):
            _text_from_openai_response(response)

    def test_openai_normal_finish_returns_content(self):
        response = SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop", message=SimpleNamespace(content='{"a": 1}'),
        )])
        assert _text_from_openai_response(response) == '{"a": 1}'

    def test_truncated_error_is_a_parse_error(self):
        """The CLI catches ParseError, so truncation must be a subclass."""
        assert issubclass(TruncatedResponseError, ParseError)
