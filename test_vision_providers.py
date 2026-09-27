"""Unit tests for provider request context and structured result plumbing."""

from __future__ import annotations

import os
from io import BytesIO
from urllib.error import HTTPError
from unittest.mock import patch

import vision_providers


def test_reference_prompt_is_only_used_when_a_reference_is_present() -> None:
    assert (
        "selected target"
        not in vision_providers._prompt_for_reference(None).lower()
    )
    assert (
        "selected target"
        in vision_providers._prompt_for_reference(b"reference").lower()
    )


def test_presenter_prompt_requires_nested_box_and_confidence() -> None:
    prompt = " ".join(vision_providers.PRESENTER_PROMPT.split())

    assert '"confidence": 0.92' in prompt
    assert '"box": {"x": 100' in prompt
    assert "Always include found, label, confidence, and box" in prompt
    assert "Never put x, y, width, or height at the top level" in prompt
    assert "entire visible head" in prompt
    assert "Do not place the box above the head" in prompt


def test_automatic_prompt_includes_target_priority_order() -> None:
    prompt = vision_providers._prompt_for_reference(
        None,
        "person, face, presenter",
    ).lower()

    assert "target priorities, highest first: person, face, presenter" in prompt
    assert "select the first clearly visible target" in prompt


@patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}, clear=False)
@patch(
    "vision_providers._groq_chat_completion",
    return_value={
        "choices": [
            {
                "message": {
                    "content": '{"found": false, "label": "presenter", '
                    '"confidence": 0.1, "box": {'
                    '"x": 0, "y": 0, "width": 1, "height": 1}}'
                }
            }
        ]
    },
)
def test_groq_request_sends_current_frame_and_reference(
    post_json,
) -> None:
    provider = vision_providers.GroqVisionProvider()

    provider._request(
        b"current-frame",
        {"type": "json_object"},
        b"selected-reference",
    )

    body = post_json.call_args.args[1]
    content = body["messages"][0]["content"]
    image_parts = [part for part in content if part["type"] == "image_url"]
    assert len(image_parts) == 2
    assert any(
        "selected target reference crop" in part.get("text", "").lower()
        for part in content
    )


@patch.dict(
    os.environ,
    {"GROQ_API_KEY": "test-key", "GROQ_MODEL": ""},
    clear=False,
)
@patch(
    "vision_providers._groq_chat_completion",
    return_value={
        "choices": [
            {
                "message": {
                    "content": '{"found": true, "label": "person in blue", '
                    '"confidence": 0.91, "box": {'
                    '"x": 100, "y": 120, "width": 300, "height": 600}}'
                }
            }
        ]
    },
)
def test_groq_vision_uses_documented_qwen_json_mode(post_json) -> None:
    provider = vision_providers.GroqVisionProvider()

    result = provider.locate_presenter(b"current-image")

    body = post_json.call_args.args[1]
    assert provider.model == "qwen/qwen3.8-27b"
    assert body["model"] == "qwen/qwen3.8-27b"
    assert body["response_format"] == {"type": "json_object"}
    assert body["max_completion_tokens"] == 256
    assert "max_tokens" not in body
    assert result.label == "person in blue"


@patch.dict(
    os.environ,
    {"GROQ_API_KEY": "primary-key", "GROQ_API_KEY_2": "backup-key"},
    clear=False,
)
@patch(
    "vision_providers._groq_chat_completion",
    side_effect=[
        vision_providers.ProviderRequestError("primary failed", status_code=503),
        {
            "choices": [
                {
                    "message": {
                        "content": '{"found": false, "label": "presenter", '
                        '"confidence": 0, "box": {'
                        '"x": 0, "y": 0, "width": 0, "height": 0}}'
                    }
                }
            ]
        },
    ],
)
def test_groq_uses_backup_key_after_primary_request_failure(post_json) -> None:
    provider = vision_providers.GroqVisionProvider()

    provider.probe_text()

    assert post_json.call_count == 2
    assert post_json.call_args_list[0].args[0] == "primary-key"
    assert post_json.call_args_list[1].args[0] == "backup-key"


def test_groq_flat_box_coordinates_are_normalized() -> None:
    result = vision_providers._validate_result(
        {
            "found": True,
            "x": 320,
            "y": 290,
            "width": 680,
            "height": 410,
            "label": "person wearing blue",
            "confidence": 0.92,
        },
        "Groq",
    )

    assert result.box == (320.0, 290.0, 680.0, 410.0)
    assert result.label == "person wearing blue"


@patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}, clear=False)
@patch(
    "vision_providers._groq_chat_completion",
    return_value={"choices": [{"message": {"content": "CAMERA_TEXT_OK"}}]},
)
def test_groq_text_probe_uses_plain_text_request(post_json) -> None:
    provider = vision_providers.GroqVisionProvider()

    assert provider.probe_text() == "CAMERA_TEXT_OK"

    body = post_json.call_args.args[1]
    assert body["messages"][0]["content"] == "Reply with exactly: CAMERA_TEXT_OK"
    assert "response_format" not in body


@patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}, clear=False)
@patch(
    "vision_providers._groq_chat_completion",
    return_value={"choices": [{"message": {"content": "A blue box and red circle."}}]},
)
def test_groq_image_probe_sends_encoded_image(post_json) -> None:
    provider = vision_providers.GroqVisionProvider()

    assert "blue box" in provider.describe_image(b"test-jpeg").lower()

    body = post_json.call_args.args[1]
    content = body["messages"][0]["content"]
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith(
        "data:image/jpeg;base64,"
    )


@patch(
    "vision_providers.urlopen",
    side_effect=HTTPError(
        "https://example.invalid/?key=must-not-leak",
        403,
        "Forbidden",
        None,
        BytesIO(
            b'{"error":{"code":"model_permission_blocked_project",'
            b'"message":"Model blocked at project level"}}'
        ),
    ),
)
def test_http_errors_include_safe_provider_permission_details(urlopen) -> None:
    try:
        vision_providers._post_json(
            "https://example.invalid/?key=must-not-leak",
            {},
            {},
        )
    except vision_providers.ProviderRequestError as error:
        message = str(error)
    else:
        raise AssertionError("Expected provider request to fail")

    assert "model_permission_blocked_project" in message
    assert "Model blocked at project level" in message
    assert "must-not-leak" not in message