"""Cloud vision adapters for automatic presenter selection.
The desktop application intentionally talks to the providers through this
small module instead of importing provider SDKs.  That keeps the Windows
prototype lightweight and makes the API keys available only through local
environment variables:
    GEMINI_API_KEY
    GROQ_API_KEY
Both adapters return the same normalized result:
    {
        "found": True,
        "label": "presenter",
        "confidence": 0.94,
        "box": {"x": 100, "y": 120, "width": 300, "height": 700},
    }
Coordinates use a 0..1000 coordinate space, independent of the source
camera resolution.

Provider calls may also receive a session-only JPEG crop of the target selected
by the operator. That crop is sent only for the current request and is never
persisted by this module.
"""
from __future__ import annotations
import base64
import json
import os
import re
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_GROQ_MODEL = "qwen/qwen3.8-27b"
REQUEST_TIMEOUT_SECONDS = 20
PRESENTER_PROMPT = """
Inspect this camera frame and locate the single best presenter or main person
who should be followed by a presenter camera. Ignore UI overlays, the green
virtual-camera rectangle, and background people unless there is no clearer
presenter. Aim the camera target at the person's head when a head is visible.
The head box must cover the entire visible head: hair or scalp, forehead,
face, ears, cheeks, and chin. Do not place the box above the head or draw only
a face/forehead strip; include a small margin around the complete head. If the
head is not visible, target the upper body; if that is not visible, target the
largest clearly visible portion of the person's body. Do not use a full-body
box when a clear head or upper body is available. Return exactly this JSON
shape, keeping coordinates inside the box:
{"found": true, "label": "person in blue shirt", "confidence": 0.92,
 "box": {"x": 100, "y": 120, "width": 300, "height": 700}}
Always include found, label, confidence, and box. When no presenter is found,
set found=false, confidence=0, and box coordinates and dimensions to 0. Never
put x, y, width, or height at the top level.
Use normalized coordinates from 0 to 1000:
- x and y are the top-left corner of the person's bounding box.
- width and height are the size of the bounding box.
- confidence is a number from 0 to 1.
- Give label a short visual description, without identifying anyone by name.
- Return found=false when no presenter is clearly visible.
- Do not include markdown, explanations, or additional keys.
""".strip()

DIAGNOSTIC_TEXT_PROMPT = "Reply with exactly: CAMERA_TEXT_OK"
DIAGNOSTIC_IMAGE_PROMPT = (
    "Describe the colored geometric shapes in this calibration image in one "
    "short sentence."
)

TARGET_REFERENCE_PROMPT = """
The first image is the current camera frame. The second image is a reference
crop selected by the operator. Locate that selected target in the current
camera frame, even if other people are visible. Match the target's appearance
and position, but do not perform face recognition or identify anyone by name.
Ignore UI overlays and the green virtual-camera rectangle. Return found=false
when the selected target is not clearly visible. Return exactly one JSON object
matching the requested schema.
Use normalized coordinates from 0 to 1000:
- x and y are the top-left corner of the selected target's bounding box.
- width and height are the size of the bounding box.
- Do not include markdown, explanations, or additional keys.
""".strip()
GEMINI_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "found": {"type": "BOOLEAN"},
        "label": {"type": "STRING"},
        "confidence": {"type": "NUMBER"},
        "box": {
            "type": "OBJECT",
            "properties": {
                "x": {"type": "NUMBER"},
                "y": {"type": "NUMBER"},
                "width": {"type": "NUMBER"},
                "height": {"type": "NUMBER"},
            },
            "required": ["x", "y", "width", "height"],
        },
    },
    "required": ["found", "label", "confidence", "box"],
}
GROQ_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "found": {"type": "boolean"},
        "label": {"type": "string"},
        "confidence": {"type": "number"},
        "box": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "width": {"type": "number"},
                "height": {"type": "number"},
            },
            "required": ["x", "y", "width", "height"],
        },
    },
    "required": ["found", "label", "confidence", "box"],
}
class VisionProviderError(RuntimeError):
    """Base error shown to the desktop app without exposing credentials."""
class ProviderConfigurationError(VisionProviderError):
    """Raised when a provider's local API key is missing."""
class ProviderRequestError(VisionProviderError):
    """Raised when a provider request fails."""
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
@dataclass(frozen=True)
class VisionResult:
    """A validated presenter location in normalized 0..1000 coordinates."""
    found: bool
    label: str
    confidence: float
    box: tuple[float, float, float, float] | None
    provider: str
def _decode_json_text(text: str) -> Any:
    """Decode normal JSON and the fenced JSON commonly returned by models."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise VisionProviderError("The vision provider returned invalid JSON.")
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as error:
            raise VisionProviderError(
                "The vision provider returned invalid JSON."
            ) from error


def _structured_response_value(value: Any) -> Any:
    """Convert SDK model objects into the dictionaries used by the validator."""
    if isinstance(value, dict):
        return value

    for method_name in ("model_dump", "dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            dumped = method()
            if isinstance(dumped, dict):
                return dumped

    return value


def _gemini_response_payload(response: Any) -> Any:
    """Prefer the SDK's parsed structured result, with a text fallback."""
    parsed = _structured_response_value(getattr(response, "parsed", None))
    if isinstance(parsed, dict):
        return parsed

    response_text = getattr(response, "text", None)
    if not isinstance(response_text, str) or not response_text.strip():
        raise VisionProviderError("Gemini returned no structured presenter result.")
    return _decode_json_text(response_text)


def provider_status_code(error: Exception) -> int | None:
    """Read provider status codes across SDK and HTTP client error shapes."""
    candidates = (
        getattr(error, "status_code", None),
        getattr(error, "code", None),
        getattr(getattr(error, "response", None), "status_code", None),
    )
    for candidate in candidates:
        try:
            status_code = int(candidate)
        except (TypeError, ValueError):
            continue
        if status_code > 0:
            return status_code
    return None
def _validate_result(payload: Any, provider_name: str) -> VisionResult:
    """Validate and normalize a provider response before it reaches OpenCV."""
    if not isinstance(payload, dict):
        raise VisionProviderError("The vision response was not a JSON object.")
    found = payload.get("found")
    if not isinstance(found, bool):
        raise VisionProviderError("The vision response has no valid found flag.")
    label = str(payload.get("label") or "presenter").strip()[:80] or "presenter"
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError) as error:
        raise VisionProviderError("The vision response has invalid confidence.") from error
    if not 0.0 <= confidence <= 1.0:
        raise VisionProviderError("The vision confidence must be between 0 and 1.")
    if not found:
        return VisionResult(False, label, confidence, None, provider_name)
    box = payload.get("box")
    if box is None and all(
        coordinate in payload for coordinate in ("x", "y", "width", "height")
    ):
        box = payload
    if not isinstance(box, dict):
        raise VisionProviderError("The vision response has no valid bounding box.")
    try:
        x = float(box["x"])
        y = float(box["y"])
        width = float(box["width"])
        height = float(box["height"])
    except (KeyError, TypeError, ValueError) as error:
        raise VisionProviderError("The vision response has an invalid bounding box.") from error
    if (
        not all(value == value for value in (x, y, width, height))
        or x < 0
        or y < 0
        or width <= 0
        or height <= 0
        or x + width > 1000
        or y + height > 1000
    ):
        raise VisionProviderError("The vision bounding box is outside the frame.")
    return VisionResult(
        True,
        label,
        confidence,
        (x, y, width, height),
        provider_name,
    )
def _post_json(
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
    timeout: int = REQUEST_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """POST JSON and return a decoded object without putting secrets in errors."""
    request = Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8")
    except HTTPError as error:
        detail = ""
        error_body = b""
        try:
            error_body = error.read(8192)
            error_payload = json.loads(error_body.decode("utf-8"))
            provider_error = error_payload.get("error", {})
            if isinstance(provider_error, dict):
                parts = [
                    str(provider_error[key]).strip()
                    for key in ("code", "message")
                    if isinstance(provider_error.get(key), (str, int))
                    and str(provider_error[key]).strip()
                ]
                if parts:
                    detail = " Details: " + re.sub(
                        r"\s+",
                        " ",
                        ": ".join(parts),
                    )[:240]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            content_type = (
                error.headers.get("Content-Type", "")
                if error.headers
                else ""
            )
            if content_type.lower().startswith("text/plain") and error_body:
                safe_text = re.sub(
                    r"\s+",
                    " ",
                    error_body.decode("utf-8", errors="replace"),
                ).strip()
                if safe_text:
                    detail = f" Details: {safe_text[:240]}"

        # Never include the request URL: Gemini carries its API key in the URL.
        raise ProviderRequestError(
            f"Vision provider rejected the request (HTTP {error.code}).{detail}",
            status_code=error.code,
        ) from error
    except (URLError, TimeoutError, OSError) as error:
        raise ProviderRequestError(
            "Vision provider could not be reached. Check the internet connection."
        ) from error
    try:
        decoded = json.loads(response_body)
    except json.JSONDecodeError as error:
        raise ProviderRequestError("Vision provider returned a non-JSON response.") from error
    if not isinstance(decoded, dict):
        raise ProviderRequestError("Vision provider returned an unexpected response.")
    return decoded


def _safe_provider_error_detail(error: Exception) -> str:
    """Extract bounded error fields without exposing request credentials."""
    body = getattr(error, "body", None)
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except json.JSONDecodeError:
            pass
    if not isinstance(body, dict):
        response = getattr(error, "response", None)
        try:
            body = response.json() if response is not None else None
        except (AttributeError, ValueError):
            body = None

    if isinstance(body, dict):
        provider_error = body.get("error", body)
        if isinstance(provider_error, dict):
            parts = [
                str(provider_error[key]).strip()
                for key in ("code", "message")
                if isinstance(provider_error.get(key), (str, int))
                and str(provider_error[key]).strip()
            ]
            if parts:
                detail = ": ".join(parts)
                return " Details: " + re.sub(r"\s+", " ", detail)[:240]

    response = getattr(error, "response", None)
    try:
        content_type = response.headers.get("content-type", "")
        response_text = response.text
    except (AttributeError, ValueError):
        return ""
    if content_type.lower().startswith("text/plain") and response_text:
        return " Details: " + re.sub(r"\s+", " ", response_text).strip()[:240]
    return ""


def _groq_chat_completion(
    api_key: str,
    request_body: dict[str, Any],
) -> dict[str, Any]:
    """Send a Groq chat request through the official SDK transport."""
    try:
        from groq import Groq
    except ImportError as error:
        raise ProviderConfigurationError(
            "Groq SDK is not installed. Run: python -m pip install groq"
        ) from error

    try:
        completion = Groq(api_key=api_key).chat.completions.create(
            **request_body
        )
    except Exception as error:
        status_code = provider_status_code(error)
        if status_code is not None:
            detail = _safe_provider_error_detail(error)
            raise ProviderRequestError(
                f"Groq rejected the request (HTTP {status_code}).{detail}",
                status_code=status_code,
            ) from error
        raise ProviderRequestError(
            "Groq request failed. Check connectivity and model access."
        ) from error

    if isinstance(completion, dict):
        return completion
    model_dump = getattr(completion, "model_dump", None)
    if callable(model_dump):
        return model_dump()
    raise VisionProviderError("Groq returned an unexpected completion response.")


def _jpeg_as_base64(jpeg_bytes: bytes) -> str:
    return base64.b64encode(jpeg_bytes).decode("ascii")


def _prompt_for_reference(
    reference_jpeg_bytes: bytes | None,
    target_priority: str | None = None,
) -> str:
    """Choose a prompt that matches the images sent with the request."""
    prompt = (
        TARGET_REFERENCE_PROMPT
        if reference_jpeg_bytes is not None
        else PRESENTER_PROMPT
    )
    if reference_jpeg_bytes is None and target_priority:
        priorities = [
            item.strip()
            for item in target_priority.split(",")
            if item.strip()
        ]
        if priorities:
            prompt += (
                "\nTarget priorities, highest first: "
                + ", ".join(priorities[:12])
                + ". Evaluate these in order and select the first clearly "
                "visible target; only continue to the next item when the "
                "higher-priority item is absent."
            )
    return prompt


class GeminiVisionProvider:
    """Gemini SDK adapter using multimodal structured JSON output."""

    name = "Gemini"

    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.model = os.getenv("GEMINI_MODEL", "").strip() or DEFAULT_GEMINI_MODEL

    def _generate_probe_text(self, contents: Any) -> str:
        if not self.api_key:
            raise ProviderConfigurationError(
                "GEMINI_API_KEY is not configured in Windows environment variables."
            )

        try:
            from google import genai
            from google.genai import types
        except ImportError as error:
            raise ProviderConfigurationError(
                "Google GenAI SDK is not installed. Run: python -m pip install google-genai"
            ) from error

        try:
            response = genai.Client(api_key=self.api_key).models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=80,
                ),
            )
        except Exception as error:
            status_code = provider_status_code(error)
            if status_code is not None:
                raise ProviderRequestError(
                    f"Gemini rejected the request (HTTP {status_code}).",
                    status_code=status_code,
                ) from error
            raise ProviderRequestError(
                "Gemini request failed. Check connectivity and model access."
            ) from error

        response_text = getattr(response, "text", None)
        if not isinstance(response_text, str) or not response_text.strip():
            raise VisionProviderError("Gemini returned no text for the probe.")
        return response_text.strip()

    def probe_text(self) -> str:
        """Check that the configured Gemini model can answer a text prompt."""
        return self._generate_probe_text(DIAGNOSTIC_TEXT_PROMPT)

    def describe_image(self, jpeg_bytes: bytes) -> str:
        """Return a short caption for a diagnostic image."""
        from google.genai import types

        return self._generate_probe_text(
            [
                types.Part.from_text(text=DIAGNOSTIC_IMAGE_PROMPT),
                types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"),
            ]
        )

    def locate_presenter(
        self,
        jpeg_bytes: bytes,
        reference_jpeg_bytes: bytes | None = None,
        target_priority: str | None = None,
    ) -> VisionResult:
        if not self.api_key:
            raise ProviderConfigurationError(
                "GEMINI_API_KEY is not configured in Windows environment variables."
            )

        try:
            from google import genai
            from google.genai import types
        except ImportError as error:
            raise ProviderConfigurationError(
                "Google GenAI SDK is not installed. Run: python -m pip install google-genai"
            ) from error

        client = genai.Client(api_key=self.api_key)
        prompt = _prompt_for_reference(reference_jpeg_bytes, target_priority)
        try:
            response = client.models.generate_content(
                model=self.model,
                contents=[
                    types.Part.from_text(text=prompt),
                    types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"),
                ]
                + (
                    [
                        types.Part.from_text(
                            text="Selected target reference crop:"
                        ),
                        types.Part.from_bytes(
                            data=reference_jpeg_bytes,
                            mime_type="image/jpeg",
                        ),
                    ]
                    if reference_jpeg_bytes is not None
                    else []
                ),
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=256,
                    response_mime_type="application/json",
                    response_schema=GEMINI_RESPONSE_SCHEMA,
                ),
            )
        except Exception as error:
            status_code = provider_status_code(error)
            if status_code is not None:
                raise ProviderRequestError(
                    f"Gemini rejected the request (HTTP {status_code}).",
                    status_code=status_code,
                ) from error
            raise ProviderRequestError(
                "Gemini request failed. Check connectivity and model access."
            ) from error

        return _validate_result(_gemini_response_payload(response), self.name)


class GroqVisionProvider:
    """Groq OpenAI-compatible REST adapter for a vision-capable model."""
    name = "Groq"
    def __init__(self) -> None:
        configured_keys = (
            os.getenv("GROQ_API_KEY", "").strip(),
            os.getenv("GROQ_API_KEY_2", "").strip(),
        )
        self.api_keys = tuple(dict.fromkeys(key for key in configured_keys if key))
        self.api_key = self.api_keys[0] if self.api_keys else ""
        self.model = os.getenv("GROQ_MODEL", "").strip() or DEFAULT_GROQ_MODEL

    def _request_with_fallback(self, body: dict[str, Any]) -> dict[str, Any]:
        if not self.api_keys:
            raise ProviderConfigurationError(
                "GROQ_API_KEY or GROQ_API_KEY_2 is not configured in Windows environment variables."
            )

        last_error: Exception | None = None
        for api_key in self.api_keys:
            try:
                return _groq_chat_completion(api_key, body)
            except Exception as error:
                last_error = error

        if last_error is not None:
            raise last_error
        raise ProviderRequestError("Groq request failed.")

    def _probe(self, content: str | list[dict[str, Any]]) -> str:
        if not self.api_key:
            raise ProviderConfigurationError(
                "GROQ_API_KEY or GROQ_API_KEY_2 is not configured in Windows environment variables."
            )
        response = self._request_with_fallback(
            {
                "model": self.model,
                "temperature": 0,
                "max_completion_tokens": 80,
                "messages": [{"role": "user", "content": content}],
            },
        )
        try:
            message = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise VisionProviderError(
                "Groq returned no text for the probe."
            ) from error
        if isinstance(message, list):
            message = "".join(
                part.get("text", "")
                for part in message
                if isinstance(part, dict)
            )
        if not isinstance(message, str) or not message.strip():
            raise VisionProviderError("Groq returned no text for the probe.")
        return message.strip()

    def probe_text(self) -> str:
        """Check that the configured Groq model can answer a text prompt."""
        return self._probe(DIAGNOSTIC_TEXT_PROMPT)

    def describe_image(self, jpeg_bytes: bytes) -> str:
        """Return a short caption for a diagnostic image."""
        encoded_image = _jpeg_as_base64(jpeg_bytes)
        return self._probe(
            [
                {"type": "text", "text": DIAGNOSTIC_IMAGE_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}"
                    },
                },
            ]
        )

    def _request(
        self,
        jpeg_bytes: bytes,
        response_format: dict[str, Any],
        reference_jpeg_bytes: bytes | None = None,
        target_priority: str | None = None,
    ) -> dict[str, Any]:
        encoded_frame = _jpeg_as_base64(jpeg_bytes)
        content = [
            {
                "type": "text",
                "text": _prompt_for_reference(
                    reference_jpeg_bytes,
                    target_priority,
                ),
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_frame}"
                },
            },
        ]
        if reference_jpeg_bytes is not None:
            content.extend(
                [
                    {"type": "text", "text": "Selected target reference crop:"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                "data:image/jpeg;base64,"
                                f"{_jpeg_as_base64(reference_jpeg_bytes)}"
                            )
                        },
                    },
                ]
            )
        body = {
            "model": self.model,
            "temperature": 0,
            "max_completion_tokens": 256,
            "response_format": response_format,
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
        }
        return self._request_with_fallback(body)
    def locate_presenter(
        self,
        jpeg_bytes: bytes,
        reference_jpeg_bytes: bytes | None = None,
        target_priority: str | None = None,
    ) -> VisionResult:
        response = self._request(
            jpeg_bytes,
            {"type": "json_object"},
            reference_jpeg_bytes,
            target_priority,
        )
        try:
            text = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise VisionProviderError(
                "Groq returned no structured presenter result."
            ) from error
        if isinstance(text, list):
            text = "".join(
                part.get("text", "")
                for part in text
                if isinstance(part, dict)
            )
        return _validate_result(_decode_json_text(text), self.name)


def create_provider(selection: str):
    """Create the provider selected in the Windows UI."""
    if selection == "Gemini":
        return GeminiVisionProvider()
    if selection == "Groq":
        return GroqVisionProvider()
    raise ValueError(f"Unknown vision provider: {selection}")


def provider_model(selection: str) -> str:
    """Return the model configured for a provider without creating a request."""
    if selection == "Gemini":
        return os.getenv("GEMINI_MODEL", "").strip() or DEFAULT_GEMINI_MODEL
    if selection == "Groq":
        return os.getenv("GROQ_MODEL", "").strip() or DEFAULT_GROQ_MODEL
    raise ValueError(f"Unknown vision provider: {selection}")


def _box_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x, y, width, height = box
    return x + width / 2, y + height / 2


def locate_with_selection(
    selection: str,
    jpeg_bytes: bytes,
    reference_jpeg_bytes: bytes | None = None,
) -> VisionResult:
    """Run one configured provider, or safely compare both providers."""
    if selection != "Compare both":
        return create_provider(selection).locate_presenter(
            jpeg_bytes,
            reference_jpeg_bytes,
        )

    gemini = GeminiVisionProvider().locate_presenter(
        jpeg_bytes,
        reference_jpeg_bytes,
    )
    groq = GroqVisionProvider().locate_presenter(
        jpeg_bytes,
        reference_jpeg_bytes,
    )
    if gemini.found and groq.found and gemini.box and groq.box:
        gx, gy = _box_center(gemini.box)
        qx, qy = _box_center(groq.box)
        if ((gx - qx) ** 2 + (gy - qy) ** 2) ** 0.5 > 250:
            raise VisionProviderError(
                "Gemini and Groq disagreed about the presenter location."
            )
    return gemini if gemini.confidence >= groq.confidence else groq
