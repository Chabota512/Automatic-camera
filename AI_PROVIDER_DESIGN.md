# Cloud AI provider design

## Decision

Use cloud vision only for **target selection and tracker correction**, not for every video frame.

The normal cadence is:

```text
Gemini: one still frame every 5 seconds
Groq:   one still frame every 5 seconds, offset by 2.5 seconds
```

When local tracking loses a target, enabled providers are checked every 2
seconds for up to 10 seconds, then return to the normal cadence.

The providers run in independent worker threads and stay online between requests. The local OpenCV tracker handles the frames between cloud requests. This keeps camera movement responsive and preserves a manual fallback.

## Can Gemini and Groq support this?

Yes, both APIs can receive image input:

- Gemini supports inline base64 image data and documents object detection and segmentation capabilities.
- Groq supports vision-capable models that accept image input, including base64 data passed through an image URL content item.
- Both providers can be asked for machine-readable JSON. Gemini uses response schemas/structured output; Groq provides JSON Schema structured outputs for supported models.

The exact model IDs and free quotas can change, so model names and provider limits belong in configuration rather than being hard-coded into the camera controller.

Official references:

- Gemini vision: <https://ai.google.dev/gemini-api/docs/vision>
- Gemini structured output: <https://ai.google.dev/gemini-api/docs/structured-output>
- Groq vision: <https://console.groq.com/docs/vision>
- Groq structured outputs: <https://console.groq.com/docs/structured-outputs>

## Shared provider contract

The camera application should call both providers through the same interface:

```python
class VisionProvider:
    def locate_presenter(self, jpeg_bytes: bytes) -> dict:
        """
        Return:
        {
            "found": True,
            "label": "presenter",
            "confidence": 0.0,
            "box": {
                "x": 0,       # normalized 0..1000
                "y": 0,
                "width": 0,
                "height": 0
            }
        }
        """
```

Normalized coordinates avoid coupling the provider response to a particular camera resolution.

## Request flow

1. Capture the current wide-camera frame locally.
2. Resize it to approximately 640x360 and JPEG-compress it.
3. Send one image request to each enabled provider every five seconds, with a 2.5-second offset when both are enabled.
4. Validate each JSON response and confidence.
5. Convert each normalized box to source-frame coordinates.
6. Draw both live boxes, retaining each provider's last valid result while it is waiting for its next request.
7. Calculate temporal/disagreement error and score the best valid result.
8. Initialize or correct the local OpenCV tracker with the winning result.
9. Continue tracking locally until the next provider response.

The prompt should instruct the model to:

- Find the presenter, not the virtual-camera crop
- Return one best target
- Return `found: false` when no presenter is clear
- Return only the specified JSON structure
- Use normalized coordinates from 0 to 1000

## Provider strategy

### Gemini

Use Gemini as the first implementation because its documentation explicitly covers multimodal image input and object detection/segmentation. Use the official `google-genai` Python SDK with a current low-latency vision model configured through `GEMINI_MODEL`. The SDK receives the compressed JPEG bytes and returns the structured result that the shared validator checks before it reaches the tracker.

### Groq

Use Groq as a second adapter with a current vision-capable model configured through `GROQ_MODEL`. Groq's value here is low-latency inference, but the selected model must support both image input and the requested structured-output mode.

### Selection modes

```text
Manual control
    Always available.

Assisted tracking
    User click initializes local tracker.

Automatic AI
    Gemini or Groq initializes/corrects local tracker every 5 seconds.
```

The provider should be selectable at runtime:

```text
Gemini
Groq
Compare both
```

For Compare both, send the same compressed frame to both providers in parallel, compare valid responses, and prefer the result with the highest confidence after temporal/disagreement error. The app logs the score and error for both providers. A disagreement lowers the score; it no longer disables Automatic AI or forces a manual-mode transition.

## Failure policy

Keep Automatic AI online when:

- The API request times out
- The API returns invalid JSON
- No presenter is found from one provider
- One provider returns an error
- The returned box is outside the frame
- The returned box jumps an implausible distance
- Internet access is unavailable for one provider

Return to manual control only when the user clears tracking, selects Manual control, or uses Assisted tracking and its local tracker loses the target. Automatic AI holds the last safe camera target until another valid provider result arrives.

Never place API keys in source code. Store them in Windows environment variables or a local secrets manager and keep them out of Git.

## Data usage estimate

Each enabled provider makes 720 requests per hour at the five-second cadence. For a 50–150 KB JPEG frame, both providers together upload approximately 72–216 MB per hour, excluding API response data. Actual usage depends on image quality and provider encoding.