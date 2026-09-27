r"""Standalone Gemini/Groq vision test for the Automatic Camera project.

Examples from a VS Code terminal:

    python test_ai_models.py --image path\to\photo.jpg
    python test_ai_models.py --image path\to\photo.jpg --provider Gemini
    python test_ai_models.py --camera 0

The script never prints API keys. It uses the same provider adapters and
structured-response validation as the desktop application.
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import vision_providers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test Gemini and Groq presenter detection independently."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--image",
        type=Path,
        help="Path to a JPG, PNG, or other image file.",
    )
    source.add_argument(
        "--camera",
        type=int,
        metavar="INDEX",
        help="Capture one frame from a webcam, for example --camera 0.",
    )
    parser.add_argument(
        "--provider",
        choices=("Gemini", "Groq", "both"),
        default="both",
        help="Provider to test. Default: both.",
    )
    return parser.parse_args()


def jpeg_from_image_file(path: Path) -> bytes:
    """Read any common image format and convert it to JPEG bytes."""
    if not path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {path}")

    if path.suffix.lower() in {".jpg", ".jpeg"}:
        data = path.read_bytes()
        if not data:
            raise RuntimeError(f"Image file is empty: {path}")
        return data

    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(
            "Pillow is required for image files. Run: python -m pip install pillow"
        ) from error

    try:
        with Image.open(path) as image:
            output = io.BytesIO()
            image.convert("RGB").save(output, format="JPEG", quality=85)
            return output.getvalue()
    except Exception as error:
        raise RuntimeError(f"Could not read image {path}: {error}") from error


def jpeg_from_camera(index: int) -> bytes:
    """Capture one frame from a local webcam and encode it as JPEG."""
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "OpenCV is required for webcam testing. Run: "
            "python -m pip install opencv-contrib-python"
        ) from error

    capture = cv2.VideoCapture(index)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open webcam index {index}.")
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"Webcam index {index} returned no frame.")
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 85],
        )
        if not ok:
            raise RuntimeError("OpenCV could not encode the webcam frame.")
        return encoded.tobytes()
    finally:
        capture.release()


def enabled_providers(selection: str) -> tuple[str, ...]:
    if selection == "both":
        return ("Gemini", "Groq")
    return (selection,)


def run_provider_test(provider_name: str, jpeg_bytes: bytes) -> dict[str, object]:
    model = vision_providers.provider_model(provider_name)
    started = time.perf_counter()
    try:
        result = vision_providers.create_provider(provider_name).locate_presenter(
            jpeg_bytes
        )
    except Exception as error:
        return {
            "provider": provider_name,
            "model": model,
            "ok": False,
            "seconds": time.perf_counter() - started,
            "error": str(error),
        }

    return {
        "provider": provider_name,
        "model": model,
        "ok": True,
        "seconds": time.perf_counter() - started,
        "result": result,
    }


def print_result(result: dict[str, object]) -> None:
    provider = result["provider"]
    model = result["model"]
    seconds = result["seconds"]
    print(f"\n[{provider}] model: {model}")
    print(f"[{provider}] response time: {seconds:.2f}s")

    if not result["ok"]:
        print(f"[{provider}] ERROR: {result['error']}")
        if "403" in str(result["error"]) or "401" in str(result["error"]):
            print(
                f"[{provider}] Check the {provider.upper()}_API_KEY and "
                "whether the configured model is available to that account."
            )
        return

    detection = result["result"]
    if not isinstance(detection, vision_providers.VisionResult):
        print(f"[{provider}] ERROR: adapter returned an unexpected result.")
        return

    print(f"[{provider}] found: {detection.found}")
    print(f"[{provider}] label: {detection.label}")
    print(f"[{provider}] confidence: {detection.confidence:.0%}")
    print(f"[{provider}] box (normalized 0..1000): {detection.box}")


def main() -> int:
    args = parse_args()

    try:
        if args.image is not None:
            jpeg_bytes = jpeg_from_image_file(args.image)
            source_description = str(args.image)
        else:
            jpeg_bytes = jpeg_from_camera(args.camera)
            source_description = f"webcam index {args.camera}"
    except Exception as error:
        print(f"INPUT ERROR: {error}", file=sys.stderr)
        return 2

    providers = enabled_providers(args.provider)
    print("Automatic Camera AI model test")
    print(f"Source: {source_description}")
    print(f"Testing: {', '.join(providers)}")
    print("Requests are running independently; API keys are not displayed.")

    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=len(providers)) as executor:
        pending = {
            executor.submit(run_provider_test, provider, jpeg_bytes): provider
            for provider in providers
        }
        for future in as_completed(pending):
            result = future.result()
            results.append(result)
            print_result(result)

    passed = sum(1 for result in results if result["ok"])
    print(f"\nSummary: {passed}/{len(results)} provider request(s) succeeded.")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())