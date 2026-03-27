"""
Minimal Gemini API smoke test.

Usage:
  python google_studio_test.py "Write a 1-sentence summary of the Great Gatsby."

Auth:
  - API key mode: set `GOOGLE_API_KEY` (or `GEMINI_API_KEY`)
  - Vertex mode: set `GOOGLE_CLOUD_PROJECT` (and optionally `VERTEX_LOCATION`)
"""

import os
import sys
import argparse
from typing import Any


def main() -> None:
    try:
        from google import genai
    except ImportError as e:
        raise SystemExit("Missing dependency. Install with: pip install google-genai") from e

    parser = argparse.ArgumentParser(description="Minimal Gemini API smoke test.")
    parser.add_argument(
        "--model",
        default=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        help="Gemini model name (e.g. gemini-2.5-flash).",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Say hello and explain what you are in one sentence.",
        help="Prompt to send to the model.",
    )
    args = parser.parse_args()

    model_name = args.model
    prompt = args.prompt

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if api_key:
        client = genai.Client(api_key=api_key)
    else:
        project = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("VERTEX_LOCATION", "us-central1")
        if not project:
            raise SystemExit(
                "No API key found. Set GOOGLE_API_KEY (or GEMINI_API_KEY), "
                "or set GOOGLE_CLOUD_PROJECT for Vertex AI."
            )
        client = genai.Client(vertexai=True, project=project, location=location)

    temperature: float = float(os.getenv("GEMINI_TEMPERATURE", "0.7"))
    max_output_tokens: int = int(os.getenv("GEMINI_MAX_TOKENS", "256"))
    config: Any = {
        "temperature": temperature,
        "maxOutputTokens": max_output_tokens,
        "candidateCount": 1,
    }

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=config,
    )

    # `google-genai` responses expose the generated text at `.text`
    print(response.text)


if __name__ == "__main__":
    main()

