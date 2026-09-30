from __future__ import annotations

import json
import os
import uuid
from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))
MAX_AUDIO_SECONDS = int(os.getenv("MAX_AUDIO_SECONDS", "60"))
MAX_HOTWORDS = int(os.getenv("MAX_HOTWORDS", "100"))
MAX_HOTWORD_LENGTH = int(os.getenv("MAX_HOTWORD_LENGTH", "40"))
MAX_TTS_CHARS = int(os.getenv("MAX_TTS_CHARS", "500"))
TTS_STREAM_MAX_SENTENCE_CHARS = int(os.getenv("TTS_STREAM_MAX_SENTENCE_CHARS", "60"))

if not 20 <= TTS_STREAM_MAX_SENTENCE_CHARS <= 200:
    raise ValueError("TTS_STREAM_MAX_SENTENCE_CHARS must be between 20 and 200")

CHINESE_VOICES = ("Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric")
ALLOWED_AUDIO_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/mp4",
    "audio/x-m4a",
    "audio/webm",
    "video/webm",
    "application/octet-stream",
}


def is_allowed_audio_type(value: str | None) -> bool:
    media_type = (value or "").split(";", 1)[0].strip().lower()
    return media_type in ALLOWED_AUDIO_TYPES


def request_id(value: str | None = None) -> str:
    candidate = (value or "").strip()
    return candidate[:128] if candidate else str(uuid.uuid4())


def error_response(status: int, code: str, message: str, req_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message, "request_id": req_id}},
        headers={"X-Request-ID": req_id},
    )


def parse_hotwords(raw: str | list[str] | None) -> list[str]:
    if raw is None or raw == "":
        return []
    if isinstance(raw, str):
        try:
            value: Any = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("hotwords must be a JSON array of strings") from exc
    else:
        value = raw
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("hotwords must be a JSON array of strings")

    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        word = item.strip()
        if not word:
            continue
        if len(word) > MAX_HOTWORD_LENGTH:
            raise ValueError(f"each hotword must be <= {MAX_HOTWORD_LENGTH} characters")
        if word not in seen:
            seen.add(word)
            result.append(word)
    if len(result) > MAX_HOTWORDS:
        raise ValueError(f"at most {MAX_HOTWORDS} hotwords are allowed")
    return result


def hotword_prompt(words: list[str]) -> str | None:
    if not words:
        return None
    return f"Vocabulary: {', '.join(words)}."


def split_tts_sentences(
    text: str,
    max_chars: int = TTS_STREAM_MAX_SENTENCE_CHARS,
) -> list[str]:
    """Split TTS input into ordered, speakable chunks while preserving punctuation."""
    if not 1 <= max_chars <= MAX_TTS_CHARS:
        raise ValueError(f"max_chars must be between 1 and {MAX_TTS_CHARS}")

    normalized = text.strip()
    if not normalized:
        return []

    terminal_marks = set("。！？!?；;…\n")
    secondary_marks = set("，,、：:")
    logical_sentences: list[str] = []
    start = 0
    index = 0

    while index < len(normalized):
        character = normalized[index]
        is_terminal = character in terminal_marks
        if character == ".":
            previous = normalized[index - 1] if index else ""
            following = normalized[index + 1] if index + 1 < len(normalized) else ""
            is_decimal = previous.isdigit() and following.isdigit()
            is_terminal = not is_decimal and (not following or following.isspace())

        if is_terminal:
            end = index + 1
            if character != "\n":
                while end < len(normalized) and normalized[end] in terminal_marks - {"\n"}:
                    end += 1
            sentence = normalized[start:end].strip()
            if sentence:
                logical_sentences.append(sentence)
            start = end
            index = end
            continue
        index += 1

    remainder = normalized[start:].strip()
    if remainder:
        logical_sentences.append(remainder)

    chunks: list[str] = []
    for sentence in logical_sentences:
        remaining = sentence
        while len(remaining) > max_chars:
            window = remaining[:max_chars]
            candidates = [
                position + 1
                for position, character in enumerate(window)
                if character in secondary_marks or character.isspace()
            ]
            useful_candidates = [position for position in candidates if position >= max_chars // 2]
            split_at = useful_candidates[-1] if useful_candidates else max_chars
            chunk = remaining[:split_at].strip()
            if chunk:
                chunks.append(chunk)
            remaining = remaining[split_at:].strip()
        if remaining:
            chunks.append(remaining)
    return chunks


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    req_id = request_id(request.headers.get("x-request-id"))
    first = exc.errors()[0] if exc.errors() else {}
    message = str(first.get("msg", "request validation failed"))
    return error_response(422, "invalid_request", message, req_id)
