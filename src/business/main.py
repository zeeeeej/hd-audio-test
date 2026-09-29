from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
from fastapi import FastAPI, Form, Header, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from src.common import (
    CHINESE_VOICES,
    MAX_TTS_CHARS,
    MAX_UPLOAD_BYTES,
    error_response,
    is_allowed_audio_type,
    parse_hotwords,
    request_id,
    validation_error_handler,
)

VOICE_BASE_URL = os.getenv("VOICE_BASE_URL", "http://127.0.0.1:18000").rstrip("/")
VOICE_API_TOKEN = os.getenv("VOICE_API_TOKEN", "")
VOICE_TIMEOUT_SECONDS = float(os.getenv("VOICE_TIMEOUT_SECONDS", "120"))
DEFAULT_HOTWORDS_RAW = os.getenv("DEFAULT_HOTWORDS", "[]")
STATIC_DIR = Path(__file__).with_name("static")

app = FastAPI(title="HD Voice Business Demo", version="0.1.0")
app.add_exception_handler(RequestValidationError, validation_error_handler)


class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: str = Field(min_length=1, max_length=MAX_TTS_CHARS)
    voice: str = "Vivian"
    language: str = "Chinese"
    instruct: str = Field(default="", max_length=100)


def _auth_headers(req_id: str) -> dict[str, str]:
    headers = {"X-Request-ID": req_id}
    if VOICE_API_TOKEN:
        headers["Authorization"] = f"Bearer {VOICE_API_TOKEN}"
    return headers


def _combined_hotwords(raw: str) -> list[str]:
    defaults = parse_hotwords(DEFAULT_HOTWORDS_RAW)
    requested = parse_hotwords(raw)
    return parse_hotwords(defaults + requested)


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/voices")
async def voices() -> dict[str, object]:
    return {"default": "Vivian", "voices": list(CHINESE_VOICES)}


@app.get("/health/live")
async def live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ready")
async def ready() -> Response:
    req_id = request_id()
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            upstream = await client.get(f"{VOICE_BASE_URL}/health/ready")
        if upstream.status_code != 200:
            return error_response(503, "voice_not_ready", "voice service is not ready", req_id)
        return Response(
            content=json.dumps({"status": "ready"}),
            media_type="application/json",
            headers={"X-Request-ID": req_id},
        )
    except httpx.HTTPError:
        return error_response(503, "voice_unavailable", "voice service is unavailable", req_id)


@app.post("/api/transcriptions")
async def transcriptions(
    file: UploadFile,
    language: str = Form(default="Chinese"),
    hotwords: str = Form(default="[]"),
    x_request_id: str | None = Header(default=None),
) -> Response:
    req_id = request_id(x_request_id)
    if not is_allowed_audio_type(file.content_type):
        return error_response(415, "unsupported_audio", "unsupported audio content type", req_id)
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        return error_response(400, "empty_audio", "audio file is empty", req_id)
    if len(data) > MAX_UPLOAD_BYTES:
        return error_response(413, "audio_too_large", "audio exceeds 20 MB", req_id)
    try:
        merged_hotwords = _combined_hotwords(hotwords)
    except ValueError as exc:
        return error_response(422, "invalid_hotwords", str(exc), req_id)

    try:
        async with httpx.AsyncClient(timeout=VOICE_TIMEOUT_SECONDS) as client:
            upstream = await client.post(
                f"{VOICE_BASE_URL}/v1/audio/transcriptions",
                headers=_auth_headers(req_id),
                files={"file": (file.filename or "recording", data, file.content_type)},
                data={"language": language, "hotwords": json.dumps(merged_hotwords, ensure_ascii=False)},
            )
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/json"),
            headers={"X-Request-ID": upstream.headers.get("x-request-id", req_id)},
        )
    except httpx.TimeoutException:
        return error_response(504, "voice_timeout", "voice service timed out", req_id)
    except httpx.HTTPError:
        return error_response(502, "voice_unavailable", "voice service is unavailable", req_id)


@app.post("/api/speech")
async def speech(
    payload: SpeechRequest,
    x_request_id: str | None = Header(default=None),
) -> Response:
    req_id = request_id(x_request_id)
    if payload.voice not in CHINESE_VOICES:
        return error_response(422, "invalid_voice", "unsupported Chinese voice", req_id)
    try:
        async with httpx.AsyncClient(timeout=VOICE_TIMEOUT_SECONDS) as client:
            upstream = await client.post(
                f"{VOICE_BASE_URL}/v1/audio/speech",
                headers=_auth_headers(req_id),
                json=payload.model_dump(),
            )
        response_headers = {"X-Request-ID": upstream.headers.get("x-request-id", req_id)}
        if disposition := upstream.headers.get("content-disposition"):
            response_headers["Content-Disposition"] = disposition
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/octet-stream"),
            headers=response_headers,
        )
    except httpx.TimeoutException:
        return error_response(504, "voice_timeout", "voice service timed out", req_id)
    except httpx.HTTPError:
        return error_response(502, "voice_unavailable", "voice service is unavailable", req_id)
