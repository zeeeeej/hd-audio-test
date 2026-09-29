from __future__ import annotations

import asyncio
import os
import secrets
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Header, Request
from fastapi.responses import Response

from src.common import MAX_UPLOAD_BYTES, error_response, request_id

ASR_URL = os.getenv("ASR_URL", "http://127.0.0.1:18001").rstrip("/")
TTS_URL = os.getenv("TTS_URL", "http://127.0.0.1:18002").rstrip("/")
VOICE_API_TOKEN = os.getenv("VOICE_API_TOKEN", "")
UPSTREAM_TIMEOUT = float(os.getenv("WORKER_TIMEOUT_SECONDS", "120"))
MAX_QUEUED = int(os.getenv("MAX_QUEUED_REQUESTS", "2"))

app = FastAPI(title="HD Voice Gateway", version="0.1.0")
_gpu_semaphore = asyncio.Semaphore(1)
_admission_lock = asyncio.Lock()
_in_system = 0


def _authorized(header: str | None) -> bool:
    if not VOICE_API_TOKEN:
        return False
    scheme, _, token = (header or "").partition(" ")
    return scheme.lower() == "bearer" and secrets.compare_digest(token, VOICE_API_TOKEN)


@asynccontextmanager
async def _gpu_slot():
    global _in_system
    async with _admission_lock:
        if _in_system >= 1 + MAX_QUEUED:
            raise RuntimeError("queue_full")
        _in_system += 1
    try:
        async with _gpu_semaphore:
            yield
    finally:
        async with _admission_lock:
            _in_system -= 1


async def _proxy(request: Request, target: str, req_id: str) -> Response:
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_UPLOAD_BYTES + 65536:
                return error_response(413, "request_too_large", "request exceeds size limit", req_id)
        except ValueError:
            return error_response(400, "invalid_content_length", "invalid Content-Length header", req_id)
    body = await request.body()
    if len(body) > MAX_UPLOAD_BYTES + 65536:
        return error_response(413, "request_too_large", "request exceeds size limit", req_id)
    try:
        async with _gpu_slot():
            async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
                upstream = await client.request(
                    request.method,
                    target,
                    content=body,
                    headers={
                        "Content-Type": request.headers.get("content-type", "application/octet-stream"),
                        "X-Request-ID": req_id,
                    },
                )
        headers = {"X-Request-ID": upstream.headers.get("x-request-id", req_id)}
        if disposition := upstream.headers.get("content-disposition"):
            headers["Content-Disposition"] = disposition
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/octet-stream"),
            headers=headers,
        )
    except RuntimeError as exc:
        if str(exc) == "queue_full":
            return error_response(503, "queue_full", "voice queue is full", req_id)
        raise
    except httpx.TimeoutException:
        return error_response(504, "worker_timeout", "model worker timed out", req_id)
    except httpx.HTTPError:
        return error_response(502, "worker_unavailable", "model worker is unavailable", req_id)


@app.get("/health/live")
async def live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ready")
async def ready() -> Response:
    req_id = request_id()
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            asr, tts = await asyncio.gather(
                client.get(f"{ASR_URL}/health/ready"), client.get(f"{TTS_URL}/health/ready")
            )
        if asr.status_code == 200 and tts.status_code == 200:
            return Response('{"status":"ready"}', media_type="application/json")
    except httpx.HTTPError:
        pass
    return error_response(503, "workers_not_ready", "model workers are not ready", req_id)


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    request: Request,
    authorization: str | None = Header(default=None),
    x_request_id: str | None = Header(default=None),
) -> Response:
    req_id = request_id(x_request_id)
    if not _authorized(authorization):
        return error_response(401, "unauthorized", "invalid voice API token", req_id)
    return await _proxy(request, f"{ASR_URL}/transcribe", req_id)


@app.post("/v1/audio/speech")
async def speech(
    request: Request,
    authorization: str | None = Header(default=None),
    x_request_id: str | None = Header(default=None),
) -> Response:
    req_id = request_id(x_request_id)
    if not _authorized(authorization):
        return error_response(401, "unauthorized", "invalid voice API token", req_id)
    return await _proxy(request, f"{TTS_URL}/speech", req_id)
