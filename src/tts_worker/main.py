from __future__ import annotations

import io
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
import soundfile as sf
from fastapi import FastAPI, Header
from fastapi.exceptions import RequestValidationError
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field

from src.common import (
    CHINESE_VOICES,
    MAX_TTS_CHARS,
    error_response,
    request_id,
    validation_error_handler,
)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("voice.tts")

MODEL_ID = os.getenv("TTS_MODEL_PATH", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
LOAD_MODEL_ON_START = os.getenv("LOAD_MODEL_ON_START", "1") == "1"
TTS_DTYPE = os.getenv("TTS_DTYPE", "float32").strip().lower()


class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: str = Field(min_length=1, max_length=MAX_TTS_CHARS)
    voice: str = "Vivian"
    language: str = "Chinese"
    instruct: str = Field(default="", max_length=100)


class TTSRuntime:
    def __init__(self) -> None:
        self.model = None
        self.error: str | None = None

    def load(self) -> None:
        import torch
        from qwen_tts import Qwen3TTSModel

        supported_dtypes = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        if TTS_DTYPE not in supported_dtypes:
            raise ValueError(f"unsupported TTS_DTYPE: {TTS_DTYPE}")
        dtype = supported_dtypes[TTS_DTYPE]
        logger.info("loading TTS model=%s dtype=%s", MODEL_ID, TTS_DTYPE)
        self.model = Qwen3TTSModel.from_pretrained(
            MODEL_ID,
            device_map="cuda:0",
            dtype=dtype,
            attn_implementation="sdpa",
        )
        self.error = None
        logger.info("TTS model ready")

    def synthesize(self, payload: SpeechRequest) -> bytes:
        if self.model is None:
            raise RuntimeError("TTS model is not loaded")
        kwargs: dict[str, object] = {
            "text": payload.input,
            "language": payload.language,
            "speaker": payload.voice,
        }
        if payload.instruct:
            kwargs["instruct"] = payload.instruct
        wavs, sample_rate = self.model.generate_custom_voice(**kwargs)
        buffer = io.BytesIO()
        sf.write(buffer, wavs[0], sample_rate, format="WAV", subtype="PCM_16")
        return buffer.getvalue()


runtime = TTSRuntime()


@asynccontextmanager
async def lifespan(_: FastAPI):
    if LOAD_MODEL_ON_START:
        try:
            await anyio.to_thread.run_sync(runtime.load)
        except Exception as exc:
            runtime.error = f"{type(exc).__name__}: {exc}"
            logger.exception("failed to load TTS model")
            raise
    yield


app = FastAPI(title="HD Qwen3 TTS Worker", version="0.1.0", lifespan=lifespan)
app.add_exception_handler(RequestValidationError, validation_error_handler)


@app.get("/health/live")
async def live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ready")
async def ready() -> Response:
    if runtime.model is not None or not LOAD_MODEL_ON_START:
        return Response('{"status":"ready"}', media_type="application/json")
    return error_response(503, "model_not_ready", runtime.error or "TTS model is not ready", request_id())


@app.post("/speech")
async def speech(
    payload: SpeechRequest,
    x_request_id: str | None = Header(default=None),
) -> Response:
    req_id = request_id(x_request_id)
    started = time.perf_counter()
    if runtime.model is None:
        return error_response(503, "model_not_ready", "TTS model is not ready", req_id)
    if payload.voice not in CHINESE_VOICES:
        return error_response(422, "invalid_voice", "unsupported Chinese voice", req_id)
    try:
        audio = await anyio.to_thread.run_sync(runtime.synthesize, payload)
    except Exception:
        logger.exception("TTS inference failed request_id=%s", req_id)
        return error_response(500, "tts_failed", "speech synthesis failed", req_id)
    elapsed = round((time.perf_counter() - started) * 1000)
    logger.info("TTS complete request_id=%s inference_ms=%d", req_id, elapsed)
    return Response(
        content=audio,
        media_type="audio/wav",
        headers={
            "X-Request-ID": req_id,
            "X-Inference-MS": str(elapsed),
            "X-Model": Path(MODEL_ID).name,
            "Content-Disposition": 'inline; filename="speech.wav"',
        },
    )
