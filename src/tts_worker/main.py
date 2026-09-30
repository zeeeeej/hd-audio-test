from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
from fastapi import FastAPI, Header
from fastapi.exceptions import RequestValidationError
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from src.common import (
    CHINESE_VOICES,
    MAX_TTS_CHARS,
    error_response,
    request_id,
    split_tts_sentences,
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

    def synthesize_text(
        self,
        text: str,
        voice: str,
        language: str,
        instruct: str,
    ) -> tuple[bytes, int]:
        if self.model is None:
            raise RuntimeError("TTS model is not loaded")
        import soundfile as sf

        kwargs: dict[str, object] = {
            "text": text,
            "language": language,
            "speaker": voice,
        }
        if instruct:
            kwargs["instruct"] = instruct
        wavs, sample_rate = self.model.generate_custom_voice(**kwargs)
        buffer = io.BytesIO()
        sf.write(buffer, wavs[0], sample_rate, format="WAV", subtype="PCM_16")
        return buffer.getvalue(), int(sample_rate)

    def synthesize(self, payload: SpeechRequest) -> tuple[bytes, int]:
        return self.synthesize_text(
            payload.input,
            payload.voice,
            payload.language,
            payload.instruct,
        )


runtime = TTSRuntime()
_generation_lock = asyncio.Lock()


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


def _event_line(event: dict[str, object]) -> bytes:
    return (json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n").encode()


async def _speech_events(
    payload: SpeechRequest,
    req_id: str,
) -> AsyncIterator[bytes]:
    sentences = split_tts_sentences(payload.input)
    total_inference_ms = 0
    yield _event_line(
        {
            "type": "start",
            "request_id": req_id,
            "sentence_count": len(sentences),
        }
    )

    for index, sentence in enumerate(sentences):
        started = time.perf_counter()
        try:
            async with _generation_lock:
                audio, sample_rate = await anyio.to_thread.run_sync(
                    runtime.synthesize_text,
                    sentence,
                    payload.voice,
                    payload.language,
                    payload.instruct,
                )
        except Exception:
            logger.exception(
                "streaming TTS failed request_id=%s sentence_index=%d",
                req_id,
                index,
            )
            yield _event_line(
                {
                    "type": "error",
                    "index": index,
                    "error": {
                        "code": "tts_failed",
                        "message": "speech synthesis failed",
                        "request_id": req_id,
                    },
                }
            )
            return

        inference_ms = round((time.perf_counter() - started) * 1000)
        total_inference_ms += inference_ms
        logger.info(
            "TTS sentence complete request_id=%s sentence_index=%d inference_ms=%d",
            req_id,
            index,
            inference_ms,
        )
        yield _event_line(
            {
                "type": "audio",
                "index": index,
                "text": sentence,
                "media_type": "audio/wav",
                "sample_rate": sample_rate,
                "inference_ms": inference_ms,
                "audio_base64": base64.b64encode(audio).decode("ascii"),
            }
        )

    yield _event_line(
        {
            "type": "done",
            "sentence_count": len(sentences),
            "total_inference_ms": total_inference_ms,
        }
    )


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
        async with _generation_lock:
            audio, _ = await anyio.to_thread.run_sync(runtime.synthesize, payload)
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


@app.post("/speech/stream")
async def speech_stream(
    payload: SpeechRequest,
    x_request_id: str | None = Header(default=None),
) -> Response:
    req_id = request_id(x_request_id)
    if runtime.model is None:
        return error_response(503, "model_not_ready", "TTS model is not ready", req_id)
    if payload.voice not in CHINESE_VOICES:
        return error_response(422, "invalid_voice", "unsupported Chinese voice", req_id)
    return StreamingResponse(
        _speech_events(payload, req_id),
        media_type="application/x-ndjson; charset=utf-8",
        headers={
            "X-Request-ID": req_id,
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
