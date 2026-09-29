from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
import soundfile as sf
from fastapi import FastAPI, Form, Header, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import Response

from src.common import (
    MAX_AUDIO_SECONDS,
    MAX_UPLOAD_BYTES,
    error_response,
    hotword_prompt,
    is_allowed_audio_type,
    parse_hotwords,
    request_id,
    validation_error_handler,
)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("voice.asr")

MODEL_ID = os.getenv("ASR_MODEL_PATH", "Qwen/Qwen3-ASR-0.6B-hf")
ADAPTER_PATH = os.getenv("ASR_ADAPTER_PATH", "").strip()
LOAD_MODEL_ON_START = os.getenv("LOAD_MODEL_ON_START", "1") == "1"
FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")


class ASRRuntime:
    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self.error: str | None = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        logger.info("loading ASR model=%s", MODEL_ID)
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            MODEL_ID,
            dtype=torch.float16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        ).to("cuda:0")
        if ADAPTER_PATH:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, ADAPTER_PATH)
        self.model.eval()
        self.error = None
        logger.info("ASR model ready")

    def transcribe(self, wav_path: str, language: str, prompt: str | None) -> str:
        if self.model is None or self.processor is None:
            raise RuntimeError("ASR model is not loaded")
        import torch

        audio, sample_rate = sf.read(wav_path, dtype="float32", always_2d=False)
        if sample_rate != 16000:
            raise ValueError(f"expected 16000 Hz audio, got {sample_rate} Hz")
        kwargs: dict[str, object] = {"audio": audio, "language": language}
        if prompt:
            kwargs["prompt"] = prompt
        inputs = self.processor.apply_transcription_request(**kwargs)
        inputs = inputs.to("cuda:0", torch.float16)
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
        generated = output_ids[:, inputs["input_ids"].shape[1] :]
        return self.processor.decode(generated, return_format="transcription_only")[0].strip()


runtime = ASRRuntime()


@asynccontextmanager
async def lifespan(_: FastAPI):
    if LOAD_MODEL_ON_START:
        try:
            await anyio.to_thread.run_sync(runtime.load)
        except Exception as exc:
            runtime.error = f"{type(exc).__name__}: {exc}"
            logger.exception("failed to load ASR model")
            raise
    yield


app = FastAPI(title="HD Qwen3 ASR Worker", version="0.1.0", lifespan=lifespan)
app.add_exception_handler(RequestValidationError, validation_error_handler)


def _convert_to_wav(source: Path, output: Path) -> float:
    command = [
        FFMPEG_BIN,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-t",
        str(MAX_AUDIO_SECONDS + 1),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=90, check=False)
    if completed.returncode != 0:
        message = completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else "decode failed"
        raise ValueError(f"invalid or unsupported audio: {message[:200]}")
    info = sf.info(output)
    return float(info.frames) / float(info.samplerate)


@app.get("/health/live")
async def live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ready")
async def ready() -> Response:
    if runtime.model is not None or not LOAD_MODEL_ON_START:
        return Response('{"status":"ready"}', media_type="application/json")
    return error_response(503, "model_not_ready", runtime.error or "ASR model is not ready", request_id())


@app.post("/transcribe")
async def transcribe(
    file: UploadFile,
    language: str = Form(default="Chinese"),
    hotwords: str = Form(default="[]"),
    x_request_id: str | None = Header(default=None),
) -> Response:
    req_id = request_id(x_request_id)
    started = time.perf_counter()
    if runtime.model is None:
        return error_response(503, "model_not_ready", "ASR model is not ready", req_id)
    if not is_allowed_audio_type(file.content_type):
        return error_response(415, "unsupported_audio", "unsupported audio content type", req_id)
    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    if not payload:
        return error_response(400, "empty_audio", "audio file is empty", req_id)
    if len(payload) > MAX_UPLOAD_BYTES:
        return error_response(413, "audio_too_large", "audio exceeds 20 MB", req_id)
    try:
        words = parse_hotwords(hotwords)
    except ValueError as exc:
        return error_response(422, "invalid_hotwords", str(exc), req_id)

    suffix = Path(file.filename or "recording").suffix[:10] or ".bin"
    try:
        with tempfile.TemporaryDirectory(prefix="voice-asr-") as tmp:
            source = Path(tmp) / f"input{suffix}"
            wav = Path(tmp) / "normalized.wav"
            source.write_bytes(payload)
            duration = await anyio.to_thread.run_sync(_convert_to_wav, source, wav)
            if duration <= 0:
                return error_response(400, "empty_audio", "audio contains no samples", req_id)
            if duration > MAX_AUDIO_SECONDS:
                return error_response(
                    422, "audio_too_long", f"audio exceeds {MAX_AUDIO_SECONDS} seconds", req_id
                )
            text = await anyio.to_thread.run_sync(
                runtime.transcribe, str(wav), language, hotword_prompt(words)
            )
    except subprocess.TimeoutExpired:
        return error_response(422, "audio_decode_timeout", "audio decoding timed out", req_id)
    except ValueError as exc:
        return error_response(422, "invalid_audio", str(exc), req_id)
    except Exception:
        logger.exception("ASR inference failed request_id=%s", req_id)
        return error_response(500, "asr_failed", "speech recognition failed", req_id)

    elapsed = round((time.perf_counter() - started) * 1000)
    logger.info("ASR complete request_id=%s inference_ms=%d", req_id, elapsed)
    return Response(
        json.dumps(
            {
                "text": text,
                "language": language,
                "model": Path(MODEL_ID).name,
                "inference_ms": elapsed,
                "request_id": req_id,
            },
            ensure_ascii=False,
        ),
        media_type="application/json",
        headers={"X-Request-ID": req_id},
    )
